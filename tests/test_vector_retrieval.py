"""
Unit tests for FastRAG-Vector-Search
=====================================
Run with:  pytest tests/test_vector_retrieval.py -v
"""

import pytest
import torch
import torch.nn.functional as F

from fastrag_vector_search import (
    RetrievalConfig,
    BaselineVectorRetrieval,
    OptimizedVectorRetrieval,
    VectorRetrievalBenchmark,
    validate_accuracy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cpu_config():
    return RetrievalConfig(
        embedding_dim=128,
        num_documents=500,
        top_k=5,
        batch_size=8,
        use_fp16=False,
        device="cpu",
        warmup_iterations=1,
        benchmark_iterations=2,
    )


@pytest.fixture
def baseline(cpu_config):
    return BaselineVectorRetrieval(cpu_config)


@pytest.fixture
def optimized(cpu_config):
    return OptimizedVectorRetrieval(cpu_config)


# ---------------------------------------------------------------------------
# RetrievalConfig tests
# ---------------------------------------------------------------------------

class TestRetrievalConfig:
    def test_defaults(self):
        cfg = RetrievalConfig()
        assert cfg.embedding_dim == 1536
        assert cfg.num_documents == 100_000
        assert cfg.top_k == 5

    def test_custom(self):
        cfg = RetrievalConfig(embedding_dim=768, num_documents=1000, top_k=10)
        assert cfg.embedding_dim == 768
        assert cfg.num_documents == 1000
        assert cfg.top_k == 10


# ---------------------------------------------------------------------------
# BaselineVectorRetrieval tests
# ---------------------------------------------------------------------------

class TestBaselineVectorRetrieval:
    def test_index_shape(self, baseline, cpu_config):
        assert baseline.document_embeddings.shape == (
            cpu_config.num_documents,
            cpu_config.embedding_dim,
        )

    def test_index_normalised(self, baseline):
        norms = baseline.document_embeddings.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_retrieve_output_shape(self, baseline, cpu_config):
        queries = torch.randn(cpu_config.batch_size, cpu_config.embedding_dim)
        scores, indices = baseline.retrieve(queries)
        assert scores.shape == (cpu_config.batch_size, cpu_config.top_k)
        assert indices.shape == (cpu_config.batch_size, cpu_config.top_k)

    def test_scores_in_range(self, baseline, cpu_config):
        queries = torch.randn(cpu_config.batch_size, cpu_config.embedding_dim)
        scores, _ = baseline.retrieve(queries)
        assert (scores >= -1.0).all()
        assert (scores <= 1.0 + 1e-5).all()

    def test_indices_in_range(self, baseline, cpu_config):
        queries = torch.randn(cpu_config.batch_size, cpu_config.embedding_dim)
        _, indices = baseline.retrieve(queries)
        assert (indices >= 0).all()
        assert (indices < cpu_config.num_documents).all()

    def test_scores_descending(self, baseline, cpu_config):
        queries = torch.randn(cpu_config.batch_size, cpu_config.embedding_dim)
        scores, _ = baseline.retrieve(queries)
        for row in scores:
            assert (row[:-1] >= row[1:]).all()

    def test_memory_bytes(self, baseline, cpu_config):
        expected = cpu_config.num_documents * cpu_config.embedding_dim * 4  # FP32
        assert baseline.memory_bytes() == expected


# ---------------------------------------------------------------------------
# OptimizedVectorRetrieval tests
# ---------------------------------------------------------------------------

class TestOptimizedVectorRetrieval:
    def test_index_shape(self, optimized, cpu_config):
        assert optimized.document_embeddings.shape == (
            cpu_config.num_documents,
            cpu_config.embedding_dim,
        )

    def test_index_contiguous(self, optimized):
        assert optimized.document_embeddings.is_contiguous()

    def test_retrieve_output_shape(self, optimized, cpu_config):
        queries = torch.randn(cpu_config.batch_size, cpu_config.embedding_dim)
        scores, indices = optimized.retrieve_optimized(queries)
        assert scores.shape == (cpu_config.batch_size, cpu_config.top_k)
        assert indices.shape == (cpu_config.batch_size, cpu_config.top_k)

    def test_output_is_float32(self, optimized, cpu_config):
        """Scores must always be FP32 regardless of index precision."""
        queries = torch.randn(cpu_config.batch_size, cpu_config.embedding_dim)
        scores, _ = optimized.retrieve_optimized(queries)
        assert scores.dtype == torch.float32

    def test_scores_descending(self, optimized, cpu_config):
        queries = torch.randn(cpu_config.batch_size, cpu_config.embedding_dim)
        scores, _ = optimized.retrieve_optimized(queries)
        for row in scores:
            assert (row[:-1] >= row[1:]).all()

    def test_memory_bytes_fp32(self, cpu_config):
        cfg = RetrievalConfig(**{**cpu_config.__dict__, "use_fp16": False})
        eng = OptimizedVectorRetrieval(cfg)
        expected = cfg.num_documents * cfg.embedding_dim * 4
        assert eng.memory_bytes() == expected


# ---------------------------------------------------------------------------
# Accuracy validation tests
# ---------------------------------------------------------------------------

class TestAccuracyValidation:
    def test_top1_agreement_fp32(self, cpu_config):
        """FP32 baseline and FP32 optimised should perfectly agree."""
        cfg = RetrievalConfig(**{**cpu_config.__dict__, "use_fp16": False})
        agreement = validate_accuracy(cfg, num_queries=50)
        assert agreement >= 0.95  # Should be essentially 1.0, allow small float gap

    def test_top1_agreement_returns_float(self, cpu_config):
        result = validate_accuracy(cpu_config, num_queries=20)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------

class TestVectorRetrievalBenchmark:
    def test_benchmark_baseline_returns_dict(self, cpu_config):
        bench = VectorRetrievalBenchmark(cpu_config)
        results = bench.benchmark_baseline(num_queries=8, num_iterations=2)
        for key in ("latency_mean_ms", "throughput_qps", "index_memory_mb"):
            assert key in results
            assert results[key] > 0

    def test_benchmark_optimized_returns_dict(self, cpu_config):
        bench = VectorRetrievalBenchmark(cpu_config)
        results = bench.benchmark_optimized(num_queries=8, num_iterations=2)
        for key in ("latency_mean_ms", "throughput_qps", "index_memory_mb"):
            assert key in results
            assert results[key] > 0

    def test_comparison_report_runs(self, cpu_config):
        bench = VectorRetrievalBenchmark(cpu_config)
        bench.benchmark_baseline(num_queries=8, num_iterations=2)
        bench.benchmark_optimized(num_queries=8, num_iterations=2)
        bench.generate_comparison_report()  # Should not raise

    def test_report_raises_before_benchmarks(self, cpu_config):
        bench = VectorRetrievalBenchmark(cpu_config)
        with pytest.raises(RuntimeError):
            bench.generate_comparison_report()
