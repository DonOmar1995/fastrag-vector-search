#!/usr/bin/env bash
# ===========================================================================
# benchmarks/run_benchmarks.sh
# FastRAG-Vector-Search – Benchmark Runner
# ===========================================================================
#
# Usage:
#   chmod +x benchmarks/run_benchmarks.sh
#   ./benchmarks/run_benchmarks.sh [--small] [--gpu]
#
# Flags:
#   --small   Use a small 10,000-document index (fast, for CI / CPU machines)
#   --gpu     Force GPU mode (requires CUDA)
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Defaults
NUM_DOCS=100000
DEVICE="auto"

# Parse flags
for arg in "$@"; do
  case $arg in
    --small) NUM_DOCS=10000 ;;
    --gpu)   DEVICE="cuda"  ;;
  esac
done

echo "============================================================"
echo "  FastRAG-Vector-Search – Benchmark Suite"
echo "============================================================"
echo "  Index size : ${NUM_DOCS} documents"
echo "  Device     : ${DEVICE}"
echo "------------------------------------------------------------"

cd "$REPO_ROOT"

python - <<EOF
import torch
from fastrag_vector_search import RetrievalConfig, VectorRetrievalBenchmark, validate_accuracy

device_arg = "${DEVICE}"
if device_arg == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = device_arg

config = RetrievalConfig(
    embedding_dim=1536,
    num_documents=${NUM_DOCS},
    top_k=5,
    batch_size=32,
    use_fp16=(device == "cuda"),
    device=device,
    warmup_iterations=3,
    benchmark_iterations=10,
)

bench = VectorRetrievalBenchmark(config)
bench.benchmark_baseline()
bench.benchmark_optimized()
bench.generate_comparison_report()
validate_accuracy(config)
EOF

echo ""
echo "  ✅ Benchmarks complete!"
echo "  📄 Results saved to benchmarks/results/"
