# FastRAG-Vector-Search — Experiment Report

> **Experiment date:** 2026-04-12 19:22 UTC  
> **Device:** CPU  
> **Index:** 50,000 documents × 768 dimensions  
> **Batch size:** 32 queries | **Iterations:** 8

---

## Experiment Design

Two retrieval approaches were benchmarked against the same embedding index:

| | **WITHOUT Optimisation** | **WITH Optimisation** |
|---|---|---|
| Index precision | FP32 | **FP16** |
| Query processing | One-at-a-time loop | **Batched** |
| Operations | Separate normalise / matmul / topk | **Fused** |
| Memory layout | Standard | **Contiguous (coalesced)** |

---

## Results

| Metric | WITHOUT | WITH | Δ |
|---|---|---|---|
| Mean latency (ms) | 301.14 | 28.56 | **10.54× faster** |
| Min latency (ms) | 281.47 | 26.35 | |
| p95 latency (ms) | 332.04 | 30.43 | |
| Throughput (QPS) | 106.3 | 1120.5 | **+954.1%** |
| Index memory (MB) | 146.5 | 73.2 | **50.0% savings** |
| Top-1 accuracy | 100% | 99.5% | negligible loss |

---

## Key Findings

- ⚡ **10.54× latency speedup** through batched + fused operations
- 📈 **+954.1% throughput** gain (queries per second)
- 💾 **50.0% memory reduction** via FP16 quantisation
- 🎯 **99.5% top-1 accuracy** maintained (negligible degradation)

---

## Chart

![Benchmark Results](benchmark_chart.png)

---

## Conclusion

The optimised approach delivers substantial improvements across all three dimensions —
speed, throughput, and memory — with effectively zero accuracy cost.
On a GPU these gains would be even more pronounced (tensor cores accelerate FP16 matmul by 2×).

These techniques directly mirror what production vector databases like **Pinecone** and **Milvus** use internally.
