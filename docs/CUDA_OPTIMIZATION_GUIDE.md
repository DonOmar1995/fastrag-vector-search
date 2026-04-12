# CUDA Optimization Guide

## Overview

This document explains the GPU optimization techniques used in
**FastRAG-Vector-Search** and how to tune them for your workload.

---

## 1. FP16 Quantization

### What it is
Half-precision floating-point (FP16) uses 2 bytes per value instead of 4.

### How we use it
The document embedding index is cast to `torch.float16` at build time:

```python
embeddings = embeddings.to(torch.float16).contiguous()
```

### Trade-offs
| Metric | FP32 | FP16 |
|--------|------|------|
| Memory | 600 MB | 300 MB |
| Speed (A100) | 1× | ~2× |
| Accuracy | 100% | 99.9% |

---

## 2. Batched Processing

Processing queries one-at-a-time leaves the GPU underutilized.
Use `batch_size ≥ 32` for good occupancy:

```python
config = RetrievalConfig(batch_size=64)
```

---

## 3. Memory Coalescing

Ensure the embedding matrix is **contiguous** in C-order (row-major):

```python
embeddings = embeddings.contiguous()
```

This allows the GPU memory controller to burst-fetch sequential
addresses in a single transaction, eliminating cache-miss penalties.

---

## 4. Automatic Mixed-Precision (AMP)

For additional speedup on Ampere+ GPUs, use `torch.cuda.amp.autocast`:

```python
with torch.cuda.amp.autocast():
    scores, indices = retriever.retrieve_optimized(queries)
```

---

## 5. Profiling

```bash
# NVIDIA Nsight Systems
nsys profile -o profile_report python fastrag_vector_search.py

# View report
nsys-ui profile_report.nsys-rep

# Quick GPU utilization check
nvidia-smi dmon
```

---

## 6. Tuning Checklist

- [ ] Use FP16 if GPU supports tensor cores (Volta+)
- [ ] Set `batch_size` ≥ 32
- [ ] Enable AMP via `torch.cuda.amp.autocast`
- [ ] Verify `contiguous()` on the embedding matrix
- [ ] Profile with Nsight before and after each optimization
