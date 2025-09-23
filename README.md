# LoSoU: An Experimental Recursive Language Model Block

## Overview

**LoSoU (Long-Short Update)** is an experimental building block for language modeling that replaces attention with a simple **recursive update mechanism**.
Instead of computing full pairwise interactions, LoSoU maintains an **exponential moving average (EMA)** across tokens, combined with lightweight gating and normalization.

This design is inspired by recurrent-style formulations (e.g., RWKV) but is implemented with modern stabilization techniques for TPU/GPU training.

## Key Ideas

* **Recursive update**: processes tokens step-by-step with EMA smoothing.
* **Integrated gating**: similar to GLU/SwiGLU, fused directly into the block.
* **Normalization & residuals**: improve stability and reduce NaN issues.
* **Efficiency**: linear complexity (*O(N·d)*) and faster than Transformers on moderate sequence lengths.

## Status

* 🚧 **Experimental** — not competitive with state-of-the-art Transformers yet.
* ✅ Trains stably on TPUs with mixed precision.
* 📊 Shows lower training loss and faster throughput than a baseline Transformer of similar size in early tests.

## Intended Use

LoSoU is released mainly for:

* Researchers interested in **attention-free alternatives**.
* Experimentation with **recursive and convolutional hybrids**.
* Benchmarking efficiency vs. attention-based architectures.
