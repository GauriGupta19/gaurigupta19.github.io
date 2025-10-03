---
layout: post
title: "Large Language Model Optimization: Memory, Compute, and Inference Techniques"
date: 2024-01-15
author: "Gauri Gupta"
categories: ["Research", "Machine Learning", "Optimization"]
excerpt: "A comprehensive guide to optimizing large language models across memory, compute, and inference dimensions. Covering techniques from Flash Attention to advanced parallelism strategies."
---

# Large Language Model Optimization: Memory, Compute, and Inference Techniques

Training and deploying large language models efficiently is one of the most critical challenges in modern AI. As models grow to billions of parameters, traditional approaches quickly become infeasible. In this post, I'll share a comprehensive overview of optimization techniques across memory, compute, and inference dimensions.

---

## 1. Memory Optimization Techniques

### 1.1 Flash Attention

The attention mechanism has quadratic time and memory complexity in sequence length, presenting significant runtime and memory challenges for longer sequences.

**Tiling Technique**: Decomposes inputs based on shared memory size and calculates softmax one tile at a time. Instead of working on entire query, key, value tensors at once, it makes several passes and combines results in subsequent steps.

**Recomputation Technique**: Stores softmax normalization factors (linear to sequence length) instead of softmax results (quadratic to sequence length), using these factors to recompute attention scores. This reduces memory requirements and I/O traffic between global and shared memory.

**Key Resources**:
- [Online softmax and tiling](https://www.youtube.com/watch?v=LKwyHWYEIMQ&t=14s)
- [Matrix multiplication tiling](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)

### 1.2 Multi-Query and Grouped Query Attention

- **MQA (Multi-Query Attention)**: Reduces memory by sharing keys and values across attention heads
- **GQA (Grouped Query Attention)**: Balances efficiency and quality by grouping queries

### 1.3 Activation Recomputation

Input activations easily saturate device memory when training LLMs with large sequence lengths or micro-batch sizes. Checkpointing a few activations and recomputing the rest reduces device memory requirements.

---

## 2. Compute Optimization Techniques

### 2.1 Sequence Packing

A training technique where multiple training sequences are concatenated into one long sequence. This eliminates padding and allows more tokens to be processed per micro-batch, maximizing both GPU compute and memory utilization.

### 2.2 Efficient Transformers

**BigBird**: Uses a combination of local, random, and global attention patterns to reduce complexity to O(n).

**Longformer**: Utilizes sliding window (local) attention combined with global attention for improved efficiency.

**Low-Rank Approximations**: Projects key and value matrices into lower-dimensional spaces.

**LongNet**: At lower layers, tokens attend to nearby tokens (small dilation). At higher layers, dilation factor grows, allowing tokens to reach further. Scales linearly with sequence length O(Nd).

**Resources**:
- [LongNet Video](https://www.youtube.com/watch?v=nC2nU9j9DVQ)
- [LongNet Paper](https://arxiv.org/pdf/2307.02486)

---

## 3. Inference Optimization Techniques

### 3.1 KV Caching

KV caching stores computed key-value pairs to avoid recomputation during generation. This is essential for efficient autoregressive generation.

**Advanced KV Cache Optimizations**:
- **Grouped Multi Query Attention**: Reduces KV cache memory by grouping multiple queries with same keys and values
- **Multi-head Latent Attention**: Projects K, V, Q into lower-dimensional latent space, computing attention in latent space then projecting back
- **Cross Layer KV-sharing**: Ties KV cache across neighboring attention layers
- **Interleaving Local and Global Attention**: Uses global attention in every 4-6 layers

**Resources**:
- [KV Caching Video](https://www.youtube.com/watch?v=UiX8K-xBUpE&t=4822s)
- [FLOPS computation with KV cache](https://docs.google.com/presentation/d/14hK7SmkUNfSEIRGyptFD2bGO7K9sJOTnwjAVg3vgg6g/edit?slide=id.g286de50af37_0_933#slide=id.g286de50af37_0_933)

### 3.2 Stateful Caching

In chat settings, each user query is appended to long dialogue history. When the model runs on input [prefix + user_message], it computes attention KV for that sequence and stores it in cache, keyed by rolling hash of prefix tokens.

**Example**: If you cached "Hello, how are you?" and the new prefix is "Hello, how are you doing today?", you can reuse the overlapping part. For new queries, compute rolling hashes for all prefixes and find the longest cached match.

KV cache is organized in a tree structure with LRU (least recently used) eviction, so you can drop old contexts if memory is full.

### 3.3 Speculative Decoding

Uses a smaller draft LLM to generate responses, then uses the target LLM to verify the response, significantly speeding up inference.

### 3.4 Model Compression Techniques

**Distillation**: Builds smaller, cheaper "student models" by transferring skills from pre-trained "teacher models".

**Quantization**: Compresses models by representing weights/activations with fewer bits instead of standard fp32.

**Quantization Types**:
- **min/max**: Simple but susceptible to outliers
- **MSE**: Minimizes MSE between original and quantized values
- **Cross-entropy**: Preserves order of largest values after quantization for softmax

**Post-Training Quantization (PTQ)**: Converts weights to lower precision after training convergence. Cheap to implement but can fail with outlier features in large models.

**Quantization-Aware Training (QAT)**: Applies quantization during pre-training or fine-tuning, simulating quantization error as a regularizer.

**Resources**:
- [Quantization Video](https://www.youtube.com/watch?v=0VdNflU08yA)
- [Character.ai Optimization Guide](https://research.character.ai/optimizing-inference/)
- [Lilian Weng's Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)

---

## 4. Training Optimization

### 4.1 Mixed Precision Training

Uses bfloat16 and other reduced precision formats to reduce memory usage while maintaining training stability.

### 4.2 Parallelism Approaches

#### 4.2.1 Data Parallelism

- **DataParallel**: Single-process, multi-threaded approach for single-GPU models
- **Distributed Data Parallel (DDP)**: Each GPU has its own process, works across multiple nodes
- **ZeRO**: Optimizes not just parameters and gradients but also optimizer states

**ZeRO Stages**:
1. **Optimizer State Partitioning**: 4x memory reduction
2. **Gradient Partitioning**: 8x memory reduction  
3. **Parameter Partitioning**: Linear memory reduction with DP degree

**Resources**:
- [Scaling ML Models](https://www.youtube.com/watch?v=hc0u4avAkuM)
- [Training Optimization](https://www.youtube.com/watch?v=toUSzwR0EV8)

#### 4.2.2 Pipeline Parallelism

- **GPipe**: Splits minibatches into microbatches, enabling simultaneous processing
- **PipeDream**: Alternates forward and backward passes across workers
- **Zero Bubble Pipeline**: Eliminates pipeline bubbles through advanced scheduling

#### 4.2.3 Tensor Parallelism

- **Column-wise Parallel**: Splits matrices by columns
- **Row-wise Parallel**: Splits matrices by rows
- **Megatron-LM**: Open source implementation of tensor parallelism

#### 4.2.4 Context Parallelism

Parallelizes sequence length across multiple GPUs. Each GPU handles a segment of the sequence, storing necessary KV pairs, then reassembles them during backward pass.

#### 4.2.5 Expert Parallelism

Instead of processing every token with the same dense network, introduces expert sub-networks. Tokens are routed to specific experts sharded across devices.

**Mixture of Experts (MoE)**:
- **Top-1 routing**: Picks highest scoring expert
- **Top-k routing**: Picks k experts and combines outputs
- **Load Balancing**: Ensures even distribution of tokens across experts

---

## 5. Key Resources

### 5.1 Academic Courses
- [Stanford CS229s](https://cs229s.stanford.edu/fall2023/calendar/)
- [Stanford CS224n](https://web.stanford.edu/class/cs224n/)

### 5.2 Technical Resources
- [NVIDIA NeMo Framework](https://docs.nvidia.com/nemo-framework/user-guide/24.07/nemotoolkit/index.html)
- [Character.ai Optimization Guide](https://research.character.ai/optimizing-inference/)
- [Lilian Weng's Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)

### 5.3 Video Lectures
- [Scaling ML Models](https://www.youtube.com/watch?v=hc0u4avAkuM)
- [Training Optimization](https://www.youtube.com/watch?v=toUSzwR0EV8)
- [Communication Overhead](https://www.youtube.com/watch?v=UVX7SYGCKkA)

---

## Conclusion

Optimizing large language models requires careful consideration across multiple dimensions. The techniques discussed here represent the current state-of-the-art in LLM optimization, from memory-efficient attention mechanisms to advanced parallelism strategies. As models continue to grow, these optimization techniques become increasingly critical for practical deployment.

*This post covers the essential optimization techniques for large language models. Feel free to reach out if you'd like to discuss any of these topics in more detail!*
