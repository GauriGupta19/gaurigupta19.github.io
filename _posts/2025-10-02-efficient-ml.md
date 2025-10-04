---
layout: post
title: "LLM Optimization Interview Notes: Memory, Compute & Inference Techniques"
date: 2024-01-15
author: "Gauri Gupta"
categories: ["Interview Prep", "Machine Learning", "Optimization"]
excerpt: "Job preparation notes covering essential LLM optimization techniques for AI lab interviews. Quick reference for memory, compute, and inference optimization strategies."
---

## Large Language Model Optimization: Memory, Compute, and Inference Techniques

When I was preparing for interviews at big AI labs, I found myself constantly reviewing the same core optimization concepts that kept coming up in technical discussions. These aren't comprehensive explanations, but rather the essential techniques I brushed up on and referenced during interviews at companies like Google, Meta, Anthropic, and other leading AI research labs.

Training and deploying large language models efficiently is one of the most critical challenges in modern AI. As models grow to billions of parameters, traditional approaches quickly become infeasible. In this post, I'll share the optimization techniques that proved most valuable during my interview preparation and actual technical discussions.

---

### 1. Memory Optimization Techniques

Memory is the biggest bottleneck in LLM training/inference. These techniques reduce memory footprint while maintaining model quality.

#### 1.1 Flash Attention

The attention mechanism has quadratic time and memory complexity in sequence length, presenting significant runtime and memory challenges for longer sequences.
Flash Attention reduces attention memory complexity from O(N²) to O(N) through tiling and recomputation techniques. Instead of processing entire attention matrices at once, it processes attention in blocks and stores normalization factors instead of full attention matrices. The tiling technique decomposes inputs based on shared memory size, while recomputation stores softmax normalization factors (linear to sequence length) instead of softmax results (quadratic to sequence length).

**Tiling Technique**: Decomposes inputs based on shared memory size and calculates softmax one tile at a time. Instead of working on entire query, key, value tensors at once, it makes several passes and combines results in subsequent steps.

**Recomputation Technique**: Stores softmax normalization factors (linear to sequence length) instead of softmax results (quadratic to sequence length), using these factors to recompute attention scores. This reduces memory requirements and I/O traffic between global and shared memory.

**Key Resources**:
[1] [Matrix multiplication tiling](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)
[2] [Online softmax and tiling](https://www.youtube.com/watch?v=LKwyHWYEIMQ&t=14s)

#### 1.2 Multi-Query and Grouped Query Attention

- **MQA (Multi-Query Attention)**: Reduces memory by sharing keys and values across attention heads
- **GQA (Grouped Query Attention)**: Balances efficiency and quality by grouping queries

#### 1.3 Activation Recomputation

Input activations easily saturate device memory when training LLMs with large sequence lengths or micro-batch sizes. Checkpointing a few activations and recomputing the rest reduces device memory requirements.

---

### 2. Compute Optimization Techniques

Maximize GPU utilization and reduce computational overhead through smarter data handling and model architectures.

#### 2.1 Sequence Packing

A training technique where multiple training sequences are concatenated into one long sequence. This eliminates padding and allows more tokens to be processed per micro-batch, maximizing both GPU compute and memory utilization.

#### 2.2 Efficient Transformers

**BigBird**: Uses a combination of local, random, and global attention patterns to reduce complexity to O(n).

**Longformer**: Utilizes sliding window (local) attention combined with global attention for improved efficiency.

**Low-Rank Approximations**: Projects key and value matrices into lower-dimensional spaces.

**LongNet**: At lower layers, tokens attend to nearby tokens (small dilation). At higher layers, dilation factor grows, allowing tokens to reach further. Scales linearly with sequence length O(Nd).

**Resources**: [1][Scaling Transformers with LongNet](https://www.youtube.com/watch?v=nC2nU9j9DVQ)

---

### 3. Inference Optimization Techniques

Inference is where most production costs occur. These techniques dramatically speed up generation while maintaining quality.

#### 3.1 KV Caching

KV caching stores computed key-value pairs to avoid recomputation during generation. This is essential for efficient autoregressive generation.

**Advanced KV Cache Optimizations**:
- **Grouped Multi Query Attention**: Reduces KV cache memory by grouping multiple queries with same keys and values
- **Multi-head Latent Attention**: Projects K, V, Q into lower-dimensional latent space, computing attention in latent space then projecting back
- **Cross Layer KV-sharing**: Ties KV cache across neighboring attention layers
- **Interleaving Local and Global Attention**: Uses global attention in every 4-6 layers

**Resources**: 
[1][KV Caching Video](https://www.youtube.com/watch?v=Mn_9W1nCFLo&t=3869s)
[2][FLOPS computation efficiency with KV cache](https://docs.google.com/presentation/d/14hK7SmkUNfSEIRGyptFD2bGO7K9sJOTnwjAVg3vgg6g/edit?slide=id.g286de50af37_0_933#slide=id.g286de50af37_0_933)

#### 3.2 Stateful Caching

Stateful caching stores conversation history using rolling hashes, allowing reuse of overlapping prefixes. For example, if "Hello, how are you?" is cached, it can be reused when the new prefix is "Hello, how are you doing today?" The cache is organized in a tree structure with LRU eviction to manage memory efficiently. 
For a new query, compute rolling hashes for all its prefixes and find the longest cached match.Load the KV tensors from cache and continue computation only from the new tokens.
KV cache organized in a tree structure with LRU (least recently used) eviction, so you can drop old contexts if memory is full.


#### 3.3 Speculative Decoding

Speculative decoding uses a smaller draft model to generate responses, then uses the target model to verify them, achieving 2-3x speedup in inference. The draft model must be fast and well-aligned with the target model for this technique to be effective.

#### 3.4 Model Compression
**Distillation** trains smaller "student" models by transferring knowledge from larger "teacher" models, trading model quality for size and speed improvements.

**Quantization** compresses models by representing weights and activations with fewer bits. Post-training quantization (PTQ) is cheap to implement but can fail with outlier features in large models, while quantization-aware training (QAT) applies quantization during training for better quality but higher cost. Different quantization methods include min/max (simple but susceptible to outliers), MSE (minimizes mean squared error), and cross-entropy (preserves order of largest values for softmax).

#### 3.4 Quantization Techniques

Compressing a model by representing weights/activations with fewer bits instead of standard fp32 (32-bit float).

**Quantization Types**:
- **min/max**: Simple but susceptible to outliers
- **MSE**: Minimizes MSE between original and quantized values
- **Cross-entropy**: Preserves order of largest values after quantization for softmax; argmin(softmax(v), softmax(v’))


**Post-Training Quantization (PTQ)**: A model is first trained to convergence and then we convert its weights to lower precision without more training. It is usually quite cheap to implement, in comparison to training. As the model size continues to grow to billions of parameters, outlier features of high magnitude start to emerge in all transformer layers, causing failure of simple low-bit quantization. To quantize the input x, attach observers that collect statistical data like mean and std and use it to quantize.
■ Mixed-precision quantization: Don’t quantize everything to the same bit width. Implement quantization at different precision for weights vs activation. 


**Quantization-Aware Training (QAT)**: Quantization is applied during pre-training or further fine-tuning. Siimulate quantization and dequantization in the forward pass. This simulaes the quantization error and acts as a regualizer to make the model robust to it. 
Backprop: quantization is not differentiable. Approx gradient with STE(straight-through approximator) -> 1 in range(alpha, beta) and 0 outside


**Resources**:
[1] [Quantization Video](https://www.youtube.com/watch?v=0VdNflU08yA)
[2] [Lilian Weng's Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)

---
### 4. Training Optimization

Training large models requires sophisticated parallelism strategies. Know the different approaches and their trade-offs.

#### 4.1 Mixed Precision Training

Mixed precision training uses bfloat16 and fp16 formats with loss scaling to reduce memory usage while maintaining training stability. This provides 2x memory reduction and faster training, but requires careful handling of numerical stability issues.

#### 4.2 Parallelism Approaches

**Data Parallelism** 

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

##### 4.2.2 Pipeline Parallelism

- **GPipe**: Splits minibatches into microbatches, enabling simultaneous processing
- **PipeDream**: Alternates forward and backward passes across workers
- **Zero Bubble Pipeline**: Eliminates pipeline bubbles through advanced scheduling

##### 4.2.3  Tensor Parallelism
splits matrix operations across GPUs, either column-wise or row-wise. Megatron-LM provides an open-source implementation of tensor parallelism for large language models.
- **Column-wise Parallel**: Splits matrices by columns
- **Row-wise Parallel**: Splits matrices by rows
- **Megatron-LM**: Open source implementation of tensor parallelism

##### 4.2.4 Context Parallelism
 splits sequence length across multiple GPUs, with each GPU handling a segment of the sequence. This is useful for very long sequences that don't fit on a single GPU.

##### 4.2.5 Expert Parallelism (MoE)
routes tokens to specialized expert networks instead of processing every token with the same dense network. Routing can be Top-1 (single expert) or Top-k (multiple experts), with the main challenge being load balancing across experts. The benefit is scaling model size without proportional compute increase.

---

### 5. Other Key Resources
[1] [Stanford CS229s](https://cs229s.stanford.edu/fall2023/calendar/)
[2] [Stanford CS224n](https://web.stanford.edu/class/cs224n/)
[3] [NVIDIA NeMo Framework](https://docs.nvidia.com/nemo-framework/user-guide/24.07/nemotoolkit/index.html)
[4] [Lilian Weng's Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)
[5] [Scaling ML Models](https://www.youtube.com/watch?v=hc0u4avAkuM)
[6] [Training Optimization](https://www.youtube.com/watch?v=toUSzwR0EV8)
[7] [Communication Overhead](https://www.youtube.com/watch?v=UVX7SYGCKkA)

---

### Conclusion

Optimizing large language models requires careful consideration across multiple dimensions. The techniques discussed here represent the current state-of-the-art in LLM optimization, from memory-efficient attention mechanisms to advanced parallelism strategies. As models continue to grow, these optimization techniques become increasingly critical for practical deployment.

*This post covers the essential optimization techniques for large language models. Feel free to reach out if you'd like to discuss any of these topics in more detail!*
