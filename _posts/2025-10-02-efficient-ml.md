---
layout: post
title: "LLM Optimization Interview Notes: Memory, Compute & Inference Techniques"
date: 2024-01-15
author: "Gauri Gupta"
categories: ["Interview Prep", "Machine Learning", "Optimization"]
excerpt: "Job preparation notes covering essential LLM optimization techniques for AI lab interviews. Quick reference for memory, compute, and inference optimization strategies."
---

# LLM Optimization Interview Notes
*Essential techniques for LLM large scale optimizations - not comprehensive explanations but key concepts to know*

Training and deploying large language models efficiently is one of the most critical challenges in modern AI. As models grow to billions of parameters, traditional approaches quickly become infeasible. In this post, I'll share a comprehensive overview of optimization techniques across memory, compute, and inference dimensions.

---

## **1. MEMORY OPTIMIZATION**
*Memory is the biggest bottleneck in LLM training/inference. These techniques reduce memory footprint while maintaining model quality.*

### **Flash Attention**

The attention mechanism has quadratic time and memory complexity in sequence length, presenting significant runtime and memory challenges for longer sequences.
Flash Attention reduces attention memory complexity from O(N²) to O(N) through tiling and recomputation techniques. Instead of processing entire attention matrices at once, it processes attention in blocks and stores normalization factors instead of full attention matrices. The tiling technique decomposes inputs based on shared memory size, while recomputation stores softmax normalization factors (linear to sequence length) instead of softmax results (quadratic to sequence length).

Decomposes inputs based on shared memory size and calculates softmax one tile at a time. Instead of working on entire query, key, value tensors at once, it makes several passes and combines results in subsequent steps.

Stores softmax normalization factors (linear to sequence length) instead of softmax results (quadratic to sequence length), using these factors to recompute attention scores. This reduces memory requirements and I/O traffic between global and shared memory.

**Key Resources**:
- [Matrix multiplication tiling](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)
- [Online softmax and tiling](https://www.youtube.com/watch?v=LKwyHWYEIMQ&t=14s)

### **Multi-Query & Grouped Query Attention**

- **MQA (Multi-Query Attention)**: Reduces memory by sharing keys and values across attention heads
- **GQA (Grouped Query Attention)**: Balances efficiency and quality by grouping queries

### 1.3 Activation Recomputation

Input activations easily saturate device memory when training LLMs with large sequence lengths or micro-batch sizes. Checkpointing a few activations and recomputing the rest reduces device memory requirements.

---

## **2. COMPUTE OPTIMIZATION**
*Maximize GPU utilization and reduce computational overhead through smarter data handling and model architectures.*

### **Sequence Packing**

Sequence packing concatenates multiple training sequences into one long sequence, eliminating padding and allowing more tokens to be processed per micro-batch. This maximizes both GPU compute and memory utilization, but requires careful attention masking to prevent tokens from different sequences from attending to each other.

### 2.2 Efficient Transformers
Several transformer variants reduce computational complexity
**BigBird**: Uses a combination of local, random, and global attention patterns to reduce complexity to O(n).

**Longformer**: Utilizes sliding window (local) attention combined with global attention for improved efficiency.

**Low-Rank Approximations**: Projects key and value matrices into lower-dimensional spaces.

**LongNet**: At lower layers, tokens attend to nearby tokens (small dilation). At higher layers, dilation factor grows, allowing tokens to reach further. Scales linearly with sequence length O(Nd).

**Resources**:
- [LongNet Video](https://www.youtube.com/watch?v=nC2nU9j9DVQ)
- [LongNet Paper](https://arxiv.org/pdf/2307.02486)

---

## **3. INFERENCE OPTIMIZATION**
*Primer: Inference is where most production costs occur. These techniques dramatically speed up generation while maintaining quality.*

### **KV Caching**

KV caching stores computed key-value pairs to avoid recomputation during generation. This is essential for efficient autoregressive generation.

**Advanced KV Cache Optimizations**:
- **Grouped Multi Query Attention**: Reduces KV cache memory by grouping multiple queries with same keys and values
- **Multi-head Latent Attention**: Projects K, V, Q into lower-dimensional latent space, computing attention in latent space then projecting back
- **Cross Layer KV-sharing**: Ties KV cache across neighboring attention layers
- **Interleaving Local and Global Attention**: Uses global attention in every 4-6 layers

**Resources**:
- [KV Caching Video](https://www.youtube.com/watch?v=UiX8K-xBUpE&t=4822s)
- [FLOPS computation efficiency with KV cache](https://docs.google.com/presentation/d/14hK7SmkUNfSEIRGyptFD2bGO7K9sJOTnwjAVg3vgg6g/edit?slide=id.g286de50af37_0_933#slide=id.g286de50af37_0_933)

### **Stateful Caching**

Stateful caching stores conversation history using rolling hashes, allowing reuse of overlapping prefixes. For example, if "Hello, how are you?" is cached, it can be reused when the new prefix is "Hello, how are you doing today?" The cache is organized in a tree structure with LRU eviction to manage memory efficiently.

### **Speculative Decoding**

Speculative decoding uses a smaller draft model to generate responses, then uses the target model to verify them, achieving 2-3x speedup in inference. The draft model must be fast and well-aligned with the target model for this technique to be effective.

### **Model Compression**
**Distillation** trains smaller "student" models by transferring knowledge from larger "teacher" models, trading model quality for size and speed improvements.

**Quantization** compresses models by representing weights and activations with fewer bits. Post-training quantization (PTQ) is cheap to implement but can fail with outlier features in large models, while quantization-aware training (QAT) applies quantization during training for better quality but higher cost. Different quantization methods include min/max (simple but susceptible to outliers), MSE (minimizes mean squared error), and cross-entropy (preserves order of largest values for softmax).

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

## **4. TRAINING OPTIMIZATION**
*Primer: Training large models requires sophisticated parallelism strategies. Know the different approaches and their trade-offs.*

### **Mixed Precision Training**

Mixed precision training uses bfloat16 and fp16 formats with loss scaling to reduce memory usage while maintaining training stability. This provides 2x memory reduction and faster training, but requires careful handling of numerical stability issues.

### **Parallelism Strategies**

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

**Pipeline Parallelism** splits the model across multiple GPUs.

- **GPipe**: Splits minibatches into microbatches, enabling simultaneous processing
- **PipeDream**: Alternates forward and backward passes across workers
- **Zero Bubble Pipeline**: Eliminates pipeline bubbles through advanced scheduling

**Tensor Parallelism** splits matrix operations across GPUs, either column-wise or row-wise. Megatron-LM provides an open-source implementation of tensor parallelism for large language models.
- **Column-wise Parallel**: Splits matrices by columns
- **Row-wise Parallel**: Splits matrices by rows
- **Megatron-LM**: Open source implementation of tensor parallelism

**Context Parallelism** splits sequence length across multiple GPUs, with each GPU handling a segment of the sequence. This is useful for very long sequences that don't fit on a single GPU.

**Expert Parallelism (MoE)** routes tokens to specialized expert networks instead of processing every token with the same dense network. Routing can be Top-1 (single expert) or Top-k (multiple experts), with the main challenge being load balancing across experts. The benefit is scaling model size without proportional compute increase.

---

## **5. KEY OHER RESOURCES**

### **Academic Courses**
- [Stanford CS229s](https://cs229s.stanford.edu/fall2023/calendar/) - Scaling ML
- [Stanford CS224n](https://web.stanford.edu/class/cs224n/) - NLP

### **Technical References**
- [NVIDIA NeMo Framework](https://docs.nvidia.com/nemo-framework/user-guide/24.07/nemotoolkit/index.html)
- [Character.ai Optimization Guide](https://research.character.ai/optimizing-inference/)
- [Lilian Weng's Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)

### **Video Lectures**
- [Scaling ML Models](https://www.youtube.com/watch?v=hc0u4avAkuM)
- [Training Optimization](https://www.youtube.com/watch?v=toUSzwR0EV8)
- [Communication Overhead](https://www.youtube.com/watch?v=UVX7SYGCKkA)

---

## **INTERVIEW TALKING POINTS**

### **Memory Questions**
- "How would you reduce memory for a 70B parameter model?"
- "What's the trade-off between activation recomputation and compute time?"
- "When would you use Flash Attention vs other attention optimizations?"

### **Inference Questions**  
- "How does KV caching work and what are its limitations?"
- "Explain speculative decoding and when it's most effective"
- "What's the difference between PTQ and QAT quantization?"

### **Training Questions**
- "Compare data parallelism vs pipeline parallelism"
- "When would you use ZeRO Stage 3 vs other approaches?"
- "How does MoE routing work and what are the challenges?"

---

*These notes cover the essential optimization techniques for LLM systems. Focus on understanding the trade-offs and when each technique is most appropriate. Good luck with your interviews!*
