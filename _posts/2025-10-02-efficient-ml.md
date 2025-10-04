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

#### 4.2 Data Parallelism

**DataParallel**: Single-process, multi-threaded approach that works when the model fits on a single GPU. Each GPU keeps a copy of the model, processes different microbatches, then averages gradients across GPUs. The main bottleneck is that it relies on single-process, multi-threaded communication, leading to inefficient inter-GPU communication and potential slowdowns due to CPU overhead. DataParallel runs in a single process with multiple threads, so it suffers from GIL contention.

**Synchronization Approaches**:
At the end of each minibatch, workers need to synchronize gradients or weights to avoid staleness. There are two main synchronization approaches:

- **Bulk Synchronous Parallel (BSP)**: Workers synchronize at the end of every minibatch. It prevents model weights staleness and provides good learning efficiency, but each machine has to halt and wait for others to send gradients.

- **Asynchronous Parallel (ASP)**: Every GPU worker processes the data asynchronously with no waiting or stalling. However, it can easily lead to stale weights being used and thus lower the statistical learning efficiency. Even though it increases the computation time, it may not speed up training time to convergence.

**Distributed Data Parallel (DDP)**: Each GPU has its own process and can work on multiple nodes/machines. Uses Ring All-Reduce algorithm to avoid central bottlenecks. DDP has lower communication overhead compared to DataParallel.

**ZeRO (Zero Redundancy Optimizer)**: Not just the model parameters and gradients, but the optimizer state (including Adam momentum, variance) also takes a lot of memory. ZeRO-DP has three main optimization stages:

1. **Optimizer State Partitioning**: 4x memory reduction, same communication volume as DP. Gradient computation can be done independently for each GPU. When it is being done for parameters that are not on the current GPU, we incur a communication cost but this is the same as gradient averaging in DP. So always do this ZeRO stage-1.

2. **Gradient Partitioning**: 8x memory reduction, same communication volume as DP. This is similar to optimizer state partitioning in practice. Optimizer states are calculated per parameter anyway so this doesn't incur any extra cost.

3. **Parameter Partitioning**: Memory reduction is linear with DP degree. For example, splitting across 64 GPUs will yield a 64x memory reduction. There is a modest 50% increase in communication volume. This works because at any time doing forward or backward, only a subset of parameters (in a layer) are required for the operation. At best you're gonna need memory equivalent to a layer size. The model parameters can be sliced in any manner (vertically or horizontally). The way it's different from tensor parallelism or pipeline parallelism is that every computation still happens on each GPU using full tensors, just that the parameters are not all on single GPU.

**Key Implementations**:
- **FSDP (Fully Sharded Data Parallel)**: PyTorch's implementation of ZeRO
- **DeepSpeed**: Microsoft's open-source implementation of ZeRO

**Communication**: All-reduce = reduce-scatter + all-gather. Ring-reduce overhead: 2 × (N-1) × X/N bytes

**Resources**:
[1] [Scaling ML Models](https://www.youtube.com/watch?v=hc0u4avAkuM)
[2] [Training Optimization](https://www.youtube.com/watch?v=toUSzwR0EV8)
[3] [Communication Overhead](https://www.youtube.com/watch?v=UVX7SYGCKkA)
[4] [Communication overhead slides](https://docs.google.com/presentation/d/14SxjHdkvIw80FCAu5c1NGvFKDVF5DgvD2MJ1OwQ-5Gs/edit?slide=id.g24fe79ce068_0_154#slide=id.g24fe79ce068_0_154)



#### 4.3 Pipeline Parallelism

**Naive Model Parallel**: Partition the model by layers and put each partition on a separate GPU. The main deficiency and why this one is called "naive" MP, is that all but one GPU is idle at any given moment.

**GPipe**: Pipeline parallelism (PP) combines model parallelism with data parallelism to reduce inefficient time "bubbles". The main idea is to split one minibatch into multiple microbatches and enable each stage worker to process one microbatch simultaneously. Given m evenly split microbatches and d partitions, the bubble is (d-1)/(m+d-1).

**Activation Recomputation**: Only activations at partition boundaries are saved and communicated between workers. Intermediate activations at intra-partition layers are still needed for computing gradients so they are recomputed during backward passes. With activation recomputation, the memory cost for training M(l) is M(l) = O(l/d) + O(d) = O(√l).

**PipeDream**: It schedules each worker to alternatively process the forward and backward passes. PipeDream does not have an end-of-batch global gradient sync across all the workers. A naive implementation of 1F1B can easily lead to the forward and backward passes of one microbatch using different versions of model weights, thus lowering the learning efficiency. PipeDream proposed a few designs to tackle this issue:

- **Weight Stashing**: Each worker keeps track of several model versions and makes sure that the same version of weights are used in the forward and backward passes given one data batch.

- **Vertical Sync**: The version of model weights flows between stage workers together with activations and gradients. Then the computation adopts the corresponding stashed version propagated from the previous worker. This process keeps version consistency across workers.

- **PipeDream-flush**: PipeDream-flush adds a globally synchronized pipeline flush periodically, just like GPipe.

- **PipeDream-2BW**: PipeDream-2BW maintains only two versions of model weights, where "2BW" is short for "double-buffered weights". It generates a new model version every k microbatches and k should be larger than pipeline depth d. A newly updated model version cannot fully replace the old version immediately since some leftover backward passes still depend on the old version. In total only two versions need to be saved so the memory cost is much reduced.

**Advanced Pipeline Techniques**:

- **Breadth First Pipeline Parallelism**: Looped pipeline with the principle of GPipe constitutes breadth first search approach whereas Looped pipeline with principle of 1F1B constitutes a depth first search approach.

- **Zero Bubble Pipeline Parallelism**: Split Backward pass into two: Backward for Input and Backward for weights. Backward for input needs to be done first, backward for weights can be done later.
  - **ZB-H1**: Bubble reduction is because B is initiated earlier across all workers compared to 1F1B, and the tail-end bubbles are filled by the later-starting W passes.
  - **ZB-H2**: We introduce more F passes during the warm-up phase to fill the bubble preceding the initial B. We also reorder the W passes at the tail, which changes the layout from trapezoid into a parallelogram, eliminating all the bubbles in the pipeline.

- **Bypassing optimizer synchronization**: Use post-validation strategy to replace optimizer synchronization.

- **LLaMA-3**: Current implementations of pipeline parallelism have batch size constraint, memory imbalance due to embedding layer and warmup microbatches and computation imbalance due to output & loss calculation making the last stage execution latency bottleneck. They modify the pipeline schedule to run an arbitrary number of microbatches in each batch. To balance the pipeline, we reduce one Transformer layer each from the first and the last stages, respectively. This means that the first model chunk on the first stage has only the embedding, and the last model chunk on the last stage has only output projection and loss calculation.

- **DeepSeek-V3**: The key idea of DualPipe is to overlap the computation and communication within a pair of individual forward and backward chunks. It employs a bidirectional pipeline scheduling, which feeds micro-batches from both ends of the pipeline simultaneously and a significant portion of communications can be fully overlapped.

**Resources**:
[1] [GPipe Paper](https://arxiv.org/abs/1811.06965)
[2] [PipeDream Paper](https://arxiv.org/abs/1806.03377)
[3] [Zero Bubble Pipeline](https://arxiv.org/abs/2011.06448)

#### 4.4 Tensor Parallelism

**Column-wise Parallel**: X, A = [A1, A2] → O = [X@A1, X@A2]
**Row-wise Parallel**: X = [X1, X2], A = [A1|A2] → O = X1@A1 + X2@A2

**Key Characteristics**:
- Splits model vertically, partitioning computation and parameters across devices
- Requires significant communication between layers
- Works well within single nodes (high inter-GPU bandwidth)
- Efficiency degrades beyond single node due to communication overhead

**Implementation**: Megatron-LM provides open-source tensor parallelism implementation

**Resources**:
[1] [Megatron-LM Paper](https://arxiv.org/abs/1909.08053)
[2] [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM)

#### 4.5 Context Parallelism

Context Parallelism is about how to parallelize the sequence length into multiple GPUs. During forward propagation, each GPU handles a segment of the sequence, storing only the necessary Key and Value (KV) pairs. In the backward pass, these KV pairs are reassembled across GPUs using advanced communication schemes like all-gather and reduce-scatter transformed into point-to-point communications in a ring topology.

**Resources**:
[1] [Context Parallelism Paper](https://arxiv.org/abs/2105.03824)
[2] [Sequence Parallelism](https://arxiv.org/abs/2104.04473)

#### 4.6 Expert Parallelism (MoE)

Instead of every token being processed by the same dense network, we introduce a set of experts (sub-networks, usually feed-forward MLPs inside Transformer blocks). Each token is assigned (via a gating function) to one or a few experts. Experts are sharded across devices (GPUs/TPUs). Each device hosts a subset of experts, and tokens are routed to whichever device hosts their assigned expert.

**Mixture of Experts**: A router (usually a softmax gate) decides which expert(s) handle each token.

**Routing Strategies**:
- **Top-1 routing** (Switch Transformer): Pick the highest scoring expert
- **Top-k routing** (GShard, GLaM): Pick k experts and combine outputs

**Load Balancing Challenges**:
Some experts (on some GPUs) may get overloaded while others sit idle. This leads to device imbalance → bottlenecked training/inference.

- **Device Balance Loss**: Add a regularization term ("device balance loss") to the training objective that encourages an even distribution of tokens across devices.

- **Communication Balance Loss**: Additional loss term to balance communication patterns.

- **Auxiliary Free Load Balancing**: Instead of adding a penalty in loss, use architectural or algorithmic tricks to achieve balance automatically:
  - Randomized routing with constraints
  - Capacity-based routing (set a hard cap on tokens per expert)
  - Priority dropping (drop excess tokens if an expert is full)

**Benefits**: Scale model size without proportional compute increase

**Resources**:
[1] [Switch Transformer Paper](https://arxiv.org/abs/2101.03961)
[2] [GLaM Paper](https://arxiv.org/abs/2112.06905)
[3] [GShard Paper](https://arxiv.org/abs/2006.16668)
[4] [Mixture of Experts Survey](https://arxiv.org/abs/2202.08906)

---

### 5. Other Key Resources
- [1] [Stanford CS229s](https://cs229s.stanford.edu/fall2023/calendar/)
- [2] [Stanford CS224n](https://web.stanford.edu/class/cs224n/)
- [3] [NVIDIA NeMo Framework](https://docs.nvidia.com/nemo-framework/user-guide/24.07/nemotoolkit/index.html)
- [4] [Lilian Weng's Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)
- [5] [Scaling ML Models](https://www.youtube.com/watch?v=hc0u4avAkuM)
- [6] [Training Optimization](https://www.youtube.com/watch?v=toUSzwR0EV8)
- [7] [Communication Overhead](https://www.youtube.com/watch?v=UVX7SYGCKkA)

---

### Conclusion

Optimizing large language models requires careful consideration across multiple dimensions. The techniques discussed here represent the current state-of-the-art in LLM optimization, from memory-efficient attention mechanisms to advanced parallelism strategies. As models continue to grow, these optimization techniques become increasingly critical for practical deployment.

*This post covers the essential optimization techniques for large language models. Feel free to reach out if you'd like to discuss any of these topics in more detail!*
