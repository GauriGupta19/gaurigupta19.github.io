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

#### 4.2. Data Parallelism
DataParallel: Single-process, multi-threaded. Works when the model fits on a single GPU. Keep a copy of the model on each GPU. Do forward and backward pass for a microbatch on each GPU separately. Average the gradients, perform weight update and send the updated copy of the weights to all GPUs. The main bottleneck is that it relies on single-process, multi-threaded communication, leading to inefficient inter-GPU communication and potential slowdowns due to CPU overhead. DataParallel runs in a single process with multiple threads, so it suffers from GIL contention. At the end of each minibatch, workers need to synchronize gradients or weights to avoid staleness. There are two main synchronization approaches and both have clear pros & cons. 
○ Bulk synchronous parallels (BSP): Workers sync data at the end of every minibatch. It prevents model weights staleness and good learning efficiency but each machine has to halt and wait for others to send gradients. 
○ Asynchronous parallel (ASP): Every GPU worker processes the data asynchronously, no waiting or stalling. However, it can easily lead to stale weights being used and thus lower the statistical learning efficiency. Even though it increases the computation time, it may not speed up training time to convergence. 
● Distributed Data Parallel: Each GPU has its own process. Can work on multiple nodes/machines. Uses Ring all reduce algorithm avoiding a central bottleneck. DDP has lower communication overhead. 
● ZeRO: Not just the model parameters and graidents but the optimizer state (including avg, momentum from Adam) also take a lot of memory
ZeRO-DP has three main optimization stages: 
○ Optimizer State Partitioning: 4x memory reduction, same communication volume as DP. Gradient computation can be done independently for each GPU and when it is being done for parameters that are not on the current GPU, we incur a communication cost but this is the same as gradient averaging in DP. So always do this ZeRO stage-1.
○ Add Gradient Partitioning: 8x memory reduction, same communication volume as DP. This is similar to optimizer state partitioning in practice. Optimizer states are calculated per parameter anyway so this doesn’t incur any extra cost. 
○ Add Parameter Partitioning: Memory reduction is linear with DP degree Nd. For example, splitting across 64 GPUs (Nd = 64) will yield a 64x memory reduction. There is a modest 50% increase in communication volume. This works because at any time doing forward or backward, only a subset of parameters (in a layer) for example are required for the operation. At best you’re gonna need memory equivalent to a layer size. The model parameters can be sliced in any manner (vertically or horizontally). The way its different from tensor parallelism or pipeline parallelism is that every computation still happens on each GPU using full tensors, just that the parameters are not all on single GPU. 
● FSDP (Fully Sharded Data Parallel): It is same as ZeRO. Just a different name. 
● DeepSpeed: Deepspeed is an open source implementation of Zero-DP.

Communication: all reduce = reduce scatter + all gather, 
Ring-reduce -> overhead 2 * (N-1)  * X/N bytes
Video: https://www.youtube.com/watch?v=UVX7SYGCKkA, https://www.youtube.com/watch?v=toUSzwR0EV8
Communication overhead slides: https://docs.google.com/presentation/d/14SxjHdkvIw80FCAu5c1NGvFKDVF5DgvD2MJ1OwQ-5Gs/edit?slide=id.g24fe79ce068_0_154#slide=id.g24fe79ce068_0_154


- **DataParallel**: Single-process, multi-threaded approach for single-GPU models
- **Distributed Data Parallel (DDP)**: Each GPU has its own process, works across multiple nodes
- **ZeRO**: Optimizes not just parameters and gradients but also optimizer states

**ZeRO Stages**:
1. **Optimizer State Partitioning**: 4x memory reduction
2. **Gradient Partitioning**: 8x memory reduction  
3. **Parameter Partitioning**: Linear memory reduction with DP degree

**Resources**:
[1] [Scaling ML Models](https://www.youtube.com/watch?v=hc0u4avAkuM)
[2] [Training Optimization](https://www.youtube.com/watch?v=toUSzwR0EV8)

#### 4.3 Pipeline Parallelism
 Naive Model Parallel: Partition the model by layers and put each partition on a separate GPU. The main deficiency and why this one is called “naive” MP, is that all but one GPU is idle at any given moment.
● Gpipe: Pipeline parallelism (PP) combines model parallelism with data parallelism to reduce inefficient time “bubbles’’. The main idea is to split one minibatch into multiple microbatches and enable each stage worker to process one microbatch simultaneously. Given m evenly split microbatches and d partitions, the bubble is 
(d-1)/(m+d-1) 
○ Activation Recomputation: Only activations at partition boundaries are saved and communicated between workers. Intermediate activations at intra-partition layers are still needed for computing gradients so they are recomputed during backward passes. With activation recomputation, the memory cost for training M(l) is 
M(l) = O(l/d) + O(d) = O(sqrt(l)) 
● Pipedream: It schedules each worker to alternatively process the forward and backward passes. PipeDream does not have an end-of-batch global gradient sync across all the workers, an naive implementation of 1F1B can easily lead to the forward and backward passes of one microbatch using different versions of model weights, thus lowering the learning efficiency. PipeDream proposed a few designs to tackle this issue: Weight stashing: Each worker keeps track of several model versions and makes sure that the same version of weights are used in the forward and backward passes given one data batch. Vertical Sync: The version of model weights flows between stage workers together with activations and gradients. Then the computation adopts the corresponding stashed version propagated from the previous worker. This process keeps version consistency across workers. 
○ Pipedream-flush: PipeDream-flush adds a globally synchronized pipeline flush periodically, just like GPipe. 
○ Pipedream-2BW: PipeDream-2BW maintains only two versions of model weights, where “2BW” is short for “double-buffered weights”. It generates a new model version every k microbatches and k should be larger than pipeline depth d. A newly updated model version cannot fully replace the old version immediately since some leftover backward passes still depend on the old version. In total only two versions need to be saved so the memory cost is much reduced. 
● Breadth First Pipeline Parallelism: Looped pipeline with the principe of Gpipe constitutes breadth first search approach where as Looped pipeline with principe of 1F1B constitutes a depth first search approach. 
● Zero Bubble Pipeline Parallelism: Split Backward pass into two: Backward for Input and Backward for weights. Backward for input needs to be done first, backward for weights can be done later. ○ ZB-H1: Bubble reduction is because B is initiated earlier across all workers compared to 1F1B, and the tail-end bubbles are filled by the later-starting W passes. 
○ ZB-H2: We introduce more F passes during the warm-up phase to fill the bubble preceding the initial B. We also reorder the W passes at the tail, which changes the layout from trapezoid into a parallelogram, eliminating all the bubbles in the pipeline. 
■ Bypassing optimizer synchronization: Use post-validation strategy to replace optimizer synchronization. 
● LLAMA3: Current implementations of pipeline parallelism have batch size constraint, memory imbalance due to embedding layer and warmup microbatches and computation imbalance due to output & loss calculation making the last stage execution latency bottleneck. They modify the pipeline schedule to run an arbitrary number of microbatches in each batch. To balance the pipeline, we reduce one Transformer layer each from the first and the last stages, respectively. This means that the
first model chunk on the first stage has only the embedding, and the last model chunk on the last stage has only output projection and loss calculation. 
● DeepSeek-V3: The key idea of DualPipe is to overlap the computation and communication within a pair of individual forward and backward chunks. It employs a bidirectional pipeline scheduling, which feeds micro-batches from both ends of the pipeline simultaneously and a significant portion of communications can be fully overlapped. 

- **GPipe**: Splits minibatches into microbatches, enabling simultaneous processing
- **PipeDream**: Alternates forward and backward passes across workers
- **Zero Bubble Pipeline**: Eliminates pipeline bubbles through advanced scheduling

#### 4.4  Tensor Parallelism
Column-wise Parallel: X, A= [A1,  A2] O = [X@A1,  X@A2] 
● Row-wise Parallel X = [X1, X2], A = [A1 |  A2]  O = [X1 @ A1 + X2 @ A2] 
			
	
● Column_wise Parallel first followed by Row-wise Parallel which naturally expects to split input by columns. 
● TP splits the model vertically, partitioning the computation and parameters in each layer across multiple devices, requiring significant communication between each layer. As a result, they work well within a single node where the inter-GPU communication bandwidth is high, but the efficiency degrades quickly beyond a single node. 
● Megatron-LM: Open source implementation of Tensor Parallelism

splits matrix operations across GPUs, either column-wise or row-wise. Megatron-LM provides an open-source implementation of tensor parallelism for large language models.
- **Column-wise Parallel**: Splits matrices by columns
- **Row-wise Parallel**: Splits matrices by rows
- **Megatron-LM**: Open source implementation of tensor parallelism

#### 4.5 Context Parallelism
 splits sequence length across multiple GPUs, with each GPU handling a segment of the sequence. This is useful for very long sequences that don't fit on a single GPU. Context Parallelism is about how to parallelize the sequence length into multiple GPUs. During forward propagation, each GPU handles a segment of the sequence, storing only the necessary Key and Value (KV) pairs. In the backward pass, these KV pairs are reassembled across GPUs using advanced communication schemes like all-gather and reduce-scatter transformed into point-to-point communications in a ring topology. 


#### 4.6 Expert Parallelism (MoE)
Instead of every token being processed by the same dense network, we introduce a set of experts (sub-networks, usually feed-forward MLPs inside Transformer blocks). Each token is assigned (via a gating function) to one or a few experts. Experts are sharded across devices (GPUs/TPUs). Each device hosts a subset of experts, and tokens are routed to whichever device hosts their assigned expert.
● Mixture of Experts: A router (usually a softmax gate) decides which expert(s) handle each token.
Top-1 routing (Switch Transformer): pick the highest scoring expert.
Top-k routing (GShard, GLaM): pick k experts and combine outputs.
● Device Balance Loss: Some experts (on some GPUs) may get overloaded while others sit idle. This leads to device imbalance → bottlenecked training/inference. Add a regularization term (“device balance loss”) to the training objective that encourages an even distribution of tokens across devices.
● Communication Balance Loss: 
● Auxiliary Free Load Balancing: instead of adding a penalty in loss, use architectural o algorithmic tricks to achieve balance automatically - Randomized routing with constraints, Capacity-based routing (set a hard cap on tokens per expert), Priority dropping (drop excess tokens if an expert is full).

routes tokens to specialized expert networks instead of processing every token with the same dense network. Routing can be Top-1 (single expert) or Top-k (multiple experts), with the main challenge being load balancing across experts. The benefit is scaling model size without proportional compute increase.

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
