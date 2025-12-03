
## Large Language Model Optimization: Memory, Compute, and Inference Techniques

Here is a collection of my personal notes from preparing for interviews at several leading AI labs and revisiting the core ideas behind efficient large-scale model training. Along the way, I compiled these notes, part interview preparation, part personal revision, and thought they might be worth sharing. 

Training and deploying large language models efficiently has become one of the most critical challenges in modern AI. As models scale into the billions of parameters, traditional approaches quickly start to break down. The techniques outlined here highlight some of the essential optimization strategies that have emerged as industry standards for building and deploying large-scale models.

These notes aren’t meant to be exhaustive or perfectly structured, but rather a reflection of the key concepts and techniques that repeatedly came up in technical discussions and that form the backbone of large-scale model development. This is also not a deeply detailed discussion of each topic, but rather a high-level overview of the core approaches and concepts providing a concise summary of the most common practices and ideas used in efficient training and inference. I hope they’re useful to anyone looking to dive deeper into these optimization techniques or preparing for similar technical interviews.

---

### 1. Memory Optimization Techniques

Memory is the biggest bottleneck in LLM training and inference. As models scale to billions of parameters, traditional memory management approaches become insufficient. The techniques in this section focus on reducing memory footprint while maintaining model quality, enabling the training and deployment of larger models on existing hardware.

#### 1.1 Flash Attention

The attention mechanism has quadratic time and memory complexity in sequence length, presenting significant runtime and memory challenges for longer sequences.
Flash Attention reduces attention memory complexity through tiling and recomputation techniques. Instead of processing entire attention matrices at once, it processes attention in blocks and stores normalization factors instead of full attention matrices. The tiling technique decomposes inputs based on shared memory size, while recomputation stores softmax normalization factors (linear to sequence length) instead of softmax results (quadratic to sequence length).

**Tiling Technique**: Decomposes inputs based on shared memory size and calculates softmax one tile at a time. Instead of working on entire query, key, value tensors at once, it makes several passes and combines results in subsequent steps.

**Recomputation Technique**: Stores softmax normalization factors (linear to sequence length) instead of softmax results (quadratic to sequence length), using these factors to recompute attention scores. This reduces memory requirements and I/O traffic between global and shared memory.

**Additional Resources**:
[1] [Matrix multiplication tiling](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)
[2] [Online softmax and tiling](https://www.youtube.com/watch?v=LKwyHWYEIMQ&t=14s)

#### 1.2 Multi-Query and Grouped Query Attention

- **MQA (Multi-Query Attention)**: Reduces memory by sharing keys and values across attention heads
- **GQA (Grouped Query Attention)**: Balances efficiency and quality by grouping queries

#### 1.3 Activation Checkpointing

Input activations easily saturate device memory when training LLMs with large sequence lengths or micro-batch sizes. Checkpointing a few activations and recomputing the rest reduces device memory requirements.

---

### 2. Compute Optimization Techniques

Maximize GPU utilization and reduce computational overhead through smarter data handling and model architectures.

#### 2.1 Sequence Packing

A training technique where multiple training sequences are concatenated into one long sequence. This eliminates padding and allows more tokens to be processed per micro-batch, maximizing both GPU compute and memory utilization.

#### 2.2 Efficient Transformers

As sequence lengths and dataset sizes grow, standard transformer architectures become prohibitively expensive due to their quadratic time and memory complexity with respect to sequence length. Efficient transformer techniques have been developed to address these limitations, enabling the processing of much longer sequences and larger datasets by reducing computational and memory requirements.

- **BigBird**: Uses a combination of local, random, and global attention patterns to reduce complexity to O(n).
- **Longformer**: Utilizes sliding window (local) attention combined with global attention for improved efficiency.
- **Low-Rank Approximations**: Projects key and value matrices into lower-dimensional spaces.
- **LongNet**: At lower layers, tokens attend to nearby tokens (small dilation). At higher layers, dilation factor grows, allowing tokens to reach further. Scales linearly with sequence length O(Nd).

**Additional Resources**: [1][Scaling Transformers with LongNet](https://www.youtube.com/watch?v=nC2nU9j9DVQ)

---

### 3. Inference Optimization Techniques

Inference is where most production costs occur. These techniques dramatically speed up generation while maintaining quality.

#### 3.1 KV Caching

KV caching works by storing the key and value tensors computed for each token as the model generates a sequence. In autoregressive generation, at each new step, the model only needs to compute attention for the newly generated token, using the cached keys and values from previous steps instead of recomputing them for the entire sequence. This drastically reduces both computation and memory usage, as the attention mechanism can reuse the stored key-value pairs for all previous tokens.

**Advanced KV Cache Optimizations**:
- Grouped Multi Query Attention: Reduces KV cache memory by grouping multiple queries with same keys and values
- Multi-head Latent Attention: Projects K, V, Q into lower-dimensional latent space, computing attention in latent space then projecting back
- Cross Layer KV-sharing: Ties KV cache across neighboring attention layers
- Interleaving Local and Global Attention: Uses global attention in every 4-6 layers

**Additional Resources**:
[1] [KV Caching Video](https://www.youtube.com/watch?v=Mn_9W1nCFLo&t=3869s)
[2] [FLOPS computation efficiency with KV cache](https://docs.google.com/presentation/d/14hK7SmkUNfSEIRGyptFD2bGO7K9sJOTnwjAVg3vgg6g/edit?slide=id.g286de50af37_0_933#slide=id.g286de50af37_0_933)

#### 3.2 Stateful Caching

Stateful caching stores conversation history using rolling hashes, allowing reuse of overlapping prefixes. For example, if "Hello, how are you?" is cached, it can be reused when the new prefix is "Hello, how are you doing today?" The cache is organized in a tree structure with LRU eviction to manage memory efficiently. 
For a new query, compute rolling hashes for all its prefixes and find the longest cached match.Load the KV tensors from cache and continue computation only from the new tokens.
KV cache organized in a tree structure with LRU (least recently used) eviction, so you can drop old contexts if memory is full.


#### 3.3 Speculative Decoding

Speculative decoding uses a smaller draft model to generate responses, then uses the target model to verify them, achieving 2-3x speedup in inference. The draft model must be fast and well-aligned with the target model for this technique to be effective.

#### 3.4 Quantization Techniques
Compressing a model by representing weights/activations with fewer bits instead of standard fp32 (32-bit float).

**Quantization Types**:
- Min/Max: Simple but susceptible to outliers
- MSE: Minimizes MSE between original and quantized values
- Cross-entropy: Preserves order of largest values after quantization for softmax; argmin(softmax(v), softmax(v’))


**Post-Training Quantization (PTQ)**: After fully training a model, its weights are converted to lower precision (e.g., int8 or float16) without further training. PTQ is generally inexpensive to implement compared to retraining, but as model sizes scale to billions of parameters, outlier activations of large magnitude can appear in transformer layers, making naive low-bit quantization less effective. To address this, quantization-aware observers are attached to collect statistics (such as mean and standard deviation) on the input data, which are then used to determine quantization parameters.

**Mixed-Precision Quantization**: Instead of quantizing all weights and activations to the same bit width, mixed-precision quantization assigns different precisions to different parts of the model. For example, sensitive layers or activations may use higher precision (e.g., 8 or 16 bits), while less sensitive parts use lower precision. This approach balances memory savings and model accuracy, and is especially useful for large models where uniform low-bit quantization would degrade performance. Mixed-precision quantization can be applied separately to weights and activations, optimizing for both efficiency and quality.

**Quantization-Aware Training (QAT)**: Quantization is applied during pre-training or further fine-tuning. Siimulate quantization and dequantization in the forward pass. This simulaes the quantization error and acts as a regualizer to make the model robust to it. Backpropagation: Quantization is not differentiable, so gradients are approximated using the straight-through estimator (STE), which sets the gradient to 1 within the quantization range (alpha, beta) and 0 outside.

**Additional Resources**:
[1] [Quantization Video](https://www.youtube.com/watch?v=0VdNflU08yA)
[2] [Lilian Weng's Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)

---

### 4. Training Optimization

Training large models requires sophisticated parallelism strategies. Know the different approaches and their trade-offs.

#### 4.1 Mixed Precision Training

Mixed precision training leverages lower-precision number formats, most notably bfloat16 (Brain Floating Point 16), to reduce memory usage and accelerate training. Bfloat16 has the same exponent range as standard 32-bit floats (fp32), but with fewer mantissa bits, allowing it to represent very large and very small numbers, which helps preserve the dynamic range needed for deep learning. However, because bfloat16 and fp16 have reduced precision, gradients can underflow or overflow during backpropagation, leading to instability. To address this, loss scaling is used: the loss is multiplied by a large constant before backpropagation, and gradients are later rescaled down, preventing small gradients from vanishing due to limited precision. Careful management of loss scaling is essential to maintain numerical stability and fully realize the benefits of mixed precision training.

#### 4.2 Data Parallelism

**DataParallel**: Single-process, multi-threaded approach that works when the model fits on a single GPU. Each GPU keeps a copy of the model, processes different microbatches, then averages gradients across GPUs. The main bottleneck is that it relies on single-process, multi-threaded communication, leading to inefficient inter-GPU communication and potential slowdowns due to CPU overhead. DataParallel runs in a single process with multiple threads, so it suffers from GIL contention.

**Synchronization Approaches**:
At the end of each minibatch, workers need to synchronize gradients or weights to avoid staleness. There are two main synchronization approaches:

- **Bulk Synchronous Parallel (BSP)**: Workers synchronize at the end of every minibatch. It prevents model weights staleness and provides good learning efficiency, but each machine has to halt and wait for others to send gradients.

- **Asynchronous Parallel (ASP)**: Every GPU worker processes the data asynchronously with no waiting or stalling. However, it can easily lead to stale weights being used and thus lower the statistical learning efficiency. Even though it increases the computation time, it may not speed up training time to convergence.

**Distributed Data Parallel (DDP)**: Each GPU has its own process and can work on multiple nodes/machines. Uses Ring All-Reduce algorithm to avoid central bottlenecks. DDP has lower communication overhead compared to DataParallel.

**ZeRO (Zero Redundancy Optimizer)**: Not just the model parameters and gradients, but the optimizer state (including Adam momentum, variance) also takes a lot of memory. ZeRO-DP has three main optimization stages:

- **Optimizer State Partitioning**: 4x memory reduction, same communication volume as DP. Gradient computation can be done independently for each GPU. When it is being done for parameters that are not on the current GPU, we incur a communication cost but this is the same as gradient averaging in DP. So always do this ZeRO stage-1.

- **Gradient Partitioning**: 8x memory reduction, same communication volume as DP. This is similar to optimizer state partitioning in practice. Optimizer states are calculated per parameter anyway so this doesn't incur any extra cost.

- **Parameter Partitioning**: Memory reduction is linear with DP degree. For example, splitting across 64 GPUs will yield a 64x memory reduction. There is a modest 50% increase in communication volume. This works because at any time doing forward or backward, only a subset of parameters (in a layer) are required for the operation. At best you're gonna need memory equivalent to a layer size. The model parameters can be sliced in any manner (vertically or horizontally). The way it's different from tensor parallelism or pipeline parallelism is that every computation still happens on each GPU using full tensors, just that the parameters are not all on single GPU.

**Communication Primitives**:
- **All-Reduce**: Each process starts with its own data and ends up with the sum (or other reduction) of all data across processes. Commonly used for synchronizing gradients. Can be implemented as a combination of reduce-scatter followed by all-gather.  
- **Ring All-Reduce**: Each GPU sends and receives data in a ring topology, minimizing bandwidth bottlenecks. Communication overhead: 2 × (N-1) × X/N bytes, where N is the number of GPUs and X is the data size.
- **Reduce-Scatter**: Each process reduces (e.g., sums) a chunk of data across all processes and keeps only its own chunk. Used as the first step in optimized all-reduce.
- **All-Gather**: Each process gathers data chunks from all other processes so that everyone ends up with the complete data. Used as the second step in optimized all-reduce.
- **Broadcast**: One process sends data to all others (e.g., distributing model weights at initialization).
- **Reduce**: Data from all processes is reduced (e.g., summed) and the result is sent to a single process.
- **Scatter**: One process splits data and sends different chunks to each process.
- **Gather**: Each process sends data to a single process, which collects all the data.

These primitives are the building blocks for distributed training and are used to synchronize parameters, gradients, and optimizer states efficiently across multiple GPUs or nodes.
All-reduce = reduce-scatter + all-gather. Ring-reduce overhead: 2 × (N-1) × X/N bytes

**Additional Resources**:
[1] [Scaling ML Models](https://www.youtube.com/watch?v=hc0u4avAkuM)
[2] [Training Optimization](https://www.youtube.com/watch?v=toUSzwR0EV8)
[3] [Understanding data parallelism, ZeRO, FSDP](https://www.youtube.com/watch?v=UVX7SYGCKkA)
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

**Additional Resources**:
[1] [GPipe Paper](https://arxiv.org/abs/1811.06965)
[2] [PipeDream Paper](https://arxiv.org/abs/1806.03377)
[3] [Zero Bubble Pipeline](https://arxiv.org/abs/2011.06448)

#### 4.4 Tensor Parallelism

**Tensor Parallelism** splits large matrix multiplications across multiple devices, enabling efficient scaling of model size. The two most common approaches are **column-wise** and **row-wise** parallelism.

**(1) Column-wise Parallelism**

The weight matrix is split along its columns. Each device holds a subset of columns and computes its portion of the output.
-   If input X and weight A = [A₁, A₂, ..., Aₙ], then  
    Output O = [X @ A₁, X @ A₂, ..., X @ Aₙ]  
    (Each device computes X @ Aᵢ for its assigned columns.)

-  
  ```
  Input X
    |
    |         ┌─────┬─────┬─────┐
    |         │ A₁  │ A₂  │ A₃  │   (A split column-wise)
    |         └─────┴─────┴─────┘
    |           |     |     |
    |           v     v     v
    |        Device 1 2 ... n
    |           |     |     |
    |         [X@A₁] [X@A₂] [X@A₃]
    |___________|_____|_____|
                |
             Concatenate
                |
              Output O
  ```

- After each device computes its partial output (X @ Aᵢ), the results must be **gathered and concatenated** (usually via an all-gather operation) to form the full output. This step incurs communication overhead proportional to the output size and the number of devices. The protocol is typically an **all-gather** across devices.

**(2) Row-wise Parallelism**

In row-wise (sometimes called output) parallelism, the **input X is split column-wise** across devices, and the weight matrix A is split row-wise. Each device receives a slice of the input (a subset of columns of X) and a corresponding slice of the weight matrix (a subset of rows of A). Each device computes a partial output of the same shape as the final output, and the final result is obtained by **summing (reducing)** these partial outputs across all devices.

- If input X = [X₁, X₂, ..., Xₙ] (split column-wise) and weight A = [A₁; A₂; ...; Aₙ] (split row-wise), then  
  Each device computes: Oᵢ = Xᵢ @ Aᵢ  
  The final output: O = sum(O₁, O₂, ..., Oₙ) (element-wise addition across devices)

- 
  ```
  Input X split by columns: [X₁ | X₂ | X₃]
    |             |      |      |
    |             v      v      v
    |         Device 1 Device 2 Device 3
    |           |        |        |
    |         ┌─────┐  ┌─────┐  ┌─────┐
    |         │ A₁  │  │ A₂  │  │ A₃  │   (A split row-wise)
    |         └─────┘  └─────┘  └─────┘
    |           |        |        |
    |        [X₁@A₁] [X₂@A₂] [X₃@A₃]
    |_____________|______|______|
                  |
               Reduce (sum)
                  |
               Output O
  ```

- After each device computes its partial output, the results must be **summed across devices** to obtain the final output. This introduces communication overhead proportional to the output size and the number of devices. The protocol is typically an **all-reduce** across devices.


**Implementation**: Megatron-LM provides open-source tensor parallelism implementation.  
For distributed attention in transformers, Megatron splits the Q, K, and V linear projections across devices, computes local attention scores and softmax independently, and then uses collective communication (such as all-reduce) to aggregate the partial attention outputs across devices for the final result.

**Additional Resources**:
[1] [Megatron-LM Paper](https://arxiv.org/abs/1909.08053)
[2] [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM)

#### 4.5 Context Parallelism

Context Parallelism is about how to parallelize the sequence length into multiple GPUs. During forward propagation, each GPU handles a segment of the sequence, storing only the necessary Key and Value (KV) pairs. In the backward pass, these KV pairs are reassembled across GPUs using advanced communication schemes like all-gather and reduce-scatter transformed into point-to-point communications in a ring topology.

**Additional Resources**:
[1] [Context Parallelism Paper](https://arxiv.org/abs/2105.03824)
[2] [Sequence Parallelism](https://arxiv.org/abs/2104.04473)

#### 4.6 Expert Parallelism (MoE)
Instead of every token being processed by the same dense network, we introduce a set of experts (sub-networks, usually feed-forward MLPs inside Transformer blocks). Each token is assigned (via a gating function) to one or a few experts. Experts are sharded across devices (GPUs/TPUs). Each device hosts a subset of experts, and tokens are routed to whichever device hosts their assigned expert.

**Mixture of Experts (MoE)**: Instead of every token being processed by the same dense network, a set of experts (sub-networks, typically feed-forward MLPs) is introduced. A router—usually implemented as a softmax gate—assigns each token to one or more experts based on routing strategies:
- Top-1 routing (e.g., Switch Transformer): Each token is sent to the single highest-scoring expert.
- Top-k routing (e.g., GShard, GLaM): Each token is sent to the top k experts, and their outputs are combined.

**Load Balancing Challenges**:
In practice, load balancing can be a significant challenge in expert parallelism. Some experts, and thus some GPUs, may become overloaded with tokens while others remain underutilized. This imbalance can create bottlenecks during training or inference, reducing overall efficiency.

- Device Balance Loss: Add a regularization term ("device balance loss") to the training objective that encourages an even distribution of tokens across devices.
- Communication Balance Loss: Additional loss term to balance communication patterns.
- Auxiliary Free Load Balancing: Instead of adding a penalty in loss, use architectural or algorithmic tricks to achieve balance automatically:
  - Randomized routing with constraints
  - Capacity-based routing (set a hard cap on tokens per expert)
  - Priority dropping (drop excess tokens if an expert is full)

**Additional Resources**:
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
- [7] [Efficient Training and Tradeoffs](https://www.youtube.com/watch?v=UVX7SYGCKkA)

---

### Conclusion
Optimizing large language models requires careful consideration across multiple dimensions. The techniques discussed here represent the current state-of-the-art in LLM optimization, from memory-efficient attention mechanisms to advanced parallelism strategies. As models continue to grow, these optimization techniques become increasingly critical for practical deployment.

Thanks for dropping by and reading till the end! Hope this was helpful in your journey!

