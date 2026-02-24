import llmOptimizationMd from '../pages/posts/llm-optimization.md?raw';

const posts = [
  {
    id: 'llm-optimization',
    title: 'LLM Optimization Notes: Memory, Compute & Inference Techniques',
    date: 'October 2025',
    author: 'Gauri Gupta',
    categories: ['LLMs', 'Distributed ML', 'Optimization'],
    excerpt:
      'These are my job preparation notes on optimizing the training and inference of large language models. This covers a range of techniques to improve memory usage, computational efficiency, and inference optimization techniques along with advanced parallelism strategies used in LLMs.',
    content: llmOptimizationMd,
  }
];

export { posts };
export default posts;
