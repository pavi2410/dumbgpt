/**
 * Multi-Head Attention mechanism for Mini-GPT
 * 
 * This is the core of the transformer architecture. Attention allows the model
 * to focus on different parts of the input sequence when making predictions.
 */

import { Tensor, gelu } from './tensor.js';

export class MultiHeadAttention {
  private nHeads: number;
  private nEmbed: number;
  private headDim: number;
  private scale: number;
  
  // Learned parameters
  private qWeight: Tensor; // Query projection
  private kWeight: Tensor; // Key projection  
  private vWeight: Tensor; // Value projection
  private outWeight: Tensor; // Output projection
  
  constructor(nEmbed: number, nHeads: number) {
    this.nEmbed = nEmbed;
    this.nHeads = nHeads;
    this.headDim = nEmbed / nHeads;
    this.scale = 1.0 / Math.sqrt(this.headDim);
    
    if (nEmbed % nHeads !== 0) {
      throw new Error(`nEmbed (${nEmbed}) must be divisible by nHeads (${nHeads})`);
    }
    
    // Initialize weight matrices
    this.qWeight = Tensor.random([nEmbed, nEmbed]);
    this.kWeight = Tensor.random([nEmbed, nEmbed]);
    this.vWeight = Tensor.random([nEmbed, nEmbed]);
    this.outWeight = Tensor.random([nEmbed, nEmbed]);
  }

  /**
   * Forward pass through multi-head attention
   * 
   * @param x Input tensor of shape [seqLength, nEmbed]
   * @returns Output tensor of shape [seqLength, nEmbed]
   */
  forward(x: Tensor): Tensor {
    const [seqLength, nEmbed] = x.shape;
    
    // Step 1: Project input to queries, keys, and values
    const queries = x.matmul(this.qWeight); // [seqLength, nEmbed]
    const keys = x.matmul(this.kWeight);    // [seqLength, nEmbed]
    const values = x.matmul(this.vWeight);  // [seqLength, nEmbed]
    
    // Step 2: Reshape for multi-head attention
    // [seqLength, nEmbed] -> [seqLength, nHeads, headDim]
    const qHeads = this.reshapeForHeads(queries, seqLength);
    const kHeads = this.reshapeForHeads(keys, seqLength);
    const vHeads = this.reshapeForHeads(values, seqLength);
    
    // Step 3: Compute attention for each head
    const attentionOutputs: Tensor[] = [];
    
    for (let h = 0; h < this.nHeads; h++) {
      const qHead = this.getHead(qHeads, h, seqLength);
      const kHead = this.getHead(kHeads, h, seqLength);
      const vHead = this.getHead(vHeads, h, seqLength);
      
      const headOutput = this.scaledDotProductAttention(qHead, kHead, vHead);
      attentionOutputs.push(headOutput);
    }
    
    // Step 4: Concatenate heads
    const concatenated = this.concatenateHeads(attentionOutputs);
    
    // Step 5: Final output projection
    const output = concatenated.matmul(this.outWeight);
    
    return output;
  }

  /**
   * Scaled dot-product attention (the core attention mechanism)
   * 
   * Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
   */
  private scaledDotProductAttention(q: Tensor, k: Tensor, v: Tensor): Tensor {
    const [seqLength, headDim] = q.shape;
    
    // Step 1: Compute attention scores (Q * K^T)
    const scores = q.matmul(k.transpose()); // [seqLength, seqLength]
    
    // Step 2: Scale by sqrt(headDim)
    const scaledScores = scores.mul(this.scale);
    
    // Step 3: Apply causal mask (for autoregressive generation)
    const maskedScores = this.applyCausalMask(scaledScores);
    
    // Step 4: Apply softmax to get attention weights
    const attentionWeights = maskedScores.softmax(-1);
    
    // Step 5: Apply attention to values
    const output = attentionWeights.matmul(v); // [seqLength, headDim]
    
    return output;
  }

  /**
   * Apply causal mask to prevent the model from looking at future tokens
   * This is crucial for autoregressive generation (predicting next token)
   */
  private applyCausalMask(scores: Tensor): Tensor {
    const [seqLength, _] = scores.shape;
    const masked = scores.clone();
    
    // Set upper triangular part to -infinity (will become 0 after softmax)
    for (let i = 0; i < seqLength; i++) {
      for (let j = i + 1; j < seqLength; j++) {
        masked.set([i, j], -Infinity);
      }
    }
    
    return masked;
  }

  /**
   * Reshape tensor for multi-head processing
   */
  private reshapeForHeads(tensor: Tensor, seqLength: number): Tensor {
    // [seqLength, nEmbed] -> [seqLength * nHeads, headDim]
    return tensor.reshape([seqLength * this.nHeads, this.headDim]);
  }

  /**
   * Extract a specific head from the reshaped tensor
   */
  private getHead(tensor: Tensor, headIndex: number, seqLength: number): Tensor {
    const headOutput = Tensor.zeros([seqLength, this.headDim]);
    
    for (let i = 0; i < seqLength; i++) {
      for (let j = 0; j < this.headDim; j++) {
        const sourceIndex = i * this.nHeads + headIndex;
        const value = tensor.get([sourceIndex, j]);
        headOutput.set([i, j], value);
      }
    }
    
    return headOutput;
  }

  /**
   * Concatenate outputs from all attention heads
   */
  private concatenateHeads(heads: Tensor[]): Tensor {
    const [seqLength, _] = heads[0].shape;
    const output = Tensor.zeros([seqLength, this.nEmbed]);
    
    for (let i = 0; i < seqLength; i++) {
      for (let h = 0; h < this.nHeads; h++) {
        for (let j = 0; j < this.headDim; j++) {
          const value = heads[h].get([i, j]);
          const outputIndex = h * this.headDim + j;
          output.set([i, outputIndex], value);
        }
      }
    }
    
    return output;
  }

  /**
   * Get all learnable parameters for training
   */
  getParameters(): Tensor[] {
    return [this.qWeight, this.kWeight, this.vWeight, this.outWeight];
  }

  /**
   * Debug: Print attention weights for visualization
   */
  debugAttentionWeights(x: Tensor): void {
    const [seqLength, nEmbed] = x.shape;
    
    console.log('\n=== Attention Debug ===');
    console.log(`Input shape: [${seqLength}, ${nEmbed}]`);
    console.log(`Heads: ${this.nHeads}, Head dim: ${this.headDim}`);
    
    // Just show first head for debugging
    const queries = x.matmul(this.qWeight);
    const keys = x.matmul(this.kWeight);
    
    const qHeads = this.reshapeForHeads(queries, seqLength);
    const kHeads = this.reshapeForHeads(keys, seqLength);
    
    const qHead = this.getHead(qHeads, 0, seqLength);
    const kHead = this.getHead(kHeads, 0, seqLength);
    
    const scores = qHead.matmul(kHead.transpose()).mul(this.scale);
    const maskedScores = this.applyCausalMask(scores);
    const weights = maskedScores.softmax(-1);
    
    console.log('Attention weights (first head):');
    for (let i = 0; i < Math.min(5, seqLength); i++) {
      const row = [];
      for (let j = 0; j < Math.min(5, seqLength); j++) {
        row.push(weights.get([i, j]).toFixed(3));
      }
      console.log(`Token ${i}: [${row.join(', ')}${seqLength > 5 ? '...' : ''}]`);
    }
  }
}

/**
 * Feed-forward network used in transformer blocks
 * This is a simple 2-layer MLP with GELU activation
 */
export class FeedForward {
  private fc1: Tensor; // First linear layer
  private fc2: Tensor; // Second linear layer
  private nEmbed: number;
  private nFF: number;
  
  constructor(nEmbed: number, nFF: number = 4 * nEmbed) {
    this.nEmbed = nEmbed;
    this.nFF = nFF;
    
    // Initialize weights
    this.fc1 = Tensor.random([nEmbed, nFF]);
    this.fc2 = Tensor.random([nFF, nEmbed]);
  }

  /**
   * Forward pass through feed-forward network
   */
  forward(x: Tensor): Tensor {
    // First layer + GELU activation
    const h1 = gelu(x.matmul(this.fc1));
    
    // Second layer
    const output = h1.matmul(this.fc2);
    
    return output;
  }

  /**
   * Get all learnable parameters
   */
  getParameters(): Tensor[] {
    return [this.fc1, this.fc2];
  }
}