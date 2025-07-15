/**
 * Transformer block and complete Mini-GPT implementation
 * 
 * This combines attention, feed-forward, and layer normalization
 * into the complete GPT architecture.
 */

import { Tensor, layerNorm } from './tensor.js';
import { MultiHeadAttention, FeedForward } from './attention.js';
import { CodeTokenizer } from './tokenizer.js';

/**
 * Single transformer block
 * Structure: LayerNorm -> Attention -> Residual -> LayerNorm -> FeedForward -> Residual
 */
export class TransformerBlock {
  private attention: MultiHeadAttention;
  private feedForward: FeedForward;
  private ln1Gamma: Tensor; // Layer norm 1 scale
  private ln1Beta: Tensor;  // Layer norm 1 bias
  private ln2Gamma: Tensor; // Layer norm 2 scale
  private ln2Beta: Tensor;  // Layer norm 2 bias
  
  constructor(nEmbed: number, nHeads: number) {
    this.attention = new MultiHeadAttention(nEmbed, nHeads);
    this.feedForward = new FeedForward(nEmbed);
    
    // Layer normalization parameters
    this.ln1Gamma = Tensor.ones([nEmbed]);
    this.ln1Beta = Tensor.zeros([nEmbed]);
    this.ln2Gamma = Tensor.ones([nEmbed]);
    this.ln2Beta = Tensor.zeros([nEmbed]);
  }

  /**
   * Forward pass through transformer block
   */
  forward(x: Tensor): Tensor {
    // Pre-norm architecture (used in modern transformers)
    
    // Attention block with residual connection
    const norm1 = layerNorm(x, this.ln1Gamma, this.ln1Beta);
    const attnOut = this.attention.forward(norm1);
    const residual1 = x.add(attnOut); // Residual connection
    
    // Feed-forward block with residual connection
    const norm2 = layerNorm(residual1, this.ln2Gamma, this.ln2Beta);
    const ffOut = this.feedForward.forward(norm2);
    const residual2 = residual1.add(ffOut); // Residual connection
    
    return residual2;
  }

  /**
   * Get all learnable parameters
   */
  getParameters(): Tensor[] {
    return [
      ...this.attention.getParameters(),
      ...this.feedForward.getParameters(),
      this.ln1Gamma,
      this.ln1Beta,
      this.ln2Gamma,
      this.ln2Beta
    ];
  }
}

/**
 * GPT Model Configuration
 */
export interface GPTConfig {
  vocabSize: number;    // Size of vocabulary
  blockSize: number;    // Context length (sequence length)
  nLayers: number;      // Number of transformer blocks
  nHeads: number;       // Number of attention heads
  nEmbed: number;       // Embedding dimension
  dropout?: number;     // Dropout rate (not implemented yet)
}

/**
 * Complete Mini-GPT implementation
 */
export class MiniGPT {
  private config: GPTConfig;
  private tokenizer: CodeTokenizer;
  
  // Model components
  private tokenEmbedding!: Tensor;     // Token embeddings
  private positionEmbedding!: Tensor;  // Position embeddings
  private blocks!: TransformerBlock[];  // Transformer blocks
  private lnF!: { gamma: Tensor, beta: Tensor }; // Final layer norm
  private head!: Tensor;               // Output projection to vocabulary
  
  constructor(config: GPTConfig) {
    this.config = config;
    this.tokenizer = new CodeTokenizer(config.vocabSize);
    
    // Initialize model components
    this.initializeModel();
  }

  private initializeModel(): void {
    const { vocabSize, blockSize, nLayers, nHeads, nEmbed } = this.config;
    
    // Token and position embeddings
    this.tokenEmbedding = Tensor.random([vocabSize, nEmbed]);
    this.positionEmbedding = Tensor.random([blockSize, nEmbed]);
    
    // Transformer blocks
    this.blocks = [];
    for (let i = 0; i < nLayers; i++) {
      this.blocks.push(new TransformerBlock(nEmbed, nHeads));
    }
    
    // Final layer norm and output head
    this.lnF = {
      gamma: Tensor.ones([nEmbed]),
      beta: Tensor.zeros([nEmbed])
    };
    this.head = Tensor.random([nEmbed, vocabSize]);
    
    console.log(`Initialized Mini-GPT with ${this.getParameterCount()} parameters`);
  }

  /**
   * Forward pass through the complete model
   */
  forward(tokenIds: number[]): Tensor {
    const seqLength = tokenIds.length;
    const { blockSize } = this.config;
    
    if (seqLength > blockSize) {
      throw new Error(`Sequence length ${seqLength} exceeds block size ${blockSize}`);
    }
    
    // Step 1: Get token embeddings
    const tokEmb = this.getTokenEmbeddings(tokenIds);
    
    // Step 2: Add position embeddings
    const posEmb = this.getPositionEmbeddings(seqLength);
    let x = tokEmb.add(posEmb);
    
    // Step 3: Pass through transformer blocks
    for (const block of this.blocks) {
      x = block.forward(x);
    }
    
    // Step 4: Final layer norm
    x = layerNorm(x, this.lnF.gamma, this.lnF.beta);
    
    // Step 5: Project to vocabulary
    const logits = x.matmul(this.head); // [seqLength, vocabSize]
    
    return logits;
  }

  /**
   * Generate text given a prompt
   */
  generate(prompt: string, maxTokens: number = 100, temperature: number = 1.0): string {
    let tokenIds = this.tokenizer.encode(prompt);
    
    for (let i = 0; i < maxTokens; i++) {
      // Get predictions for current sequence
      const logits = this.forward(tokenIds);
      
      // Get logits for the last token (next token prediction)
      const lastTokenLogits = this.getLastTokenLogits(logits);
      
      // Convert logits to probabilities
      const probs = lastTokenLogits.softmax();
      
      // Sample next token
      const nextTokenId = this.tokenizer.sampleToken(
        Array.from(probs.data.slice(0, this.config.vocabSize)),
        temperature
      );
      
      // Add to sequence
      tokenIds.push(nextTokenId);
      
      // Stop if we hit EOS token
      if (nextTokenId === this.tokenizer.getTokenId(CodeTokenizer.EOS_TOKEN)) {
        break;
      }
      
      // Truncate if we exceed context length
      if (tokenIds.length > this.config.blockSize) {
        tokenIds = tokenIds.slice(-this.config.blockSize);
      }
    }
    
    return this.tokenizer.decode(tokenIds);
  }

  /**
   * Train the model on a dataset
   */
  async train(corpus: string[], epochs: number = 1): Promise<void> {
    console.log('Training tokenizer...');
    this.tokenizer.train(corpus);
    
    console.log('Preparing training data...');
    const trainingData = this.prepareTrainingData(corpus);
    
    console.log(`Training on ${trainingData.length} sequences for ${epochs} epochs`);
    console.log(`Sample training sequence length: ${trainingData[0]?.input.length || 0}`);
    
    for (let epoch = 0; epoch < epochs; epoch++) {
      let totalLoss = 0;
      let batches = 0;
      const batchSize = 32; // Process multiple sequences at once
      const totalBatches = Math.ceil(trainingData.length / batchSize);
      
      console.log(`Starting epoch ${epoch + 1}/${epochs} with ${totalBatches} batches (batch size: ${batchSize})`);
      
      // Process in batches for efficiency
      for (let batchStart = 0; batchStart < trainingData.length; batchStart += batchSize) {
        const batchEnd = Math.min(batchStart + batchSize, trainingData.length);
        const batch = trainingData.slice(batchStart, batchEnd);
        
        let batchLoss = 0;
        
        // Process each sequence in the batch
        for (const { input, target } of batch) {
          // Forward pass
          const logits = this.forward(input);
          
          // Compute loss (cross-entropy)
          const loss = this.computeLoss(logits, target);
          batchLoss += loss;
        }
        
        // Average loss for this batch
        const avgBatchLoss = batchLoss / batch.length;
        totalLoss += avgBatchLoss;
        batches++;
        
        // Backward pass (simplified - no actual gradients yet)
        // In a real implementation, this would compute gradients and update weights
        
        // Progress updates
        if (batches % 10 === 0) {
          const progress = ((batches / totalBatches) * 100).toFixed(1);
          console.log(`Epoch ${epoch + 1}, Batch ${batches}/${totalBatches} (${progress}%), Loss: ${(totalLoss / batches).toFixed(4)}`);
        }
      }
      
      const avgLoss = totalLoss / batches;
      console.log(`Epoch ${epoch + 1} completed. Average loss: ${avgLoss.toFixed(4)}`);
    }
  }

  /**
   * Prepare training data from corpus
   */
  private prepareTrainingData(corpus: string[]): Array<{ input: number[], target: number[] }> {
    const sequences: Array<{ input: number[], target: number[] }> = [];
    const maxSequencesPerFile = 20; // Fewer sequences per file for faster training
    
    console.log(`Processing ${corpus.length} files...`);
    
    for (let i = 0; i < corpus.length; i++) {
      const text = corpus[i];
      const tokenIds = this.tokenizer.encode(text);
      
      // Skip very small texts that won't generate useful sequences
      if (tokenIds.length < this.config.blockSize + 1) {
        continue;
      }
      
      const seqs = this.tokenizer.createTrainingSequences(tokenIds, this.config.blockSize);
      
      // Limit sequences per file to prevent memory explosion
      const limitedSeqs = seqs.slice(0, maxSequencesPerFile);
      sequences.push(...limitedSeqs);
      
      // Progress update
      if (i % 10 === 0) {
        console.log(`Processed ${i + 1}/${corpus.length} files. Total sequences: ${sequences.length}`);
      }
    }
    
    console.log(`Final training data: ${sequences.length} sequences`);
    return sequences;
  }

  /**
   * Get token embeddings for a sequence
   */
  private getTokenEmbeddings(tokenIds: number[]): Tensor {
    const seqLength = tokenIds.length;
    const { nEmbed } = this.config;
    
    const embeddings = Tensor.zeros([seqLength, nEmbed]);
    
    for (let i = 0; i < seqLength; i++) {
      const tokenId = tokenIds[i];
      for (let j = 0; j < nEmbed; j++) {
        const value = this.tokenEmbedding.get([tokenId, j]);
        embeddings.set([i, j], value);
      }
    }
    
    return embeddings;
  }

  /**
   * Get position embeddings for a sequence
   */
  private getPositionEmbeddings(seqLength: number): Tensor {
    const { nEmbed } = this.config;
    const embeddings = Tensor.zeros([seqLength, nEmbed]);
    
    for (let i = 0; i < seqLength; i++) {
      for (let j = 0; j < nEmbed; j++) {
        const value = this.positionEmbedding.get([i, j]);
        embeddings.set([i, j], value);
      }
    }
    
    return embeddings;
  }

  /**
   * Extract logits for the last token
   */
  private getLastTokenLogits(logits: Tensor): Tensor {
    const [seqLength, vocabSize] = logits.shape;
    const lastLogits = Tensor.zeros([vocabSize]);
    
    for (let i = 0; i < vocabSize; i++) {
      const value = logits.get([seqLength - 1, i]);
      lastLogits.set([i], value);
    }
    
    return lastLogits;
  }

  /**
   * Compute cross-entropy loss
   */
  private computeLoss(logits: Tensor, targets: number[]): number {
    const [seqLength, vocabSize] = logits.shape;
    let totalLoss = 0;
    
    for (let i = 0; i < seqLength; i++) {
      const targetId = targets[i];
      
      // Get logits for this position
      const posLogits = Tensor.zeros([vocabSize]);
      for (let j = 0; j < vocabSize; j++) {
        posLogits.set([j], logits.get([i, j]));
      }
      
      // Apply softmax
      const probs = posLogits.softmax();
      
      // Cross-entropy loss
      const targetProb = probs.get([targetId]);
      totalLoss += -Math.log(Math.max(targetProb, 1e-10)); // Avoid log(0)
    }
    
    return totalLoss / seqLength;
  }

  /**
   * Count total parameters
   */
  private getParameterCount(): number {
    let count = 0;
    
    // Embeddings
    count += this.tokenEmbedding.size;
    count += this.positionEmbedding.size;
    
    // Blocks
    for (const block of this.blocks) {
      for (const param of block.getParameters()) {
        count += param.size;
      }
    }
    
    // Final layer norm and head
    count += this.lnF.gamma.size;
    count += this.lnF.beta.size;
    count += this.head.size;
    
    return count;
  }

  /**
   * Get tokenizer for external use
   */
  getTokenizer(): CodeTokenizer {
    return this.tokenizer;
  }

  /**
   * Get model configuration
   */
  getConfig(): GPTConfig {
    return { ...this.config };
  }
}