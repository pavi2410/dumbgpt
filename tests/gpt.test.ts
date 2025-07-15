/**
 * Test suite for Mini-GPT implementation
 * Run with: bun test
 */

import { describe, test, expect } from 'bun:test';
import { Tensor, CodeTokenizer, MultiHeadAttention, MiniGPT, type GPTConfig } from '../src/models/index.js';

describe('Tensor Operations', () => {
  test('should create tensor with correct shape', () => {
    const tensor = new Tensor([2, 3]);
    expect(tensor.shape).toEqual([2, 3]);
    expect(tensor.size).toBe(6);
  });

  test('should perform matrix multiplication', () => {
    const a = new Tensor([2, 3], [1, 2, 3, 4, 5, 6]);
    const b = new Tensor([3, 2], [1, 2, 3, 4, 5, 6]);
    const result = a.matmul(b);
    
    expect(result.shape).toEqual([2, 2]);
    expect(result.get([0, 0])).toBe(22); // 1*1 + 2*3 + 3*5 = 22
    expect(result.get([0, 1])).toBe(28); // 1*2 + 2*4 + 3*6 = 28
  });

  test('should apply softmax correctly', () => {
    const tensor = new Tensor([3], [1, 2, 3]);
    const result = tensor.softmax();
    
    // Sum should be approximately 1
    const sum = result.data[0] + result.data[1] + result.data[2];
    expect(Math.abs(sum - 1)).toBeLessThan(0.001);
    
    // Values should be positive
    expect(result.data[0]).toBeGreaterThan(0);
    expect(result.data[1]).toBeGreaterThan(0);
    expect(result.data[2]).toBeGreaterThan(0);
  });

  test('should transpose 2D tensor', () => {
    const tensor = new Tensor([2, 3], [1, 2, 3, 4, 5, 6]);
    const result = tensor.transpose();
    
    expect(result.shape).toEqual([3, 2]);
    expect(result.get([0, 0])).toBe(1);
    expect(result.get([0, 1])).toBe(4);
    expect(result.get([1, 0])).toBe(2);
    expect(result.get([1, 1])).toBe(5);
  });
});

describe('CodeTokenizer', () => {
  test('should tokenize JavaScript code', () => {
    const tokenizer = new CodeTokenizer(1000);
    const code = 'function add(a, b) { return a + b; }';
    
    // Train with a simple corpus
    tokenizer.train([code]);
    
    const tokens = tokenizer.encode(code);
    expect(tokens.length).toBeGreaterThan(0);
    
    // Should start with BOS token
    expect(tokens[0]).toBe(tokenizer.getTokenId(CodeTokenizer.BOS_TOKEN));
    
    // Should end with EOS token
    expect(tokens[tokens.length - 1]).toBe(tokenizer.getTokenId(CodeTokenizer.EOS_TOKEN));
  });

  test('should decode tokens back to text', () => {
    const tokenizer = new CodeTokenizer(1000);
    const code = 'function test() {}';
    
    tokenizer.train([code]);
    
    const tokens = tokenizer.encode(code);
    const decoded = tokenizer.decode(tokens);
    
    // Should contain the original function keyword
    expect(decoded).toContain('function');
    expect(decoded).toContain('test');
  });

  test('should create training sequences', () => {
    const tokenizer = new CodeTokenizer(1000);
    const tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const sequences = tokenizer.createTrainingSequences(tokens, 4);
    
    expect(sequences.length).toBe(6); // 10 - 4 = 6 sequences
    expect(sequences[0].input).toEqual([1, 2, 3, 4]);
    expect(sequences[0].target).toEqual([2, 3, 4, 5]);
    expect(sequences[1].input).toEqual([2, 3, 4, 5]);
    expect(sequences[1].target).toEqual([3, 4, 5, 6]);
  });

  test('should sample tokens with temperature', () => {
    const tokenizer = new CodeTokenizer(1000);
    const probs = [0.1, 0.2, 0.3, 0.4]; // Should favor last token
    
    // With temperature = 0 (greedy), should always pick index 3
    const greedyToken = tokenizer.sampleToken(probs, 0);
    expect(greedyToken).toBe(3);
    
    // With temperature > 0, should return valid index
    const sampledToken = tokenizer.sampleToken(probs, 1.0);
    expect(sampledToken).toBeGreaterThanOrEqual(0);
    expect(sampledToken).toBeLessThan(4);
  });
});

describe('MultiHeadAttention', () => {
  test('should initialize with correct parameters', () => {
    const attention = new MultiHeadAttention(64, 8);
    const params = attention.getParameters();
    
    expect(params.length).toBe(4); // qWeight, kWeight, vWeight, outWeight
    expect(params[0].shape).toEqual([64, 64]); // qWeight
    expect(params[1].shape).toEqual([64, 64]); // kWeight
    expect(params[2].shape).toEqual([64, 64]); // vWeight
    expect(params[3].shape).toEqual([64, 64]); // outWeight
  });

  test('should process input through attention', () => {
    const attention = new MultiHeadAttention(32, 4);
    const input = Tensor.random([8, 32]); // 8 tokens, 32 dimensions
    
    const output = attention.forward(input);
    
    expect(output.shape).toEqual([8, 32]); // Same shape as input
    expect(output).not.toBe(input); // Should be different object
  });
});

describe('MiniGPT', () => {
  test('should initialize with correct configuration', () => {
    const config: GPTConfig = {
      vocabSize: 100,
      blockSize: 16,
      nLayers: 2,
      nHeads: 4,
      nEmbed: 64,
    };
    
    const model = new MiniGPT(config);
    
    expect(model.getConfig()).toEqual(config);
    expect(model.getTokenizer()).toBeDefined();
  });

  test('should perform forward pass', () => {
    const config: GPTConfig = {
      vocabSize: 100,
      blockSize: 8,
      nLayers: 1,
      nHeads: 2,
      nEmbed: 32,
    };
    
    const model = new MiniGPT(config);
    const tokenIds = [1, 2, 3, 4, 5]; // Sample token sequence
    
    const logits = model.forward(tokenIds);
    
    expect(logits.shape).toEqual([5, 100]); // [sequence_length, vocab_size]
  });

  test('should generate text from prompt', () => {
    const config: GPTConfig = {
      vocabSize: 100,
      blockSize: 16,
      nLayers: 1,
      nHeads: 2,
      nEmbed: 32,
    };
    
    const model = new MiniGPT(config);
    
    // Train with minimal corpus
    const corpus = ['function test() {}', 'const x = 1;'];
    model.train(corpus, 1);
    
    const generated = model.generate('function', 5, 0.8);
    
    expect(typeof generated).toBe('string');
    expect(generated.length).toBeGreaterThan(0);
  });

  test('should train on corpus', async () => {
    const config: GPTConfig = {
      vocabSize: 100,
      blockSize: 12, // Larger block size to generate training sequences
      nLayers: 1,
      nHeads: 2,
      nEmbed: 32,
    };
    
    const model = new MiniGPT(config);
    const corpus = [
      'function add(a, b) { return a + b; }',
      'function multiply(x, y) { return x * y; }',
      'const result = add(1, 2);',
    ];
    
    // Just test that training completes without throwing
    try {
      await model.train(corpus, 1);
      expect(true).toBe(true); // Training succeeded
    } catch (error) {
      expect(error).toBeUndefined(); // Should not throw
    }
  });
});

describe('Integration Tests', () => {
  test('should complete end-to-end workflow', async () => {
    const config: GPTConfig = {
      vocabSize: 200,
      blockSize: 12,
      nLayers: 2,
      nHeads: 4,
      nEmbed: 64,
    };
    
    const model = new MiniGPT(config);
    
    // Train on JavaScript functions
    const corpus = [
      'function add(a, b) { return a + b; }',
      'function subtract(a, b) { return a - b; }',
      'function multiply(x, y) { return x * y; }',
      'const result = add(1, 2);',
      'const product = multiply(3, 4);',
    ];
    
    await model.train(corpus, 1);
    
    // Generate completion
    const generated = model.generate('function', 8, 0.7);
    
    expect(generated).toBeDefined();
    expect(typeof generated).toBe('string');
    expect(generated.length).toBeGreaterThan('function'.length);
  });
});