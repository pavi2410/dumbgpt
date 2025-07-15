/**
 * Basic Tensor implementation for Mini-GPT
 * 
 * This is a simplified tensor library for educational purposes.
 * In production, you'd use something like TensorFlow.js or PyTorch.js
 */

export class Tensor {
  data: Float32Array;
  shape: number[];
  size: number;

  constructor(shape: number[], data?: Float32Array | number[]) {
    this.shape = shape;
    this.size = shape.reduce((a, b) => a * b, 1);
    
    if (data) {
      this.data = data instanceof Float32Array ? data : new Float32Array(data);
    } else {
      this.data = new Float32Array(this.size);
    }
    
    if (this.data.length !== this.size) {
      throw new Error(`Data size ${this.data.length} doesn't match shape size ${this.size}`);
    }
  }

  // Create tensor filled with zeros
  static zeros(shape: number[]): Tensor {
    return new Tensor(shape);
  }

  // Create tensor filled with ones
  static ones(shape: number[]): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    return new Tensor(shape, new Float32Array(size).fill(1));
  }

  // Create tensor with random values (Xavier initialization)
  static random(shape: number[]): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size);
    const scale = Math.sqrt(2.0 / (shape[0] || 1)); // Xavier initialization
    
    for (let i = 0; i < size; i++) {
      data[i] = (Math.random() - 0.5) * 2 * scale;
    }
    
    return new Tensor(shape, data);
  }

  // Get element at index (supports multi-dimensional indexing)
  get(indices: number[]): number {
    const index = this.flatIndex(indices);
    return this.data[index];
  }

  // Set element at index
  set(indices: number[], value: number): void {
    const index = this.flatIndex(indices);
    this.data[index] = value;
  }

  // Convert multi-dimensional indices to flat index
  private flatIndex(indices: number[]): number {
    let index = 0;
    let stride = 1;
    
    for (let i = indices.length - 1; i >= 0; i--) {
      index += indices[i] * stride;
      stride *= this.shape[i];
    }
    
    return index;
  }

  // Matrix multiplication (for 2D tensors)
  matmul(other: Tensor): Tensor {
    if (this.shape.length !== 2 || other.shape.length !== 2) {
      throw new Error('Matrix multiplication requires 2D tensors');
    }
    
    const [aRows, aCols] = this.shape;
    const [bRows, bCols] = other.shape;
    
    if (aCols !== bRows) {
      throw new Error(`Cannot multiply matrices: ${aRows}x${aCols} * ${bRows}x${bCols}`);
    }
    
    const result = Tensor.zeros([aRows, bCols]);
    
    for (let i = 0; i < aRows; i++) {
      for (let j = 0; j < bCols; j++) {
        let sum = 0;
        for (let k = 0; k < aCols; k++) {
          sum += this.get([i, k]) * other.get([k, j]);
        }
        result.set([i, j], sum);
      }
    }
    
    return result;
  }

  // Element-wise addition
  add(other: Tensor | number): Tensor {
    if (typeof other === 'number') {
      const result = new Tensor(this.shape, new Float32Array(this.data));
      for (let i = 0; i < result.size; i++) {
        result.data[i] += other;
      }
      return result;
    }
    
    if (!this.shapeEquals(other.shape)) {
      throw new Error(`Shape mismatch: ${this.shape} vs ${other.shape}`);
    }
    
    const result = new Tensor(this.shape);
    for (let i = 0; i < this.size; i++) {
      result.data[i] = this.data[i] + other.data[i];
    }
    
    return result;
  }

  // Element-wise multiplication
  mul(other: Tensor | number): Tensor {
    if (typeof other === 'number') {
      const result = new Tensor(this.shape, new Float32Array(this.data));
      for (let i = 0; i < result.size; i++) {
        result.data[i] *= other;
      }
      return result;
    }
    
    if (!this.shapeEquals(other.shape)) {
      throw new Error(`Shape mismatch: ${this.shape} vs ${other.shape}`);
    }
    
    const result = new Tensor(this.shape);
    for (let i = 0; i < this.size; i++) {
      result.data[i] = this.data[i] * other.data[i];
    }
    
    return result;
  }

  // Apply softmax activation (for attention and output probabilities)
  softmax(dim: number = -1): Tensor {
    if (dim === -1) dim = this.shape.length - 1;
    
    const result = new Tensor(this.shape, new Float32Array(this.data));
    const dimSize = this.shape[dim];
    const outerSize = this.size / dimSize;
    
    for (let outer = 0; outer < outerSize; outer++) {
      // Find max for numerical stability
      let maxVal = -Infinity;
      for (let i = 0; i < dimSize; i++) {
        const idx = outer * dimSize + i;
        maxVal = Math.max(maxVal, result.data[idx]);
      }
      
      // Compute exp and sum
      let sum = 0;
      for (let i = 0; i < dimSize; i++) {
        const idx = outer * dimSize + i;
        result.data[idx] = Math.exp(result.data[idx] - maxVal);
        sum += result.data[idx];
      }
      
      // Normalize
      for (let i = 0; i < dimSize; i++) {
        const idx = outer * dimSize + i;
        result.data[idx] /= sum;
      }
    }
    
    return result;
  }

  // Reshape tensor
  reshape(newShape: number[]): Tensor {
    const newSize = newShape.reduce((a, b) => a * b, 1);
    if (newSize !== this.size) {
      throw new Error(`Cannot reshape tensor of size ${this.size} to ${newSize}`);
    }
    
    return new Tensor(newShape, new Float32Array(this.data));
  }

  // Transpose (for 2D tensors)
  transpose(): Tensor {
    if (this.shape.length !== 2) {
      throw new Error('Transpose only supported for 2D tensors');
    }
    
    const [rows, cols] = this.shape;
    const result = Tensor.zeros([cols, rows]);
    
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        result.set([j, i], this.get([i, j]));
      }
    }
    
    return result;
  }

  // Check if shapes are equal
  private shapeEquals(other: number[]): boolean {
    if (this.shape.length !== other.length) return false;
    return this.shape.every((dim, i) => dim === other[i]);
  }

  // Convert to string for debugging
  toString(): string {
    return `Tensor(shape=[${this.shape.join(',')}], data=[${this.data.slice(0, 10).join(', ')}${this.size > 10 ? '...' : ''}])`;
  }

  // Create a copy
  clone(): Tensor {
    return new Tensor(this.shape, new Float32Array(this.data));
  }
}

// Utility functions for common operations

// ReLU activation function
export function relu(x: Tensor): Tensor {
  const result = new Tensor(x.shape);
  for (let i = 0; i < x.size; i++) {
    result.data[i] = Math.max(0, x.data[i]);
  }
  return result;
}

// GELU activation function (used in GPT)
export function gelu(x: Tensor): Tensor {
  const result = new Tensor(x.shape);
  for (let i = 0; i < x.size; i++) {
    const val = x.data[i];
    result.data[i] = 0.5 * val * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (val + 0.044715 * val * val * val)));
  }
  return result;
}

// Layer normalization
export function layerNorm(x: Tensor, gamma: Tensor, beta: Tensor, eps: number = 1e-5): Tensor {
  const result = new Tensor(x.shape);
  const lastDim = x.shape[x.shape.length - 1];
  const batchSize = x.size / lastDim;
  
  for (let batch = 0; batch < batchSize; batch++) {
    const start = batch * lastDim;
    
    // Calculate mean
    let mean = 0;
    for (let i = 0; i < lastDim; i++) {
      mean += x.data[start + i];
    }
    mean /= lastDim;
    
    // Calculate variance
    let variance = 0;
    for (let i = 0; i < lastDim; i++) {
      const diff = x.data[start + i] - mean;
      variance += diff * diff;
    }
    variance /= lastDim;
    
    // Normalize
    const std = Math.sqrt(variance + eps);
    for (let i = 0; i < lastDim; i++) {
      const normalized = (x.data[start + i] - mean) / std;
      result.data[start + i] = normalized * gamma.data[i] + beta.data[i];
    }
  }
  
  return result;
}