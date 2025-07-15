#!/usr/bin/env bun
/**
 * Training script for DumbGPT
 * 
 * This script trains the GPT model and saves it to disk.
 * Usage: bun run src/train.ts
 */

import { writeFile } from 'node:fs/promises';
import { createGPTModel } from './models/gpt-model.js';

interface TrainingConfig {
  contextSize: number;
  maxOutputTokens: number;
  modelPath: string;
}

const config: TrainingConfig = {
  contextSize: 32,
  maxOutputTokens: 50,
  modelPath: './trained-model.json'
};

console.log('🚀 Starting DumbGPT training...');
console.log(`Context size: ${config.contextSize}, Max tokens: ${config.maxOutputTokens}`);

// Progress callback for training
const progressCallback = {
  onScanProgress: (filesScanned: number, currentFile?: string) => {
    process.stdout.write(`\r📁 Scanning files: ${filesScanned} (${currentFile || ''})`);
  },
  
  onReadProgress: (filesRead: number, totalFiles: number, currentFile: string, fileSize: number, totalSize: number, linesRead: number) => {
    const progress = ((filesRead / totalFiles) * 100).toFixed(1);
    process.stdout.write(`\r📖 Reading: ${filesRead}/${totalFiles} (${progress}%) - ${Math.round(totalSize / 1024)}KB, ${linesRead} lines`);
  },
  
  onTrainingStart: () => {
    console.log('\n🎓 Training started...');
  },
  
  onTrainingComplete: () => {
    console.log('✅ Training completed!');
  }
};

try {
  // Create and train the model
  const modelData = await createGPTModel(
    config.contextSize,
    config.maxOutputTokens,
    progressCallback
  );

  console.log('\n💾 Saving model to disk...');
  
  // Save the complete trained model including weights
  const modelToSave = {
    ...modelData.model.toJSON(),
    stats: modelData.stats,
    trained: true,
    timestamp: new Date().toISOString()
  };
  
  await writeFile(config.modelPath, JSON.stringify(modelToSave, null, 2));
  
  console.log(`✅ Model saved to ${config.modelPath}`);
  console.log(`📊 Training stats: ${modelData.stats.filesRead} files, ${Math.round(modelData.stats.totalSize / 1024)}KB`);
  
} catch (error) {
  console.error('❌ Training failed:', error);
  process.exit(1);
}