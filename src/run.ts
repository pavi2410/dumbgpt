#!/usr/bin/env bun
/**
 * DumbGPT TUI runner
 * 
 * This script runs the interactive TUI, loading a pre-trained model if available.
 * Usage: bun run src/run.ts
 */

import { readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { render } from 'ink';
import React from 'react';
import App from './ui/App.js';

const MODEL_PATH = './trained-model.json';

async function loadSavedModel() {
  if (!existsSync(MODEL_PATH)) {
    console.log('⚠️  No pre-trained model found. The app will train a new model.');
    return null;
  }

  try {
    const modelData = await readFile(MODEL_PATH, 'utf-8');
    const savedModel = JSON.parse(modelData);
    
    console.log('✅ Found pre-trained model:');
    console.log(`   📅 Trained: ${savedModel.timestamp}`);
    console.log(`   📊 Stats: ${savedModel.stats.filesRead} files, ${Math.round(savedModel.stats.totalSize / 1024)}KB`);
    console.log(`   ⚙️  Config: ${savedModel.config.nLayers} layers, ${savedModel.config.nHeads} heads`);
    
    return savedModel;
  } catch (error) {
    console.warn('⚠️  Failed to load saved model:', error);
    return null;
  }
}

async function main() {
  console.log('🧠 DumbGPT - Interactive Code Completion');
  console.log('');
  
  // Check for saved model
  const savedModel = await loadSavedModel();
  
  if (savedModel) {
    console.log('🚀 Starting TUI with pre-trained model...');
  } else {
    console.log('🚀 Starting TUI (will train new model)...');
  }
  
  console.log('');
  
  // Start the TUI
  render(React.createElement(App));
}

main().catch(console.error);