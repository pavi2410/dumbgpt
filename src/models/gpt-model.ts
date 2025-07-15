/**
 * GPT Model integration for DumbGPT
 * 
 * This creates a GPT-based model that can be used as a drop-in replacement
 * for the existing Markov chain model in DumbGPT.
 */

import { readFile } from 'node:fs/promises';
import { MiniGPT, type GPTConfig } from './gpt/transformer.js';

interface ProgressCallback {
  onScanProgress: (filesScanned: number, currentFile?: string) => void;
  onReadProgress: (filesRead: number, totalFiles: number, currentFile: string, fileSize: number, totalSize: number, linesRead: number) => void;
  onTrainingStart: () => void;
  onTrainingComplete: () => void;
}

export async function createGPTModel(
  contextSize: number,
  maxOutputTokens: number,
  progressCallback?: ProgressCallback
) {
  // GPT Configuration - optimized for very fast CPU training
  const config: GPTConfig = {
    vocabSize: 2000,        // Much smaller vocabulary
    blockSize: Math.min(contextSize, 16), // Very small context window
    nLayers: 2,             // Minimal transformer layers
    nHeads: 4,              // Fewer attention heads
    nEmbed: 128,            // Smaller embedding dimension
  };

  console.log('🧠 Initializing Mini-GPT...');
  console.log(`Configuration: ${config.nLayers} layers, ${config.nHeads} heads, ${config.nEmbed} dims`);

  // Initialize the GPT model
  const model = new MiniGPT(config);

  // Scan for training files (reuse existing logic)
  const { corpus, stats } = await scanTrainingFiles(progressCallback);
  
  // Train the model
  progressCallback?.onTrainingStart();
  console.log('🎓 Training Mini-GPT...');
  await model.train(corpus, 1); // 1 epoch for now (training is slow without GPU)
  progressCallback?.onTrainingComplete();

  console.log('✅ Mini-GPT training complete!');
  console.log(`📊 Final stats: ${stats.filesRead} files, ${stats.totalSize} bytes, ${stats.linesRead} lines`);

  // Create interface compatible with existing DumbGPT
  const generateText = (inputText: string): string[] => {
    try {
      // Generate text using the GPT model
      const generated = model.generate(inputText, maxOutputTokens, 0.8);
      
      // Simply split generated text into tokens for display
      // This is a simplified approach - in production you'd want proper tokenization
      return generated.split(/\s+/).filter(token => token.length > 0);
    } catch (error) {
      console.error('GPT generation error:', error);
      return ['Error generating text'];
    }
  };

  const getQuickCompletions = (inputText: string): string[] => {
    // Use the tokenizer to suggest common code patterns
    const tokenizer = model.getTokenizer();
    
    // Simple pattern matching for quick completions
    const patterns = [
      'function',
      'const',
      'let',
      'class',
      'import',
      'export',
      'if',
      'for',
      'while',
      'return',
      'async',
      'await',
      '=>',
      '{}',
      '[]',
      '()',
    ];
    
    return patterns.filter(pattern => 
      pattern.toLowerCase().includes(inputText.toLowerCase()) ||
      inputText.toLowerCase().includes(pattern.toLowerCase())
    );
  };

  // Enhanced completion that uses GPT for better suggestions
  const smartComplete = (inputText: string): string[] => {
    try {
      // Generate multiple completions with different temperatures
      const completions = [];
      
      for (let temp = 0.3; temp <= 0.9; temp += 0.3) {
        const completion = model.generate(inputText, 20, temp);
        completions.push(completion);
      }
      
      // Return the completions as tokens
      return completions.flatMap(completion => {
        const tokenizer = model.getTokenizer();
        const tokens = tokenizer.encode(completion);
        return tokens.slice(0, 10).map(id => tokenizer.getToken(id));
      });
    } catch (error) {
      console.error('Smart completion error:', error);
      return getQuickCompletions(inputText);
    }
  };

  return {
    model,
    generateText,
    getQuickCompletions,
    smartComplete,
    joiner: (tokens: string[]) => tokens.join(' '), // Simple joiner for now
    config,
    stats
  };
}

/**
 * Scan training files from node_modules (reuse existing logic)
 */
async function scanTrainingFiles(progressCallback?: ProgressCallback) {
  const jsGlob = new Bun.Glob('./node_modules/**/*.js');
  const tsGlob = new Bun.Glob('./node_modules/**/*.ts');

  const foundFiles = [];
  let filesScanned = 0;
  const maxFiles = 20; // Very small dataset for fast training

  // Scan JavaScript files
  for await (const entry of jsGlob.scan()) {
    if (foundFiles.length >= maxFiles) break;
    
    // Skip minified files and very large files
    if (entry.includes('.min.') || entry.includes('dist/') || entry.includes('build/')) {
      continue;
    }
    
    try {
      const stats = Bun.file(entry).size;
      if (stats > 50000) continue; // Skip large files
      
      foundFiles.push(entry);
      filesScanned++;
      progressCallback?.onScanProgress(filesScanned, entry);
    } catch (error) {
      continue;
    }
  }

  // Scan TypeScript files
  for await (const entry of tsGlob.scan()) {
    if (foundFiles.length >= maxFiles) break;
    
    // Skip definition files and large files
    if (entry.includes('.d.ts') || entry.includes('dist/') || entry.includes('build/')) {
      continue;
    }
    
    try {
      const stats = Bun.file(entry).size;
      if (stats > 50000) continue; // Skip large files
      
      foundFiles.push(entry);
      filesScanned++;
      progressCallback?.onScanProgress(filesScanned, entry);
    } catch (error) {
      continue;
    }
  }

  // Reading phase
  const corpus = [];
  let filesRead = 0;
  let totalBytesRead = 0;
  let totalLinesRead = 0;
  
  for (const filePath of foundFiles) {
    try {
      const content = await readFile(filePath);
      const text = content.toString();
      
      // Skip files with sourcemaps or that are too large
      if (text.includes('//# sourceMappingURL') || text.length > 50000) {
        continue;
      }
      
      const fileSize = text.length;
      const linesInFile = text.split('\n').length;
      
      totalBytesRead += fileSize;
      totalLinesRead += linesInFile;
      
      progressCallback?.onReadProgress(filesRead, foundFiles.length, filePath, fileSize, totalBytesRead, totalLinesRead);
      
      corpus.push(text);
      filesRead++;
    } catch (error) {
      // Skip files that can't be read
      continue;
    }
  }

  return {
    corpus,
    stats: {
      filesRead,
      totalSize: totalBytesRead,
      linesRead: totalLinesRead
    }
  };
}

