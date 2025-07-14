import { readFile } from 'node:fs/promises';
import { train } from './core/generator.js';
import { tokenizeJavaScript, joiner } from './tokenizers/javascript.js';
import { SmartCompletion } from './smart-completion.js';

interface ProgressCallback {
  onScanProgress: (filesScanned: number, currentFile?: string) => void;
  onReadProgress: (filesRead: number, totalFiles: number, currentFile: string, fileSize: number, totalSize: number, linesRead: number) => void;
  onTrainingStart: () => void;
  onTrainingComplete: () => void;
}

export async function createCodeModel(
  CONTEXT_SIZE: number, 
  MAX_OUTPUT_TOKENS: number, 
  progressCallback?: ProgressCallback
) {
    // Scanning phase
    const jsGlob = new Bun.Glob('./node_modules/**/*.js');
    const tsGlob = new Bun.Glob('./node_modules/**/*.ts');

    const foundFiles = [];
    let filesScanned = 0;
    const maxFiles = 100; // Limit to prevent overwhelming the system

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

    // Training phase
    progressCallback?.onTrainingStart();
    const markovChain = train(corpus, tokenizeJavaScript, CONTEXT_SIZE);
    progressCallback?.onTrainingComplete();

    // Create smart completion system
    const smartCompletion = new SmartCompletion({
        markovChain,
        maxTokens: MAX_OUTPUT_TOKENS,
        contextSize: CONTEXT_SIZE
    });

    const generateTextWithConfig = (inputText: string) => {
        // Use smart completion for better results
        return smartCompletion.complete(inputText);
    };

    const getQuickCompletions = (inputText: string) => {
        return smartCompletion.getQuickCompletions(inputText);
    };

    return {
        markovChain,
        generateText: generateTextWithConfig,
        getQuickCompletions,
        smartCompletion,
        joiner
    };
}