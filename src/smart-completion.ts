import { tokenizeJavaScript } from './tokenizers/javascript.js';

export interface SmartCompletionOptions {
    markovChain: Map<string, Map<string, number>>;
    maxTokens: number;
    contextSize: number;
}

export class SmartCompletion {
    private markovChain: Map<string, Map<string, number>>;
    private maxTokens: number;
    private contextSize: number;

    constructor(options: SmartCompletionOptions) {
        this.markovChain = options.markovChain;
        this.maxTokens = options.maxTokens;
        this.contextSize = options.contextSize;
    }

    // Detect what kind of completion the user wants
    detectCompletionType(input: string): 'function' | 'class' | 'import' | 'variable' | 'comment' | 'generic' {
        const trimmed = input.trim();
        
        if (trimmed.startsWith('function ') || trimmed.includes('function ')) return 'function';
        if (trimmed.startsWith('class ') || trimmed.includes('class ')) return 'class';
        if (trimmed.startsWith('import ') || trimmed.includes('import ')) return 'import';
        if (trimmed.startsWith('//') || trimmed.startsWith('/*')) return 'comment';
        if (trimmed.startsWith('const ') || trimmed.startsWith('let ') || trimmed.startsWith('var ')) return 'variable';
        
        return 'generic';
    }

    // Enhanced context-aware generation
    complete(input: string): string[] {
        const completionType = this.detectCompletionType(input);
        
        switch (completionType) {
            case 'function':
                return this.completeFunctionDefinition(input);
            case 'class':
                return this.completeClassDefinition(input);
            case 'import':
                return this.completeImportStatement(input);
            case 'variable':
                return this.completeVariableDeclaration(input);
            case 'comment':
                return this.completeComment(input);
            default:
                return this.smartContextualCompletion(input);
        }
    }

    private completeFunctionDefinition(input: string): string[] {
        // Look for function patterns in the training data
        const functionStarters = [
            'function ',
            'const ',
            'async function ',
            'export function ',
            'export const '
        ];
        
        return this.findBestMatch(input, functionStarters);
    }

    private completeClassDefinition(input: string): string[] {
        const classStarters = [
            'class ',
            'export class ',
            'export default class '
        ];
        
        return this.findBestMatch(input, classStarters);
    }

    private completeImportStatement(input: string): string[] {
        const importStarters = [
            'import ',
            'import { ',
            'import * as ',
            'import type '
        ];
        
        return this.findBestMatch(input, importStarters);
    }

    private completeVariableDeclaration(input: string): string[] {
        const varStarters = [
            'const ',
            'let ',
            'var '
        ];
        
        return this.findBestMatch(input, varStarters);
    }

    private completeComment(input: string): string[] {
        // For comments, just use basic completion
        return this.basicCompletion(input);
    }

    private smartContextualCompletion(input: string): string[] {
        // Try multiple strategies for better completion
        const strategies = [
            () => this.completeByLastWord(input),
            () => this.completeByPattern(input),
            () => this.basicCompletion(input)
        ];

        for (const strategy of strategies) {
            const result = strategy();
            if (result.length > 0) {
                return result;
            }
        }

        return [];
    }

    private completeByLastWord(input: string): string[] {
        const tokens = tokenizeJavaScript(input);
        if (tokens.length === 0) return [];

        const lastToken = tokens[tokens.length - 1];
        
        // Find contexts that start with the last token
        for (const [context, predictions] of this.markovChain) {
            if (context.startsWith(lastToken.toLowerCase())) {
                return this.generateFromContext(context, predictions);
            }
        }

        return [];
    }

    private completeByPattern(input: string): string[] {
        // Look for common JavaScript patterns
        const patterns = [
            /\.$/,           // Property access
            /\($/,           // Function call
            /\{$/,           // Object start
            /\[$/,           // Array access
            /=$/,            // Assignment
            /=>$/,           // Arrow function
            /if\s*\($/,      // If statement
            /for\s*\($/,     // For loop
            /while\s*\($/,   // While loop
        ];

        for (const pattern of patterns) {
            if (pattern.test(input)) {
                return this.findPatternCompletion(input, pattern);
            }
        }

        return [];
    }

    private findPatternCompletion(input: string, pattern: RegExp): string[] {
        // Find training examples that match this pattern
        const results: string[] = [];
        
        for (const [context, predictions] of this.markovChain) {
            if (pattern.test(context)) {
                results.push(...this.generateFromContext(context, predictions));
                if (results.length >= this.maxTokens) break;
            }
        }

        return results.slice(0, this.maxTokens);
    }

    private findBestMatch(input: string, starters: string[]): string[] {
        for (const starter of starters) {
            if (input.toLowerCase().includes(starter.toLowerCase())) {
                return this.findMatchingContexts(starter);
            }
        }
        return this.basicCompletion(input);
    }

    private findMatchingContexts(starter: string): string[] {
        const results: string[] = [];
        
        for (const [context, predictions] of this.markovChain) {
            if (context.toLowerCase().includes(starter.toLowerCase())) {
                results.push(...this.generateFromContext(context, predictions));
                if (results.length >= this.maxTokens) break;
            }
        }

        return results.slice(0, this.maxTokens);
    }

    private generateFromContext(context: string, predictions: Map<string, number>): string[] {
        const results: string[] = [];
        const sortedPredictions = Array.from(predictions.entries())
            .sort((a, b) => b[1] - a[1])
            .slice(0, 10);

        for (const [token, count] of sortedPredictions) {
            results.push(token);
            if (results.length >= 5) break;
        }

        return results;
    }

    private basicCompletion(input: string): string[] {
        const tokens = tokenizeJavaScript(input);
        const generatedTokens = [...tokens];
        
        for (let i = 0; i < this.maxTokens; i++) {
            const nextToken = this.generateNextToken(generatedTokens);
            if (!nextToken) break;
            generatedTokens.push(nextToken);
        }
        
        return generatedTokens.slice(tokens.length);
    }

    private generateNextToken(tokens: string[]): string | null {
        for (let contextLength = this.contextSize; contextLength >= 1; contextLength--) {
            const context = tokens.slice(-contextLength).join(' ').toLowerCase();
            const predictions = this.markovChain.get(context);
            
            if (predictions && predictions.size > 0) {
                return this.selectBestToken(predictions);
            }
        }
        return null;
    }

    private selectBestToken(predictions: Map<string, number>): string {
        const entries = Array.from(predictions.entries());
        const total = entries.reduce((sum, [, count]) => sum + count, 0);
        
        // Add some randomness but favor more common tokens
        const random = Math.random() * total;
        let accumulated = 0;
        
        for (const [token, count] of entries) {
            accumulated += count;
            if (accumulated >= random) {
                return token;
            }
        }
        
        return entries[0][0]; // Fallback
    }

    // Quick completions for common patterns
    getQuickCompletions(input: string): string[] {
        const quick = [
            'function ',
            'const ',
            'let ',
            'if (',
            'for (',
            'while (',
            'class ',
            'import ',
            'export ',
            'return ',
            'console.log(',
            'async ',
            'await ',
            'try {',
            'catch (',
            '.map(',
            '.filter(',
            '.reduce(',
            '.forEach(',
            '.find(',
            '=>',
            '&&',
            '||',
        ];

        return quick.filter(completion => 
            completion.toLowerCase().startsWith(input.toLowerCase()) ||
            input.toLowerCase().includes(completion.toLowerCase())
        );
    }
}