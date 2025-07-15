/**
 * Tokenizer for JavaScript/TypeScript code
 * 
 * This implements a simplified version of BPE (Byte Pair Encoding) tokenization
 * specifically designed for code. In real GPT, this would be much more sophisticated.
 */

export class CodeTokenizer {
  private vocabulary: Map<string, number> = new Map();
  private reverseVocabulary: Map<number, string> = new Map();
  private merges: Array<[string, string]> = [];
  private vocabSize: number = 0;
  
  // Special tokens
  static readonly PAD_TOKEN = '<PAD>';
  static readonly UNK_TOKEN = '<UNK>';
  static readonly BOS_TOKEN = '<BOS>'; // Beginning of sequence
  static readonly EOS_TOKEN = '<EOS>'; // End of sequence
  
  constructor(vocabSize: number = 10000) {
    this.vocabSize = vocabSize;
    this.initializeSpecialTokens();
  }

  private initializeSpecialTokens(): void {
    const specialTokens = [
      CodeTokenizer.PAD_TOKEN,
      CodeTokenizer.UNK_TOKEN, 
      CodeTokenizer.BOS_TOKEN,
      CodeTokenizer.EOS_TOKEN
    ];
    
    for (let i = 0; i < specialTokens.length; i++) {
      this.vocabulary.set(specialTokens[i], i);
      this.reverseVocabulary.set(i, specialTokens[i]);
    }
  }

  /**
   * Pre-tokenize code into initial tokens
   * This splits code into meaningful chunks before BPE
   */
  private preTokenize(code: string): string[] {
    // Enhanced regex for JavaScript/TypeScript
    const codeRegex = /\/\/.*$|\/\*[\s\S]*?\*\/|`(?:[^`\\]|\\.)*`|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|\b\d+\.?\d*\b|\b(?:const|let|var|function|class|interface|type|import|export|from|as|default|async|await|return|if|else|for|while|do|switch|case|break|continue|try|catch|finally|throw|new|this|super|extends|implements|public|private|protected|static|readonly|abstract|namespace|module|declare|enum|void|null|undefined|true|false)\b|\b[a-zA-Z_$][a-zA-Z0-9_$]*\b|=>|[{}()\[\];,.]|[+\-*/=<>!&|?:]+|\r?\n|[ \t]+/gm;
    
    const tokens: string[] = [];
    let match;
    
    while ((match = codeRegex.exec(code)) !== null) {
      const token = match[0];
      
      // Handle different token types
      if (token === '\n' || token === '\r\n') {
        tokens.push('\\n'); // Normalize newlines
      } else if (token.match(/^[ \t]+$/)) {
        // Preserve indentation but normalize
        const spaces = token.length;
        tokens.push(`\\s${spaces}`); // Represent as space token
      } else if (token.trim().length > 0) {
        tokens.push(token);
      }
    }
    
    return tokens;
  }

  /**
   * Train the tokenizer using BPE (Byte Pair Encoding)
   * This finds the most common pairs of tokens and merges them
   */
  train(corpus: string[]): void {
    console.log('Training tokenizer...');
    
    // Step 1: Pre-tokenize all code
    const allTokens: string[] = [];
    for (const code of corpus) {
      allTokens.push(...this.preTokenize(code));
    }
    
    console.log(`Initial tokens: ${allTokens.length}`);
    
    // Step 2: Build character-level vocabulary
    const charSet = new Set<string>();
    for (const token of allTokens) {
      for (const char of token) {
        charSet.add(char);
      }
    }
    
    // Add characters to vocabulary
    let tokenId = this.vocabulary.size;
    for (const char of Array.from(charSet).sort()) {
      this.vocabulary.set(char, tokenId);
      this.reverseVocabulary.set(tokenId, char);
      tokenId++;
    }
    
    // Step 3: Build initial word vocabulary from pre-tokens
    const wordCounts = new Map<string, number>();
    for (const token of allTokens) {
      wordCounts.set(token, (wordCounts.get(token) || 0) + 1);
    }
    
    // Add common words to vocabulary
    const sortedWords = Array.from(wordCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, this.vocabSize - this.vocabulary.size);
    
    for (const [word, count] of sortedWords) {
      if (!this.vocabulary.has(word)) {
        this.vocabulary.set(word, tokenId);
        this.reverseVocabulary.set(tokenId, word);
        tokenId++;
      }
    }
    
    console.log(`Final vocabulary size: ${this.vocabulary.size}`);
    console.log(`Common tokens: ${Array.from(this.vocabulary.keys()).slice(0, 20).join(', ')}`);
  }

  /**
   * Encode text to token IDs
   */
  encode(text: string): number[] {
    const tokens = this.preTokenize(text);
    const tokenIds: number[] = [];
    
    // Add BOS token
    tokenIds.push(this.vocabulary.get(CodeTokenizer.BOS_TOKEN)!);
    
    for (const token of tokens) {
      if (this.vocabulary.has(token)) {
        tokenIds.push(this.vocabulary.get(token)!);
      } else {
        // Fallback to character-level encoding
        for (const char of token) {
          const charId = this.vocabulary.get(char);
          if (charId !== undefined) {
            tokenIds.push(charId);
          } else {
            tokenIds.push(this.vocabulary.get(CodeTokenizer.UNK_TOKEN)!);
          }
        }
      }
    }
    
    // Add EOS token
    tokenIds.push(this.vocabulary.get(CodeTokenizer.EOS_TOKEN)!);
    
    return tokenIds;
  }

  /**
   * Decode token IDs back to text
   */
  decode(tokenIds: number[]): string {
    const tokens: string[] = [];
    
    for (const tokenId of tokenIds) {
      const token = this.reverseVocabulary.get(tokenId);
      if (token) {
        // Handle special formatting tokens
        if (token === '\\n') {
          tokens.push('\n');
        } else if (token.startsWith('\\s')) {
          const spaces = parseInt(token.slice(2));
          tokens.push(' '.repeat(spaces));
        } else if (token === CodeTokenizer.BOS_TOKEN || token === CodeTokenizer.EOS_TOKEN) {
          // Skip special tokens in output
          continue;
        } else {
          tokens.push(token);
        }
      } else {
        tokens.push(CodeTokenizer.UNK_TOKEN);
      }
    }
    
    return tokens.join('');
  }

  /**
   * Get vocabulary size
   */
  getVocabSize(): number {
    return this.vocabulary.size;
  }

  /**
   * Get token ID for a specific token
   */
  getTokenId(token: string): number {
    return this.vocabulary.get(token) ?? this.vocabulary.get(CodeTokenizer.UNK_TOKEN)!;
  }

  /**
   * Get token for a specific ID
   */
  getToken(tokenId: number): string {
    return this.reverseVocabulary.get(tokenId) ?? CodeTokenizer.UNK_TOKEN;
  }

  /**
   * Create context windows for training
   * This splits the tokenized text into overlapping sequences
   */
  createTrainingSequences(tokenIds: number[], sequenceLength: number): Array<{ input: number[], target: number[] }> {
    const sequences: Array<{ input: number[], target: number[] }> = [];
    
    for (let i = 0; i <= tokenIds.length - sequenceLength - 1; i++) {
      const input = tokenIds.slice(i, i + sequenceLength);
      const target = tokenIds.slice(i + 1, i + sequenceLength + 1);
      sequences.push({ input, target });
    }
    
    return sequences;
  }

  /**
   * Sample a token from probabilities (for generation)
   */
  sampleToken(probabilities: number[], temperature: number = 1.0): number {
    if (temperature === 0) {
      // Greedy sampling
      let maxProb = -Infinity;
      let maxIndex = 0;
      for (let i = 0; i < probabilities.length; i++) {
        if (probabilities[i] > maxProb) {
          maxProb = probabilities[i];
          maxIndex = i;
        }
      }
      return maxIndex;
    }
    
    // Temperature sampling
    const scaledProbs = probabilities.map(p => Math.exp(Math.log(p) / temperature));
    const sum = scaledProbs.reduce((a, b) => a + b, 0);
    const normalizedProbs = scaledProbs.map(p => p / sum);
    
    const random = Math.random();
    let cumulative = 0;
    
    for (let i = 0; i < normalizedProbs.length; i++) {
      cumulative += normalizedProbs[i];
      if (random < cumulative) {
        return i;
      }
    }
    
    return probabilities.length - 1; // Fallback
  }

  /**
   * Debug: Print vocabulary stats
   */
  printVocabularyStats(): void {
    console.log(`Vocabulary size: ${this.vocabulary.size}`);
    console.log('Sample tokens:');
    
    const sampleTokens = Array.from(this.vocabulary.entries())
      .slice(0, 50)
      .map(([token, id]) => `${id}: "${token}"`)
      .join('\n');
    
    console.log(sampleTokens);
  }
}