import { buildMarkovChain, predictNextWord } from './markov.js';

export function train(corpus: string[], tokenizer: (text: string) => string[], k: number): Map<string, Map<string, number>> {
    const tokens = corpus.map(tokenizer).flat();
    return buildMarkovChain(tokens, k);
}

export function generateNextWord(markovChain: Map<string, Map<string, number>>, inputTokens: string[], k: number): string {
    for (let j = k; j >= 0; j--) {
        const context = inputTokens.slice(-j).join(' ').toLowerCase();
        const nextWord = predictNextWord(markovChain, context);
        if (nextWord) {
            return nextWord;
        }
    }
    return '';
}

export function generateText(markovChain: Map<string, Map<string, number>>, inputText: string, tokenizer: (text: string) => string[], k: number, m: number) {
    const generatedText = tokenizer(inputText);
    for (let i = 0; i < m; i++) {
        const nextWord = generateNextWord(markovChain, generatedText, k);
        if (!nextWord) {
            break;
        }
        generatedText.push(nextWord);
    }
    return generatedText;
}