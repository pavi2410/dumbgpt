import { readFile } from 'node:fs/promises';
import { train, generateText } from './core/generator.js';
import { tokenizeJavaCode, joiner } from './tokenizers/code.js';

export async function createCodeModel(CONTEXT_SIZE: number, MAX_OUTPUT_TOKENS: number) {
    console.time('Reading corpus');
    const glob = new Bun.Glob('./corpus/code/*.java');

    const corpus = [];
    for await (const entry of glob.scan()) {
        console.log(`Reading ${entry}`);
        const content = await readFile(entry);
        corpus.push(content.toString());
    }
    console.timeEnd('Reading corpus');

    console.time('Training');
    const markovChain = train(corpus, tokenizeJavaCode, CONTEXT_SIZE);
    console.timeEnd('Training');

    const generateTextWithConfig = (inputText: string) => 
        generateText(markovChain, inputText, tokenizeJavaCode, CONTEXT_SIZE, MAX_OUTPUT_TOKENS);

    return {
        markovChain,
        generateText: generateTextWithConfig,
        joiner
    };
}