import readline from 'node:readline/promises';

export async function createCLI() {
    return readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });
}

export async function runCLI(
    markovChain: Map<string, Map<string, number>>,
    generateText: (input: string) => string[],
    joiner: (tokens: string[]) => string,
    CONTEXT_SIZE: number,
    MAX_OUTPUT_TOKENS: number
) {
    const rl = await createCLI();

    console.time('Generation');
    const generatedText = generateText('who are you');
    console.log(joiner(generatedText));
    console.timeEnd('Generation');

    while (true) {
        const command = await rl.question('✨ ');
        if (command === '/q') {
            process.exit(0);
        }
        console.time('Generation');
        const generatedText = generateText(command);
        console.log(joiner(generatedText));
        console.timeEnd('Generation');
    }
}