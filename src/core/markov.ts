import * as R from 'remeda';

export function buildMarkovChain(tokens: string[], k: number): Map<string, Map<string, number>> {
    const markovChain = new Map<string, Map<string, number>>();

    for (let i = 0; i < tokens.length; i++) {
        const nextWord = tokens[i];
        for (let j = 0; j <= k; j++) {
            const start = Math.max(0, i - j);
            const context = tokens.slice(start, i).join(' ').toLowerCase();

            if (!markovChain.has(context)) {
                markovChain.set(context, new Map<string, number>());
            }

            const nextWordMap = markovChain.get(context)!;
            nextWordMap.set(nextWord, (nextWordMap.get(nextWord) ?? 0) + 1);
        }
    }

    // Convert frequencies to probabilities
    const probabilities = new Map<string, Map<string, number>>();
    for (const [context, nextWordMap] of markovChain) {
        const totalFrequency = R.sum(Array.from(nextWordMap.values()));
        const probabilityMap = new Map<string, number>();
        for (const [word, frequency] of nextWordMap) {
            probabilityMap.set(word, frequency / totalFrequency);
        }
        probabilities.set(context, probabilityMap);
    }

    return probabilities;
}

export function predictNextWord(markovChain: Map<string, Map<string, number>>, context: string) {
    const nextWordMap = markovChain.get(context);
    if (!nextWordMap || nextWordMap.size === 0) {
        return;
    }

    const maxProb = Math.max(...Array.from(nextWordMap.values()));

    return R.pipe(
        Array.from(nextWordMap.entries()),
        R.filter(([word, prob]) => prob === maxProb),
        R.shuffle(),
        R.first(),
        (w) => w?.[0],
    );
}