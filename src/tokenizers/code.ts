import { styleText } from 'node:util';

const colors = ['red', 'blue', 'green', 'yellow', 'magenta', 'cyan'] as const;

export function tokenizeJavaCode(code: string) {
    // Regular expression to match Java tokens: words, numbers, punctuation, and Java-specific symbols
    const regex = /\b\w+\b|\/\/.*|\/\*[\s\S]*?\*\/|".*?"|'.*?'|[{}()\[\];,]|[+-/*=<>!&|]+/g;
    const tokens = [];
    let match;

    while ((match = regex.exec(code)) !== null) {
        tokens.push(match[0]);
    }

    return tokens;
}

export function joiner(array: string[]) {
    let out = '';
    for (let i = 0; i < array.length; i++) {
        out += styleText(colors[i % colors.length], array[i]);
        if (i !== array.length - 1 && /^\w|^[\"\[\()].+/.exec(array[i + 1])) {
            out += ' ';
        }
    }
    return out;
}