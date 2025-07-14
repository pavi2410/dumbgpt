import { styleText } from 'node:util';

const colors = ['red', 'blue', 'green', 'yellow', 'magenta', 'cyan'] as const;

export function tokenize(text: string): string[] {
    return text.split(/[\s\*]+|(?=\W)/); // for novel
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