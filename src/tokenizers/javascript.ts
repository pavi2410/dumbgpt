import { styleText } from 'node:util';

const colors = ['red', 'blue', 'green', 'yellow', 'magenta', 'cyan'] as const;

export function tokenizeJavaScript(code: string) {
    // Enhanced regex for JavaScript/TypeScript tokens - preserves all whitespace
    const regex = /\/\/.*$|\/\*[\s\S]*?\*\/|`(?:[^`\\]|\\.)*`|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|\b\d+\.?\d*\b|\b(?:const|let|var|function|class|interface|type|import|export|from|as|default|async|await|return|if|else|for|while|do|switch|case|break|continue|try|catch|finally|throw|new|this|super|extends|implements|public|private|protected|static|readonly|abstract|namespace|module|declare|enum|void|null|undefined|true|false)\b|\b[a-zA-Z_$][a-zA-Z0-9_$]*\b|=>|[{}()\[\];,.]|[+\-*/=<>!&|?:]+|\r?\n|[ \t]+/gm;
    
    const tokens = [];
    let match;

    while ((match = regex.exec(code)) !== null) {
        const token = match[0];
        
        // Preserve all tokens including whitespace
        if (token === '\n' || token === '\r\n') {
            tokens.push('\n');
        } else if (token.match(/^[ \t]+$/)) {
            // Preserve indentation/spaces as whitespace tokens
            tokens.push(token);
        } else if (token.trim().length > 0) {
            // Regular code tokens
            tokens.push(token);
        }
    }

    return tokens;
}

export function joiner(array: string[]) {
    let out = '';
    for (let i = 0; i < array.length; i++) {
        const token = array[i];
        
        // Handle newlines
        if (token === '\n') {
            out += '\n';
            continue;
        }
        
        // Handle whitespace tokens (spaces/tabs) - preserve as-is
        if (token.match(/^[ \t]+$/)) {
            out += token;
            continue;
        }
        
        // Handle regular code tokens with colors
        out += styleText(colors[i % colors.length], token);
    }
    return out;
}