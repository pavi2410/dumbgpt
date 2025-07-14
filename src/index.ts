import { createTextModel } from './text-model.js';
import { createCodeModel } from './code-model.js';

const CONTEXT_SIZE = 4;
const MAX_OUTPUT_TOKENS = 100;

// Choose which model to use
const MODEL_TYPE = 'code'; // 'text' or 'code'

async function main() {
    if (MODEL_TYPE === 'text') {
        const textModel = await createTextModel(CONTEXT_SIZE, MAX_OUTPUT_TOKENS);
        await textModel.runCLI();
    } else {
        const codeModel = await createCodeModel(CONTEXT_SIZE, MAX_OUTPUT_TOKENS);
        await codeModel.runCLI();
    }
}

main().catch(console.error);