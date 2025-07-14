import { useState, memo } from 'react';
import { Box, Text, useInput, useApp, useFocus } from 'ink';
import { colors } from './theme.js';

interface InputAreaProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: (value: string) => void;
  disabled?: boolean;
  modelType: 'text' | 'code';
  contextSize: number;
  maxTokens: number;
  isLoading: boolean;
}

export const InputArea = memo(function InputArea({ value, onChange, onSubmit, disabled, modelType, contextSize, maxTokens, isLoading }: InputAreaProps) {
  const [inputValue, setInputValue] = useState('');
  const { exit } = useApp();
  useFocus({ autoFocus: true });

  const MAX_LINES = 5;
  const lines = inputValue.split('\n');

  useInput((input, key) => {
    if (key.ctrl && input === 'c') {
      exit();
      return;
    }

    if (disabled) return;

    // Check if input ends with backslash and user pressed Enter
    if (key.return && inputValue.endsWith('\\')) {
      // Remove the backslash and add a new line
      if (lines.length < MAX_LINES) {
        setInputValue(prev => prev.slice(0, -1) + '\n');
      }
      return;
    }

    // Enter submits the input
    if (key.return) {
      onSubmit(inputValue);
      setInputValue('');
      return;
    }

    // Backspace handling
    if (key.backspace || key.delete) {
      setInputValue(prev => {
        if (prev.length === 0) return prev;
        return prev.slice(0, -1);
      });
      return;
    }

    if (key.ctrl) return;

    if (input) {
      setInputValue(prev => prev + input);
    }
  });

  const terminalWidth = process.stdout.columns || 80;

  return (
    <Box flexDirection="column">
      <Box borderStyle="round" borderColor={colors.inputBorder} paddingX={1} marginBottom={1}>
        {lines.map((line, index) => {
          const isFirstLine = index === 0;
          const isLastLine = index === lines.length - 1;
          
          return (
            <Box key={index}>
              {isFirstLine && <Text color={colors.prompt} bold>❯ </Text>}
              {!isFirstLine && <Text color={colors.textDim}>  </Text>}
              <Text color={colors.text}>
                {line}
                {isLastLine && !disabled && <Text color={colors.cursor}>▌</Text>}
                {isLastLine && disabled && (
                  <Text color={colors.warning}>
                    {' (generating...)'}
                  </Text>
                )}
              </Text>
            </Box>
          );
        })}
      </Box>
      
      <Box justifyContent="space-between">
        <Text color={colors.textDim}>
          enter send • \+enter new line ({lines.length}/{MAX_LINES})
        </Text>
        <Text color={colors.textDim}>
          {modelType === 'code' ? 'Java' : 'Text'} Model • {contextSize}:{maxTokens}
        </Text>
      </Box>
    </Box>
  );
});