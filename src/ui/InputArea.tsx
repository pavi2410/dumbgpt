import { memo } from 'react';
import { Box, Text, useFocus } from 'ink';
import { colors } from './theme.js';
import { useInputHistory } from './hooks/useInputHistory.js';

interface InputAreaProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: (value: string) => void;
  disabled?: boolean;
  modelType: 'text' | 'code';
  contextSize: number;
  maxTokens: number;
  isLoading: boolean;
  quickCompletions?: string[];
}

export const InputArea = memo(function InputArea({ value, onChange, onSubmit, disabled, modelType, contextSize, maxTokens, isLoading, quickCompletions }: InputAreaProps) {
  const { inputValue, lines, stats } = useInputHistory({
    onSubmit,
    disabled,
    maxLines: 5,
    maxHistory: 50
  });

  useFocus({ autoFocus: true });

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
          enter send • \+enter new line • ↑↓ history ({stats.currentLines}/{5})
        </Text>
        <Text color={colors.textDim}>
          {modelType === 'code' ? 'JS/TS' : 'Text'} GPT • {contextSize}:{maxTokens} • History: {stats.historyCount}
        </Text>
      </Box>
      
      {/* Quick Completions */}
      {quickCompletions && quickCompletions.length > 0 && (
        <Box flexDirection="column" marginTop={1}>
          <Text color={colors.textDim}>Quick suggestions:</Text>
          <Box flexDirection="row" flexWrap="wrap" gap={1}>
            {quickCompletions.slice(0, 6).map((completion, index) => (
              <Box key={index} borderStyle="round" borderColor={colors.textDim} paddingX={1}>
                <Text color={colors.secondary}>{completion}</Text>
              </Box>
            ))}
          </Box>
        </Box>
      )}
    </Box>
  );
});