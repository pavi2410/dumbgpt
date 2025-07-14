import React from 'react';
import { Box, Text } from 'ink';
import { colors } from './theme.js';

const tokenColors = ['red', 'blue', 'green', 'yellow', 'magenta', 'cyan'] as const;

function TokenizedContent({ tokens }: { tokens: string[] }) {
  const result = [];
  let currentLine = [];
  let tokenIndex = 0;

  for (const token of tokens) {
    if (token === '\n') {
      // End current line and start new one
      if (currentLine.length > 0) {
        result.push(<Text key={`line-${result.length}`}>{currentLine}</Text>);
        currentLine = [];
      } else {
        // Empty line
        result.push(<Text key={`line-${result.length}`}> </Text>);
      }
    } else {
      // Add token to current line with color
      const color = tokenColors[tokenIndex % tokenColors.length];
      currentLine.push(
        <Text key={`token-${tokenIndex}`} color={color}>
          {token}
        </Text>
      );
      
      // Add space if next token starts with word character
      const nextToken = tokens[tokens.indexOf(token) + 1];
      if (nextToken && nextToken !== '\n' && /^\w|^[\"\[\()]/.test(nextToken)) {
        currentLine.push(' ');
      }
      
      tokenIndex++;
    }
  }

  // Add final line if it exists
  if (currentLine.length > 0) {
    result.push(<Text key={`line-${result.length}`}>{currentLine}</Text>);
  }

  return <>{result}</>;
}

interface Message {
  type: 'user' | 'assistant';
  content: string | string[];
  timestamp: Date;
}

interface ChatAreaProps {
  messages: Message[];
}

export function ChatArea({ messages }: ChatAreaProps) {
  return (
    <Box flexDirection="column" flexGrow={1} marginBottom={4}>
      {messages.length > 0 && (
        messages.map((message, index) => (
          <Box key={index} flexDirection="column" marginBottom={2}>
            <Box 
              flexDirection="column" 
              alignItems={message.type === 'user' ? 'flex-end' : 'flex-start'}
              width="100%"
            >
              <Box 
                flexDirection={message.type === 'user' ? 'row-reverse' : 'row'} 
                alignItems="center"
                marginBottom={1}
              >
                <Text color={message.type === 'user' ? colors.prompt : colors.success} bold>
                  {message.type === 'user' ? '❯' : '●'}
                </Text>
                <Text color={colors.textDim} dimColor>
                  {message.type === 'user' ? ' ' : ' '}{message.timestamp.toLocaleTimeString()}
                </Text>
              </Box>
              
              <Box 
                maxWidth="80%" 
                paddingX={2} 
                paddingY={1}
                borderStyle="round"
                borderColor={message.type === 'user' ? colors.prompt : colors.success}
                alignSelf={message.type === 'user' ? 'flex-end' : 'flex-start'}
                flexDirection="column"
              >
                {Array.isArray(message.content) ? (
                  // Render tokens with highlighting and handle newlines
                  <TokenizedContent tokens={message.content} />
                ) : (
                  // Render regular string content with newlines
                  message.content.split('\n').map((line, lineIndex) => (
                    <Text key={lineIndex} color={colors.text}>{line}</Text>
                  ))
                )}
              </Box>
            </Box>
          </Box>
        ))
      )}
    </Box>
  );
}