import React from 'react';
import { Box, Text } from 'ink';
import { colors } from './theme.js';

interface Message {
  type: 'user' | 'assistant';
  content: string;
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
              >
                <Text color={colors.text}>{message.content}</Text>
              </Box>
            </Box>
          </Box>
        ))
      )}
    </Box>
  );
}