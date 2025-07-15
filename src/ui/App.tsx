import React, { useState, useEffect } from 'react';
import { Box, Text, useApp } from 'ink';
import BigText from 'ink-big-text';
import { ChatArea } from './ChatArea.js';
import { InputArea } from './InputArea.js';
import { createGPTModel } from '../models/index.js';
import { colors } from './theme.js';
import { readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';

interface Message {
  type: 'user' | 'assistant';
  content: string | string[]; // Can be string for commands or string[] for generated tokens
  timestamp: Date;
}

export default function App() {
  const { exit } = useApp();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [quickCompletions, setQuickCompletions] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [model, setModel] = useState<any>(null);
  const [modelError, setModelError] = useState<string | null>(null);
  const [config, setConfig] = useState({
    contextSize: 32,     // Larger context for GPT
    maxOutputTokens: 50, // Reasonable output length
    modelType: 'code' as 'text' | 'code' // Keep for UI compatibility
  });

  useEffect(() => {
    async function initializeModel() {
      setIsLoading(true);
      setModelError(null);
      
      try {
        // Check if we have a pre-trained model
        if (existsSync('./trained-model.json')) {
          // Load pre-trained model
          const modelData = await readFile('./trained-model.json', 'utf-8');
          const savedModel = JSON.parse(modelData);
          
          // Load the actual trained model weights
          const { MiniGPT } = await import('../models/gpt/transformer.js');
          const trainedModel = MiniGPT.fromJSON(savedModel);
          
          // Create wrapper with the same interface as createGPTModel
          const newModel = {
            model: trainedModel,
            generateText: (inputText: string) => {
              try {
                const generated = trainedModel.generate(inputText, config.maxOutputTokens, 0.8);
                return generated.split(/\s+/).filter(token => token.length > 0);
              } catch (error) {
                console.error('GPT generation error:', error);
                return ['Error generating text'];
              }
            },
            getQuickCompletions: (inputText: string) => {
              const patterns = ['function', 'const', 'let', 'class', 'import', 'export', 'if', 'for', 'while', 'return', 'async', 'await', '=>', '{}', '[]', '()'];
              return patterns.filter(pattern => 
                pattern.toLowerCase().includes(inputText.toLowerCase()) ||
                inputText.toLowerCase().includes(pattern.toLowerCase())
              );
            },
            config: savedModel.config,
            stats: savedModel.stats
          };
          setModel(newModel);
          
          // Initialize with helpful suggestions
          if (newModel.getQuickCompletions) {
            setQuickCompletions(newModel.getQuickCompletions(''));
          }
          
          // Show model info
          setMessages([{
            type: 'assistant',
            content: `✅ Loaded pre-trained model (${savedModel.timestamp})
📊 Training stats: ${savedModel.stats.filesRead} files, ${Math.round(savedModel.stats.totalSize / 1024)}KB
⚙️ Config: ${savedModel.config.nLayers} layers, ${savedModel.config.nHeads} heads

Type your code and I'll help complete it!`,
            timestamp: new Date()
          }]);
        } else {
          // No pre-trained model found
          setModelError('No pre-trained model found. Please run "bun run train" first.');
        }
      } catch (error) {
        setModelError(`Error loading model: ${error}`);
      } finally {
        setIsLoading(false);
      }
    }

    initializeModel();
  }, [config.maxOutputTokens]);

  const handleSubmit = async (text: string) => {
    if (!text.trim() || !model) return;

    // Add user message
    const userMessage: Message = {
      type: 'user',
      content: text,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, userMessage]);

    // Handle commands
    if (text.startsWith(':')) {
      handleCommand(text);
      return;
    }

    // Generate response
    setIsLoading(true);
    try {
      const generatedTokens = model.generateText(text);
      console.log('Generated tokens:', generatedTokens.slice(0, 10)); // Debug: see what tokens look like
      
      const assistantMessage: Message = {
        type: 'assistant',
        content: generatedTokens, // Store the actual tokens array
        timestamp: new Date()
      };
      setMessages(prev => [...prev, assistantMessage]);
      
      // Update quick completions based on the current input
      if (model.getQuickCompletions) {
        setQuickCompletions(model.getQuickCompletions(text));
      }
    } catch (error) {
      const errorMessage: Message = {
        type: 'assistant',
        content: `Error: ${error}`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Command definitions
  const commands = [
    { 
      name: 'help', 
      description: 'show help', 
      shortcut: 'ctrl+x h',
      handler: () => {
        setMessages(prev => [...prev, {
          type: 'assistant',
          content: `🤖 DumbGPT Smart Code Completion

GPT FEATURES:
• Transformer-based neural language model
• Trained on JavaScript/TypeScript code from node_modules
• Context-aware code completion and generation
• Multi-head attention mechanism for better understanding
• Quick suggestions based on code patterns

COMMANDS:
${commands.map(cmd => `:${cmd.name} - ${cmd.description}`).join('\n')}

EXAMPLES:
• "function add" → GPT generates function completion
• "const data = " → Variable assignment completion
• "if (" → Conditional statement completion
• "import " → Module import completion

Config keys: contextSize, maxOutputTokens, modelType`,
          timestamp: new Date()
        }]);
      }
    },
    {
      name: 'config',
      description: 'show configuration',
      shortcut: 'ctrl+x c',
      handler: () => {
        setMessages(prev => [...prev, {
          type: 'assistant',
          content: `Configuration:
Context Size: ${config.contextSize}
Max Tokens: ${config.maxOutputTokens}
Model Type: ${config.modelType}
Engine: Mini-GPT (Transformer)`,
          timestamp: new Date()
        }]);
      }
    },
    {
      name: 'set',
      description: 'change settings',
      shortcut: 'ctrl+x s',
      handler: (args: string[]) => {
        if (args.length === 2) {
          const [key, value] = args;
          handleConfigChange(key, value);
        } else {
          setMessages(prev => [...prev, {
            type: 'assistant',
            content: 'Usage: :set <key> <value>',
            timestamp: new Date()
          }]);
        }
      }
    },
    {
      name: 'clear',
      description: 'clear messages',
      shortcut: 'ctrl+x l',
      handler: () => {
        setMessages([]);
      }
    },
    {
      name: 'quit',
      description: 'exit application',
      shortcut: 'ctrl+x q',
      handler: () => {
        exit();
      }
    }
  ];

  // Create command handlers map from commands array
  const commandHandlers = commands.reduce((acc, cmd) => {
    acc[cmd.name] = cmd.handler;
    return acc;
  }, {} as Record<string, (args?: string[]) => void>);

  const handleCommand = (command: string) => {
    const [cmd, ...args] = command.slice(1).split(' ');
    
    const handler = commandHandlers[cmd as keyof typeof commandHandlers];
    if (handler) {
      handler(args);
    } else {
      setMessages(prev => [...prev, {
        type: 'assistant',
        content: `Unknown command: ${cmd}. Type :help for available commands.`,
        timestamp: new Date()
      }]);
    }
  };

  const handleConfigChange = (key: string, value: string) => {
    switch (key) {
      case 'contextSize':
        const contextSize = parseInt(value);
        if (!isNaN(contextSize) && contextSize >= 1 && contextSize <= 10) {
          setConfig(prev => ({ ...prev, contextSize }));
          setMessages(prev => [...prev, {
            type: 'assistant',
            content: `Context size set to ${contextSize}`,
            timestamp: new Date()
          }]);
        } else {
          setMessages(prev => [...prev, {
            type: 'assistant',
            content: 'Context size must be between 1 and 10',
            timestamp: new Date()
          }]);
        }
        break;
        
      case 'maxOutputTokens':
        const maxTokens = parseInt(value);
        if (!isNaN(maxTokens) && maxTokens >= 1 && maxTokens <= 1000) {
          setConfig(prev => ({ ...prev, maxOutputTokens: maxTokens }));
          setMessages(prev => [...prev, {
            type: 'assistant',
            content: `Max output tokens set to ${maxTokens}`,
            timestamp: new Date()
          }]);
        } else {
          setMessages(prev => [...prev, {
            type: 'assistant',
            content: 'Max tokens must be between 1 and 1000',
            timestamp: new Date()
          }]);
        }
        break;
        
      case 'modelType':
        if (value === 'text' || value === 'code') {
          setConfig(prev => ({ ...prev, modelType: value }));
          setMessages(prev => [...prev, {
            type: 'assistant',
            content: `Model type set to ${value}`,
            timestamp: new Date()
          }]);
        } else {
          setMessages(prev => [...prev, {
            type: 'assistant',
            content: 'Model type must be "text" or "code"',
            timestamp: new Date()
          }]);
        }
        break;
        
        
      default:
        setMessages(prev => [...prev, {
          type: 'assistant',
          content: `Unknown config key: ${key}`,
          timestamp: new Date()
        }]);
    }
  };


  if (modelError) {
    return (
      <Box flexDirection="column" height="100%" width="100%" padding={2}>
        <Box justifyContent="center" marginBottom={3}>
          <BigText text="DUMBGPT" color={colors.brand} />
        </Box>
        <Box justifyContent="center" alignItems="center" flexGrow={1}>
          <Box flexDirection="column" alignItems="center">
            <Text color={colors.error}>❌ {modelError}</Text>
            <Text color={colors.textDim} marginTop={1}>Run "bun run train" to create a model first</Text>
          </Box>
        </Box>
      </Box>
    );
  }

  return (
    <Box flexDirection="column" height="100%" width="100%" padding={2}>
      {/* Header */}
      <Box justifyContent="center" marginBottom={3}>
        <BigText text="DUMBGPT" color={colors.brand} />
      </Box>

      {/* Commands Menu */}
      <Box flexDirection="column" alignItems="center" marginBottom={4}>
        {commands.map((cmd, index) => (
          <Box key={cmd.name} marginBottom={index < commands.length - 1 ? 1 : 0}>
            <Box width={20}>
              <Text color={colors.command}>:{cmd.name}</Text>
            </Box>
            <Box width={30}>
              <Text color={colors.textDim}>{cmd.description}</Text>
            </Box>
            <Box>
              <Text color={colors.textDim}>{cmd.shortcut}</Text>
            </Box>
          </Box>
        ))}
      </Box>
      
      <ChatArea messages={messages} />
      
      <InputArea 
        value={input}
        onChange={setInput}
        onSubmit={handleSubmit}
        disabled={isLoading}
        modelType={config.modelType}
        contextSize={config.contextSize}
        maxTokens={config.maxOutputTokens}
        isLoading={isLoading}
        quickCompletions={quickCompletions}
      />
    </Box>
  );
}