import React, { useState, useEffect } from 'react';
import { Box, Text, useApp } from 'ink';
import BigText from 'ink-big-text';
import { ChatArea } from './ChatArea.js';
import { InputArea } from './InputArea.js';
import { TrainingProgress } from './TrainingProgress.js';
import { createCodeModel } from '../code-model.js';
import { createTextModel } from '../text-model.js';
import { colors } from './theme.js';

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
  const [config, setConfig] = useState({
    contextSize: 4,
    maxOutputTokens: 100,
    modelType: 'code' as 'text' | 'code'
  });

  // Training progress state
  const [trainingPhase, setTrainingPhase] = useState<'scanning' | 'reading' | 'training' | 'complete'>('scanning');
  const [filesScanned, setFilesScanned] = useState(0);
  const [filesRead, setFilesRead] = useState(0);
  const [currentFile, setCurrentFile] = useState('');
  const [totalFiles, setTotalFiles] = useState(0);
  const [totalSize, setTotalSize] = useState(0);
  const [linesRead, setLinesRead] = useState(0);
  const [showTraining, setShowTraining] = useState(true);

  useEffect(() => {
    async function initializeModel() {
      setIsLoading(true);
      setShowTraining(true);
      setTrainingPhase('scanning');
      
      try {
        const progressCallback = {
          onScanProgress: (scanned: number, file?: string) => {
            setFilesScanned(scanned);
            if (file) {
              setCurrentFile(file);
            }
          },
          onReadProgress: (read: number, total: number, file: string, fileSize: number, totalSize: number, linesRead: number) => {
            setTrainingPhase('reading');
            setFilesRead(read);
            setTotalFiles(total);
            setCurrentFile(file);
            setTotalSize(totalSize);
            setLinesRead(linesRead);
          },
          onTrainingStart: () => {
            setTrainingPhase('training');
          },
          onTrainingComplete: () => {
            setTrainingPhase('complete');
          }
        };

        const newModel = config.modelType === 'code' 
          ? await createCodeModel(config.contextSize, config.maxOutputTokens, progressCallback)
          : await createTextModel(config.contextSize, config.maxOutputTokens);
        setModel(newModel);
        
        // Initialize with helpful suggestions
        if (newModel.getQuickCompletions) {
          setQuickCompletions(newModel.getQuickCompletions(''));
        }
      } catch (error) {
        setMessages(prev => [...prev, {
          type: 'assistant',
          content: `Error loading model: ${error}`,
          timestamp: new Date()
        }]);
      } finally {
        setIsLoading(false);
      }
    }

    initializeModel();
  }, [config.contextSize, config.maxOutputTokens, config.modelType]);

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

  const handleCommand = (command: string) => {
    const [cmd, ...args] = command.slice(1).split(' ');
    
    switch (cmd) {
      case 'help':
        setMessages(prev => [...prev, {
          type: 'assistant',
          content: `🤖 DumbGPT Smart Code Completion

SMART FEATURES:
• No need for exact context tokens - just start typing!
• Function detection: Type "function" to get function completions
• Class detection: Type "class" to get class completions  
• Import detection: Type "import" to get import completions
• Pattern matching: Type "if(", "for(", "=>" for smart completions
• Quick suggestions shown below input box

COMMANDS:
:help - Show this help
:config - Show current configuration  
:set <key> <value> - Set configuration
:clear - Clear messages
:quit - Exit application

EXAMPLES:
• "function add" → Smart function completion
• "const data = " → Variable assignment completion
• "if (" → Conditional statement completion
• "import " → Module import completion

Config keys: contextSize, maxOutputTokens, modelType`,
          timestamp: new Date()
        }]);
        break;
        
      case 'config':
        setMessages(prev => [...prev, {
          type: 'assistant',
          content: `Configuration:
Context Size: ${config.contextSize}
Max Tokens: ${config.maxOutputTokens}
Model Type: ${config.modelType}`,
          timestamp: new Date()
        }]);
        break;
        
      case 'set':
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
        break;
        
      case 'clear':
        setMessages([]);
        break;
        
      case 'quit':
        exit();
        break;
        
      default:
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


  if (showTraining) {
    return (
      <TrainingProgress
        phase={trainingPhase}
        filesScanned={filesScanned}
        filesRead={filesRead}
        currentFile={currentFile}
        totalFiles={totalFiles}
        totalSize={totalSize}
        linesRead={linesRead}
        onComplete={() => setShowTraining(false)}
      />
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
        {[
          { command: '/help', description: 'show help', shortcut: 'ctrl+x h' },
          { command: '/config', description: 'show configuration', shortcut: 'ctrl+x c' },
          { command: '/set', description: 'change settings', shortcut: 'ctrl+x s' },
          { command: '/train', description: 'retrain model', shortcut: 'ctrl+x t' },
          { command: '/clear', description: 'clear messages', shortcut: 'ctrl+x l' }
        ].map((item, index) => (
          <Box key={item.command} marginBottom={index < 4 ? 1 : 0}>
            <Box width={20}>
              <Text color={colors.command}>{item.command}</Text>
            </Box>
            <Box width={30}>
              <Text color={colors.textDim}>{item.description}</Text>
            </Box>
            <Box>
              <Text color={colors.textDim}>{item.shortcut}</Text>
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