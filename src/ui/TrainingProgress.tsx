import React, { useState, useEffect } from 'react';
import { Box, Text, useInput } from 'ink';
import { colors } from './theme.js';

interface TrainingProgressProps {
  phase: 'scanning' | 'reading' | 'training' | 'complete';
  filesScanned: number;
  filesRead: number;
  currentFile: string;
  totalFiles: number;
  totalSize: number;
  linesRead: number;
  onComplete: () => void;
}

export function TrainingProgress({ 
  phase, 
  filesScanned, 
  filesRead, 
  currentFile, 
  totalFiles,
  totalSize,
  linesRead,
  onComplete 
}: TrainingProgressProps) {
  const [dots, setDots] = useState('');
  const [waitingForKey, setWaitingForKey] = useState(false);
  const [logEntries, setLogEntries] = useState<string[]>([]);

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes}B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
  };

  const formatNumber = (num: number) => {
    return num.toLocaleString();
  };

  useEffect(() => {
    const interval = setInterval(() => {
      setDots(prev => prev.length >= 3 ? '' : prev + '.');
    }, 500);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (phase === 'complete') {
      setTimeout(() => setWaitingForKey(true), 1000);
    }
  }, [phase, onComplete]);

  // Update log entries based on phase changes
  useEffect(() => {
    switch (phase) {
      case 'scanning':
        setLogEntries(['Starting file scan in node_modules...']);
        break;
      case 'reading':
        setLogEntries(prev => [...prev, `Found ${totalFiles} files, starting read process...`]);
        break;
      case 'training':
        setLogEntries(prev => [...prev, `Read ${filesRead} files, building Markov chains...`]);
        break;
      case 'complete':
        setLogEntries(prev => [...prev, `Training complete! Model ready with ${filesRead} files (${formatFileSize(totalSize)}, ${formatNumber(linesRead)} lines) processed.`]);
        break;
    }
  }, [phase, totalFiles, filesRead]);

  // Add log entry when currentFile changes
  useEffect(() => {
    if (currentFile) {
      if (phase === 'scanning') {
        setLogEntries(prev => {
          const newEntries = [...prev, `Found: ${currentFile}`];
          return newEntries.slice(-8);
        });
      } else if (phase === 'reading') {
        setLogEntries(prev => {
          const newEntries = [...prev, `Reading: ${currentFile}`];
          return newEntries.slice(-8);
        });
      }
    }
  }, [currentFile, phase]);

  useInput((input, key) => {
    if (waitingForKey && (key.return || input === ' ')) {
      onComplete();
    }
  });

  const getProgressBar = (current: number, total: number) => {
    const width = 40;
    const filled = Math.floor((current / total) * width);
    const bar = '█'.repeat(filled) + '░'.repeat(width - filled);
    return bar;
  };

  const getPhaseIcon = (currentPhase: string) => {
    switch (currentPhase) {
      case 'scanning': return '🔍';
      case 'reading': return '📖';
      case 'training': return '🧠';
      case 'complete': return '✅';
      default: return '⚡';
    }
  };

  const getPhaseColor = (currentPhase: string) => {
    switch (currentPhase) {
      case 'scanning': return colors.secondary;
      case 'reading': return colors.primary;
      case 'training': return colors.warning;
      case 'complete': return colors.success;
      default: return colors.text;
    }
  };

  return (
    <Box flexDirection="column" height="100%" width="100%" padding={2}>
      {/* Header */}
      <Box justifyContent="center" marginBottom={2}>
        <Text color={colors.brand} bold>
          {getPhaseIcon(phase)} Training DumbGPT Model
        </Text>
      </Box>

      {/* Main Content Area */}
      <Box flexDirection="row" height="100%" gap={2}>
        {/* Left Column - Progress & Stats */}
        <Box flexDirection="column" width="50%">
          {/* Current Phase */}
          <Box marginBottom={2}>
            <Text color={getPhaseColor(phase)} bold>
              {phase.charAt(0).toUpperCase() + phase.slice(1)}
              {phase !== 'complete' && (
                <Text color={colors.textDim}>{dots}</Text>
              )}
            </Text>
          </Box>

          {/* File Scanning Progress */}
          {phase === 'scanning' && (
            <Box flexDirection="column" marginBottom={3}>
              <Text color={colors.textDim}>
                Scanning node_modules for JS/TS files...
              </Text>
              <Box marginTop={1}>
                <Text color={colors.secondary}>
                  Found: {filesScanned} files
                </Text>
              </Box>
            </Box>
          )}

          {/* File Reading Progress */}
          {(phase === 'reading' || phase === 'training' || phase === 'complete') && (
            <Box flexDirection="column" marginBottom={3}>
              <Box marginBottom={1}>
                <Text color={colors.textDim}>
                  Reading Files: {filesRead}/{totalFiles}
                </Text>
              </Box>
              
              <Box marginBottom={1}>
                <Text color={colors.primary}>
                  {getProgressBar(filesRead, totalFiles)}
                </Text>
              </Box>
              
              <Box marginBottom={1}>
                <Text color={colors.textDim}>
                  {Math.round((filesRead / totalFiles) * 100)}%
                </Text>
              </Box>
            </Box>
          )}

          {/* Statistics Card */}
          {(phase === 'reading' || phase === 'training' || phase === 'complete') && (
            <Box flexDirection="column" marginBottom={3}>
              <Box marginBottom={1}>
                <Text color={colors.textDim} bold>Statistics:</Text>
              </Box>
              <Box 
                borderStyle="round" 
                borderColor={colors.textDim} 
                paddingX={2} 
                paddingY={1}
                flexDirection="column"
                gap={1}
              >
                <Box justifyContent="space-between">
                  <Text color={colors.text}>Files Processed:</Text>
                  <Text color={colors.secondary}>{formatNumber(filesRead)}</Text>
                </Box>
                <Box justifyContent="space-between">
                  <Text color={colors.text}>Total Size:</Text>
                  <Text color={colors.secondary}>{formatFileSize(totalSize)}</Text>
                </Box>
                <Box justifyContent="space-between">
                  <Text color={colors.text}>Lines of Code:</Text>
                  <Text color={colors.secondary}>{formatNumber(linesRead)}</Text>
                </Box>
              </Box>
            </Box>
          )}

          {/* Status Messages */}
          <Box flexDirection="column" marginBottom={2}>
            {phase === 'training' && (
              <Box>
                <Text color={colors.warning}>
                  🧠 Building Markov chains{dots}
                </Text>
                <Box marginTop={1}>
                  <Text color={colors.textDim}>
                    This may take a moment...
                  </Text>
                </Box>
              </Box>
            )}
            {phase === 'complete' && !waitingForKey && (
              <Box>
                <Text color={colors.success} bold>
                  ✅ Training Complete!
                </Text>
                <Box marginTop={1}>
                  <Text color={colors.textDim}>
                    Model ready for use
                  </Text>
                </Box>
              </Box>
            )}
            {phase === 'complete' && waitingForKey && (
              <Box>
                <Text color={colors.brand} bold>
                  Press Enter or Space to continue...
                </Text>
              </Box>
            )}
          </Box>

          {/* Phase Indicators */}
          <Box justifyContent="space-between" marginTop="auto">
            <Text color={phase === 'scanning' ? colors.secondary : colors.textMuted}>
              {phase === 'scanning' ? '🔍' : '✓'} Scan
            </Text>
            <Text color={phase === 'reading' ? colors.primary : 
                       ['training', 'complete'].includes(phase) ? colors.textMuted : colors.textDim}>
              {phase === 'reading' ? '📖' : 
               ['training', 'complete'].includes(phase) ? '✓' : '○'} Read
            </Text>
            <Text color={phase === 'training' ? colors.warning : 
                       phase === 'complete' ? colors.textMuted : colors.textDim}>
              {phase === 'training' ? '🧠' : 
               phase === 'complete' ? '✓' : '○'} Train
            </Text>
            <Text color={phase === 'complete' ? colors.success : colors.textDim}>
              {phase === 'complete' ? '✅' : '○'} Done
            </Text>
          </Box>
        </Box>

        {/* Right Column - Activity Log */}
        <Box flexDirection="column" width="50%">
          <Box marginBottom={1}>
            <Text color={colors.textDim} bold>Activity Log:</Text>
          </Box>
          <Box 
            borderStyle="single" 
            borderColor={colors.textDim} 
            height="100%"
            paddingX={1} 
            flexDirection="column"
          >
            {logEntries.slice(-15).map((entry, index) => (
              <Box key={index}>
                <Text color={colors.textMuted}>
                  {entry.length > 60 ? `${entry.slice(0, 57)}...` : entry}
                </Text>
              </Box>
            ))}
          </Box>
        </Box>
      </Box>
    </Box>
  );
}