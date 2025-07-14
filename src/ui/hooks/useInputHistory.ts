import { useState, useCallback } from 'react';
import { useInput, useApp } from 'ink';

interface UseInputHistoryOptions {
  onSubmit: (value: string) => void;
  disabled?: boolean;
  maxLines?: number;
  maxHistory?: number;
}

export function useInputHistory({
  onSubmit,
  disabled = false,
  maxLines = 5,
  maxHistory = 50
}: UseInputHistoryOptions) {
  const [inputValue, setInputValue] = useState('');
  const [history, setHistory] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const { exit } = useApp();

  const addToHistory = useCallback((value: string) => {
    if (value.trim()) {
      setHistory(prev => {
        const newHistory = prev.filter(item => item !== value);
        return [...newHistory, value].slice(-maxHistory);
      });
      setHistoryIndex(-1);
    }
  }, [maxHistory]);

  const navigateHistory = useCallback((direction: 'up' | 'down') => {
    if (direction === 'up') {
      if (history.length > 0) {
        const newIndex = historyIndex + 1;
        if (newIndex < history.length) {
          setHistoryIndex(newIndex);
          setInputValue(history[history.length - 1 - newIndex]);
        }
      }
    } else {
      if (historyIndex > 0) {
        const newIndex = historyIndex - 1;
        setHistoryIndex(newIndex);
        setInputValue(history[history.length - 1 - newIndex]);
      } else if (historyIndex === 0) {
        setHistoryIndex(-1);
        setInputValue('');
      }
    }
  }, [history, historyIndex]);

  const handleMultilineInput = useCallback(() => {
    const lines = inputValue.split('\n');
    if (lines.length < maxLines) {
      setInputValue(prev => prev.slice(0, -1) + '\n');
    }
  }, [inputValue, maxLines]);

  const handleBackspace = useCallback(() => {
    setInputValue(prev => {
      if (prev.length === 0) return prev;
      return prev.slice(0, -1);
    });
  }, []);

  const handleSubmit = useCallback(() => {
    addToHistory(inputValue);
    onSubmit(inputValue);
    setInputValue('');
  }, [inputValue, addToHistory, onSubmit]);

  const handleCharacterInput = useCallback((input: string) => {
    setInputValue(prev => prev + input);
  }, []);

  useInput((input, key) => {
    if (key.ctrl && input === 'c') {
      exit();
      return;
    }

    if (disabled) return;

    // History navigation
    if (key.upArrow) {
      navigateHistory('up');
      return;
    }

    if (key.downArrow) {
      navigateHistory('down');
      return;
    }

    // Multiline input with backslash+enter
    if (key.return && inputValue.endsWith('\\')) {
      handleMultilineInput();
      return;
    }

    // Submit input
    if (key.return) {
      handleSubmit();
      return;
    }

    // Backspace handling
    if (key.backspace || key.delete) {
      handleBackspace();
      return;
    }

    // Skip ctrl combinations
    if (key.ctrl) return;

    // Regular character input
    if (input) {
      handleCharacterInput(input);
    }
  });

  const lines = inputValue.split('\n');

  return {
    inputValue,
    setInputValue,
    history,
    historyIndex,
    lines,
    maxLines,
    stats: {
      currentLines: lines.length,
      historyCount: history.length
    }
  };
}