// Color scheme inspired by opencode interface
export const colors = {
  // Primary colors
  primary: 'blue',        // Main accent color
  secondary: 'cyan',      // Commands and highlights
  
  // Text colors
  text: 'white',          // Main text
  textDim: 'gray',        // Secondary text, timestamps
  textMuted: 'blackBright', // Very dim text
  
  // Status colors
  success: 'green',       // Ready status, success messages
  warning: 'yellow',      // Loading, warnings
  error: 'red',           // Error messages
  
  // Interactive colors
  prompt: 'magenta',      // Input prompt
  cursor: 'magenta',      // Cursor background
  
  // Background colors
  inputBg: 'blueBright',  // Input background
  inputBorder: 'blue',    // Input border
  
  // UI elements
  command: 'cyan',        // Command names
  version: 'gray',        // Version text
  brand: 'white',         // Brand/title text
} as const;

export type ColorName = keyof typeof colors;