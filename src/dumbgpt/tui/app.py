"""
Minimal DumbGPT Chat Interface
"""

import torch
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Input, Log, Static
from textual.binding import Binding
from textual import on

from ..model.transformer import GPTModel
from ..tokenizer.tiktoken_tokenizer import TikTokenTokenizer


def load_model():
    """Load tiktoken tokenizer for demo."""
    # For now, just return tokenizer without model
    # The saved model has different vocab size than tiktoken
    tokenizer = TikTokenTokenizer()
    return None, tokenizer


class DumbGPTApp(App):
    """Minimal chat interface for DumbGPT."""
    
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear", "Clear"),
    ]
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None
    
    def compose(self) -> ComposeResult:
        yield Static("DumbGPT Chat", id="title")
        yield Log(id="chat")
        yield Input(placeholder="Type your message...", id="input")
    
    def on_mount(self):
        """Setup initial state."""
        chat = self.query_one("#chat")
        chat.write("Loading DumbGPT model...\n")
        
        # Load model
        self.model, self.tokenizer = load_model()
        
        if self.tokenizer:
            chat.write(f"TikToken tokenizer loaded! Vocab size: {self.tokenizer.vocab_size:,}\n")
            chat.write("Type a message to test tokenization (no model inference yet).\n")
        else:
            chat.write("Error: Could not load tokenizer!\n")
    
    @on(Input.Submitted)
    def send_message(self, event: Input.Submitted):
        """Handle message input."""
        message = event.value.strip()
        if not message:
            return
        
        # Add user message
        chat = self.query_one("#chat")
        chat.write(f"You: {message}\n")
        
        # Clear input
        event.input.value = ""
        
        # Show tokenization demo
        if self.tokenizer:
            try:
                # Tokenize input
                tokens = self.tokenizer.encode(message)
                decoded = self.tokenizer.decode(tokens)
                
                chat.write(f"Tokens ({len(tokens)}): {tokens[:10]}{'...' if len(tokens) > 10 else ''}\n")
                chat.write(f"Decoded: {decoded}\n")
            except Exception as e:
                chat.write(f"Error tokenizing: {str(e)}\n")
        else:
            chat.write("Error: Tokenizer not loaded\n")
    
    def action_clear(self):
        """Clear chat."""
        chat = self.query_one("#chat")
        chat.clear()
        chat.write("Chat cleared!\n")


def main():
    """Run the minimal chat app."""
    app = DumbGPTApp()
    app.run()


if __name__ == "__main__":
    main()