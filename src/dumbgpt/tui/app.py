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
from ..tokenizer.tokenizer import CharTokenizer


def load_model():
    """Load the DumbGPT model."""
    model_path = Path("models/pytorch_model.pt")
    if not model_path.exists():
        return None, None
    
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['config']
    
    model = GPTModel(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        num_layers=config['num_layers'],
        max_seq_len=config['max_seq_len']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    tokenizer = CharTokenizer()
    tokenizer.vocab = checkpoint['tokenizer_vocab']
    tokenizer.reverse_vocab = {v: k for k, v in checkpoint['tokenizer_vocab'].items()}
    tokenizer.vocab_size = len(checkpoint['tokenizer_vocab'])
    
    return model, tokenizer


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
        
        if self.model and self.tokenizer:
            params = sum(p.numel() for p in self.model.parameters())
            chat.write(f"Model loaded! {params:,} parameters\n")
            chat.write("Type a message to start chatting.\n")
        else:
            chat.write("Error: Could not load model!\n")
            chat.write("Make sure models/pytorch_model.pt exists.\n")
    
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
        
        # Generate model response
        if self.model and self.tokenizer:
            try:
                # Tokenize input
                tokens = self.tokenizer.encode(message)
                context = torch.tensor(tokens).unsqueeze(0)
                
                # Generate response
                with torch.no_grad():
                    generated = self.model.generate(
                        context, 
                        max_length=50, 
                        temperature=0.8
                    )
                    response = self.tokenizer.decode(generated.tolist())
                
                chat.write(f"DumbGPT: {response}\n")
            except Exception as e:
                chat.write(f"Error generating response: {str(e)}\n")
        else:
            chat.write("Error: Model not loaded\n")
    
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