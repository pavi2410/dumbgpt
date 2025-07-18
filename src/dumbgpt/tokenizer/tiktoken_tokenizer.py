import tiktoken


class TikTokenTokenizer:
    """Tiktoken-based tokenizer for DumbGPT."""
    
    def __init__(self, encoding_name: str = "gpt2"):
        """Initialize with tiktoken encoding.
        
        Args:
            encoding_name: Name of the tiktoken encoding ('gpt2', 'r50k_base', 'p50k_base', 'cl100k_base')
        """
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.encoding.n_vocab
        
        # Special tokens
        self.pad_token = "<|endoftext|>"  # GPT-2 uses this as padding
        self.unk_token = "<|endoftext|>"  # No explicit UNK in GPT-2
        self.bos_token = "<|endoftext|>"  # Beginning of sequence
        self.eos_token = "<|endoftext|>"  # End of sequence
        
        # Get token IDs - allow special tokens
        self.pad_token_id = self.encoding.encode(self.pad_token, allowed_special={self.pad_token})[0]
        self.unk_token_id = self.pad_token_id
        self.bos_token_id = self.pad_token_id
        self.eos_token_id = self.pad_token_id
    
    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        return self.encoding.encode(text)
    
    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text."""
        return self.encoding.decode(tokens)
    
    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """Encode multiple texts."""
        return [self.encode(text) for text in texts]
    
    def decode_batch(self, token_lists: list[list[int]]) -> list[str]:
        """Decode multiple token lists."""
        return [self.decode(tokens) for tokens in token_lists]
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.vocab_size