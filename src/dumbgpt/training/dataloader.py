import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple


class DataLoader:
    """
    Data loader for training GPT model.
    
    Loads text data from corpus files and creates training batches
    with input-target pairs for autoregressive training.
    """
    
    def __init__(
        self,
        corpus_paths: Optional[List[str]],
        tokenizer,
        seq_length: int,
        batch_size: int,
        sample_texts: Optional[List[str]] = None
    ):
        """
        Initialize DataLoader.
        
        Args:
            corpus_paths: List of paths to corpus files
            tokenizer: Tokenizer instance
            seq_length: Length of sequences for training
            batch_size: Number of sequences in each batch
            sample_texts: Optional sample texts for testing
        """
        self.corpus_paths = corpus_paths
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.batch_size = batch_size
        
        # Load and tokenize all data
        self.token_data = self._load_and_tokenize_data(sample_texts)
    
    def _load_and_tokenize_data(self, sample_texts: Optional[List[str]] = None) -> np.ndarray:
        """
        Load text data from files and tokenize it.
        
        Args:
            sample_texts: Optional sample texts for testing
            
        Returns:
            Array of token IDs
        """
        if sample_texts is not None:
            # Use sample texts for testing
            all_text = " ".join(sample_texts)
        else:
            # Load from corpus files
            all_text = self._load_corpus_files()
        
        # Tokenize the text
        token_ids = self.tokenizer.encode(all_text)
        
        return np.array(token_ids)
    
    def _load_corpus_files(self) -> str:
        """
        Load text from corpus files.
        
        Returns:
            Combined text from all corpus files
        """
        if not self.corpus_paths:
            raise ValueError("No corpus paths provided")
        
        all_text = []
        
        for path in self.corpus_paths:
            path = Path(path)
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    all_text.append(text)
            else:
                print(f"Warning: File {path} not found")
        
        return "\n".join(all_text)
    
    def get_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a batch of training data.
        
        Returns:
            Tuple of (input_ids, target_ids) with shape (batch_size, seq_length)
        """
        batch_input = []
        batch_target = []
        
        for _ in range(self.batch_size):
            # Random starting position
            max_start = len(self.token_data) - self.seq_length - 1
            if max_start <= 0:
                # If data is too short, repeat it
                repeated_data = np.tile(self.token_data, (self.seq_length + 1) // len(self.token_data) + 1)
                start_idx = np.random.randint(0, len(repeated_data) - self.seq_length - 1)
                sequence = repeated_data[start_idx:start_idx + self.seq_length + 1]
            else:
                start_idx = np.random.randint(0, max_start)
                sequence = self.token_data[start_idx:start_idx + self.seq_length + 1]
            
            # Input is all tokens except the last
            input_seq = sequence[:-1]
            # Target is all tokens except the first (shifted by 1)
            target_seq = sequence[1:]
            
            batch_input.append(input_seq)
            batch_target.append(target_seq)
        
        return np.array(batch_input), np.array(batch_target)
    
    def get_data_size(self) -> int:
        """
        Get the total number of tokens in the dataset.
        
        Returns:
            Number of tokens
        """
        return len(self.token_data)
    
    def get_num_batches(self) -> int:
        """
        Get approximate number of batches in the dataset.
        
        Returns:
            Number of batches
        """
        return len(self.token_data) // (self.batch_size * self.seq_length)