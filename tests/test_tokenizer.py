import pytest
from pathlib import Path
from dumbgpt.tokenizer.tokenizer import CharTokenizer


class TestCharTokenizer:
    """Test suite for CharTokenizer implementation."""
    
    def test_tokenizer_initialization(self):
        """Test that tokenizer can be initialized."""
        tokenizer = CharTokenizer()
        assert tokenizer is not None
        assert hasattr(tokenizer, 'vocab_size')
        assert hasattr(tokenizer, 'vocab')
        assert hasattr(tokenizer, 'encode')
        assert hasattr(tokenizer, 'decode')
    
    def test_build_vocab_simple(self):
        """Test vocabulary building with simple text."""
        tokenizer = CharTokenizer()
        text = "hello world"
        tokenizer.build_vocab([text])
        
        # Should contain all unique characters
        expected_chars = set(text)
        for char in expected_chars:
            assert char in tokenizer.vocab
        
        # Vocab size should include special tokens
        assert tokenizer.vocab_size > len(expected_chars)
    
    def test_build_vocab_multiple_texts(self):
        """Test vocabulary building with multiple texts."""
        tokenizer = CharTokenizer()
        texts = ["hello", "world", "python"]
        tokenizer.build_vocab(texts)
        
        all_chars = set("".join(texts))
        for char in all_chars:
            assert char in tokenizer.vocab
    
    def test_special_tokens_exist(self):
        """Test that special tokens are properly added."""
        tokenizer = CharTokenizer()
        tokenizer.build_vocab(["hello"])
        
        # Check for common special tokens
        assert tokenizer.pad_token in tokenizer.vocab
        assert tokenizer.unk_token in tokenizer.vocab
        assert tokenizer.bos_token in tokenizer.vocab
        assert tokenizer.eos_token in tokenizer.vocab
    
    def test_encode_simple_text(self):
        """Test encoding simple text to token IDs."""
        tokenizer = CharTokenizer()
        text = "hello"
        tokenizer.build_vocab([text])
        
        tokens = tokenizer.encode(text)
        
        # Should return list of integers
        assert isinstance(tokens, list)
        assert all(isinstance(token, int) for token in tokens)
        assert len(tokens) == len(text)
        
        # All tokens should be valid IDs
        for token in tokens:
            assert 0 <= token < tokenizer.vocab_size
    
    def test_decode_simple_tokens(self):
        """Test decoding token IDs back to text."""
        tokenizer = CharTokenizer()
        text = "hello"
        tokenizer.build_vocab([text])
        
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        
        assert decoded == text
    
    def test_encode_decode_roundtrip(self):
        """Test that encode->decode returns original text."""
        tokenizer = CharTokenizer()
        test_texts = [
            "hello world",
            "Python is awesome!",
            "123 + 456 = 579",
            "Special chars: !@#$%^&*()",
            "Mixed\nlines\twith\ttabs"
        ]
        
        for text in test_texts:
            tokenizer.build_vocab([text])
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            assert decoded == text, f"Failed for text: {text}"
    
    def test_unknown_character_handling(self):
        """Test handling of unknown characters."""
        tokenizer = CharTokenizer()
        tokenizer.build_vocab(["hello"])
        
        # Try to encode character not in vocab
        tokens = tokenizer.encode("world")  # 'w', 'o', 'r', 'd' not in vocab
        
        # Should contain UNK tokens
        unk_id = tokenizer.vocab[tokenizer.unk_token]
        assert unk_id in tokens
    
    def test_empty_text_handling(self):
        """Test handling of empty text."""
        tokenizer = CharTokenizer()
        tokenizer.build_vocab(["hello"])
        
        # Empty string should return empty list
        tokens = tokenizer.encode("")
        assert tokens == []
        
        # Empty token list should return empty string
        decoded = tokenizer.decode([])
        assert decoded == ""
    
    def test_whitespace_preservation(self):
        """Test that whitespace is properly handled."""
        tokenizer = CharTokenizer()
        text = "hello world\nwith\ttabs"
        tokenizer.build_vocab([text])
        
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        
        assert decoded == text
        assert '\n' in tokenizer.vocab
        assert '\t' in tokenizer.vocab
        assert ' ' in tokenizer.vocab
    
    def test_code_text_handling(self):
        """Test tokenization of code text."""
        tokenizer = CharTokenizer()
        code = """class Solution {
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if(p != null && q != null && q.val == p.val) {
            return true;
        }
        return false;
    }
}"""
        tokenizer.build_vocab([code])
        
        tokens = tokenizer.encode(code)
        decoded = tokenizer.decode(tokens)
        
        assert decoded == code
        # Check that code-specific characters are in vocab
        assert '{' in tokenizer.vocab
        assert '}' in tokenizer.vocab
        assert '(' in tokenizer.vocab
        assert ')' in tokenizer.vocab
        assert ';' in tokenizer.vocab
    
    def test_vocab_consistency(self):
        """Test that vocab mapping is consistent."""
        tokenizer = CharTokenizer()
        tokenizer.build_vocab(["hello world"])
        
        # Each character should map to unique ID
        char_to_id = {}
        for char in "hello world":
            token_id = tokenizer.vocab[char]
            if char in char_to_id:
                assert char_to_id[char] == token_id
            else:
                char_to_id[char] = token_id
        
        # IDs should be within valid range
        for token_id in char_to_id.values():
            assert 0 <= token_id < tokenizer.vocab_size
    
    def test_large_vocab_performance(self):
        """Test performance with larger vocabulary."""
        tokenizer = CharTokenizer()
        
        # Create text with many unique characters
        import string
        large_text = string.ascii_letters + string.digits + string.punctuation
        tokenizer.build_vocab([large_text])
        
        # Should handle encoding/decoding efficiently
        tokens = tokenizer.encode(large_text)
        decoded = tokenizer.decode(tokens)
        
        assert decoded == large_text
        assert len(tokens) == len(large_text)
    
    def test_corpus_integration(self):
        """Test integration with actual corpus files."""
        tokenizer = CharTokenizer()
        
        # Test with small sample from corpus
        sample_texts = [
            "The Project Gutenberg eBook of Alice's Adventures in Wonderland",
            "class Solution { public boolean isSameTree() { return true; } }"
        ]
        
        tokenizer.build_vocab(sample_texts)
        
        for text in sample_texts:
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            assert decoded == text
    
    def test_vocab_size_property(self):
        """Test that vocab_size property is correct."""
        tokenizer = CharTokenizer()
        text = "hello"
        tokenizer.build_vocab([text])
        
        # vocab_size should match actual vocab length
        assert tokenizer.vocab_size == len(tokenizer.vocab)
        
        # Should be greater than just the unique characters
        unique_chars = set(text)
        assert tokenizer.vocab_size > len(unique_chars)  # Due to special tokens
    
    def test_token_ids_are_unique(self):
        """Test that all token IDs are unique."""
        tokenizer = CharTokenizer()
        tokenizer.build_vocab(["hello world!"])
        
        # All values in vocab should be unique
        ids = list(tokenizer.vocab.values())
        assert len(ids) == len(set(ids))
    
    def test_special_token_properties(self):
        """Test that special tokens have expected properties."""
        tokenizer = CharTokenizer()
        tokenizer.build_vocab(["hello"])
        
        # Special tokens should be accessible as properties
        assert hasattr(tokenizer, 'pad_token')
        assert hasattr(tokenizer, 'unk_token')
        assert hasattr(tokenizer, 'bos_token')
        assert hasattr(tokenizer, 'eos_token')
        
        # They should be strings
        assert isinstance(tokenizer.pad_token, str)
        assert isinstance(tokenizer.unk_token, str)
        assert isinstance(tokenizer.bos_token, str)
        assert isinstance(tokenizer.eos_token, str)


class TestTokenizerIntegration:
    """Integration tests with corpus data."""
    
    def test_build_vocab_from_corpus_sample(self):
        """Test building vocab from actual corpus sample."""
        tokenizer = CharTokenizer()
        
        # Sample texts that might be in corpus
        corpus_samples = [
            "Alice was beginning to get very tired of sitting by her sister",
            "class TreeNode { int val; TreeNode left; TreeNode right; }",
            "Hello, World!",
            "The quick brown fox jumps over the lazy dog."
        ]
        
        tokenizer.build_vocab(corpus_samples)
        
        # Test that all samples can be encoded and decoded
        for text in corpus_samples:
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            assert decoded == text
    
    def test_tokenizer_with_mixed_content(self):
        """Test tokenizer with mixed novel and code content."""
        tokenizer = CharTokenizer()
        
        mixed_content = [
            "Once upon a time, there was a programmer.",
            "def hello_world():\n    print('Hello, World!')",
            "She lived in a world of {brackets: 'and', semicolons: ';'}",
            "// This is a comment\nint x = 42;"
        ]
        
        tokenizer.build_vocab(mixed_content)
        
        for text in mixed_content:
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            assert decoded == text
            
            # Verify no information loss
            assert len(tokens) == len(text)


# Fixtures for common test data
@pytest.fixture
def sample_tokenizer():
    """Fixture providing a tokenizer with sample vocabulary."""
    tokenizer = CharTokenizer()
    sample_texts = [
        "hello world",
        "python programming",
        "class Solution { return true; }"
    ]
    tokenizer.build_vocab(sample_texts)
    return tokenizer


@pytest.fixture
def corpus_sample():
    """Fixture providing sample corpus-like data."""
    return [
        "The Project Gutenberg eBook of Alice's Adventures in Wonderland",
        "This ebook is for the use of anyone anywhere in the United States",
        "class Solution {",
        "    public boolean isSameTree(TreeNode p, TreeNode q) {",
        "        if(p != null && q != null && q.val == p.val) {",
        "            return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);",
        "        }",
        "        return p == null && q == null;",
        "    }",
        "}"
    ]


class TestTokenizerWithFixtures:
    """Tests using fixtures."""
    
    def test_sample_tokenizer_basic(self, sample_tokenizer):
        """Test basic functionality with sample tokenizer."""
        text = "hello python"
        tokens = sample_tokenizer.encode(text)
        decoded = sample_tokenizer.decode(tokens)
        assert decoded == text
    
    def test_corpus_sample_processing(self, corpus_sample):
        """Test processing corpus sample data."""
        tokenizer = CharTokenizer()
        tokenizer.build_vocab(corpus_sample)
        
        for text in corpus_sample:
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            assert decoded == text