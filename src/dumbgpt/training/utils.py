import os
import pickle
from ..model.transformer import GPTModel


def save_model(model: GPTModel, filepath: str):
    """
    Save a GPT model to disk.
    
    Args:
        model: GPTModel instance to save
        filepath: Path to save the model
    """
    if os.path.dirname(filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Get all model parameters
    parameters = _get_model_parameters(model)
    
    # Save model configuration and parameters
    model_data = {
        'parameters': parameters,
        'config': {
            'vocab_size': model.vocab_size,
            'd_model': model.d_model,
            'num_heads': model.num_heads,
            'd_ff': model.d_ff,
            'num_layers': model.num_layers,
            'max_seq_len': model.max_seq_len
        }
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)


def load_model(filepath: str) -> GPTModel:
    """
    Load a GPT model from disk.
    
    Args:
        filepath: Path to load the model from
        
    Returns:
        GPTModel instance with loaded parameters
    """
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    
    config = model_data['config']
    parameters = model_data['parameters']
    
    # Create new model with same configuration
    model = GPTModel(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        num_layers=config['num_layers'],
        max_seq_len=config['max_seq_len']
    )
    
    # Load parameters into model
    _set_model_parameters(model, parameters)
    
    return model


def _get_model_parameters(model: GPTModel) -> dict:
    """
    Get all model parameters as a dictionary.
    
    Args:
        model: GPTModel instance
        
    Returns:
        Dictionary of parameter arrays
    """
    parameters = {}
    
    # Embedding parameters
    parameters['embedding.weight'] = model.embedding.weight.copy()
    
    # Transformer block parameters
    for i, block in enumerate(model.transformer_blocks):
        # Attention parameters
        parameters[f'block_{i}.attention.W_q.weight'] = block.attention.W_q.weight.copy()
        parameters[f'block_{i}.attention.W_q.bias'] = block.attention.W_q.bias.copy()
        parameters[f'block_{i}.attention.W_k.weight'] = block.attention.W_k.weight.copy()
        parameters[f'block_{i}.attention.W_k.bias'] = block.attention.W_k.bias.copy()
        parameters[f'block_{i}.attention.W_v.weight'] = block.attention.W_v.weight.copy()
        parameters[f'block_{i}.attention.W_v.bias'] = block.attention.W_v.bias.copy()
        parameters[f'block_{i}.attention.W_o.weight'] = block.attention.W_o.weight.copy()
        parameters[f'block_{i}.attention.W_o.bias'] = block.attention.W_o.bias.copy()
        
        # Feed-forward parameters
        parameters[f'block_{i}.feed_forward.linear1.weight'] = block.feed_forward.linear1.weight.copy()
        parameters[f'block_{i}.feed_forward.linear1.bias'] = block.feed_forward.linear1.bias.copy()
        parameters[f'block_{i}.feed_forward.linear2.weight'] = block.feed_forward.linear2.weight.copy()
        parameters[f'block_{i}.feed_forward.linear2.bias'] = block.feed_forward.linear2.bias.copy()
        
        # Layer norm parameters
        parameters[f'block_{i}.norm1.weight'] = block.norm1.weight.copy()
        parameters[f'block_{i}.norm1.bias'] = block.norm1.bias.copy()
        parameters[f'block_{i}.norm2.weight'] = block.norm2.weight.copy()
        parameters[f'block_{i}.norm2.bias'] = block.norm2.bias.copy()
    
    # Final layer norm
    parameters['ln_f.weight'] = model.ln_f.weight.copy()
    parameters['ln_f.bias'] = model.ln_f.bias.copy()
    
    # Output layer
    parameters['lm_head.weight'] = model.lm_head.weight.copy()
    parameters['lm_head.bias'] = model.lm_head.bias.copy()
    
    return parameters


def _set_model_parameters(model: GPTModel, parameters: dict):
    """
    Set model parameters from dictionary.
    
    Args:
        model: GPTModel instance
        parameters: Dictionary of parameter arrays
    """
    # Embedding parameters
    model.embedding.weight = parameters['embedding.weight'].copy()
    
    # Transformer block parameters
    for i, block in enumerate(model.transformer_blocks):
        # Attention parameters
        block.attention.W_q.weight = parameters[f'block_{i}.attention.W_q.weight'].copy()
        block.attention.W_q.bias = parameters[f'block_{i}.attention.W_q.bias'].copy()
        block.attention.W_k.weight = parameters[f'block_{i}.attention.W_k.weight'].copy()
        block.attention.W_k.bias = parameters[f'block_{i}.attention.W_k.bias'].copy()
        block.attention.W_v.weight = parameters[f'block_{i}.attention.W_v.weight'].copy()
        block.attention.W_v.bias = parameters[f'block_{i}.attention.W_v.bias'].copy()
        block.attention.W_o.weight = parameters[f'block_{i}.attention.W_o.weight'].copy()
        block.attention.W_o.bias = parameters[f'block_{i}.attention.W_o.bias'].copy()
        
        # Feed-forward parameters
        block.feed_forward.linear1.weight = parameters[f'block_{i}.feed_forward.linear1.weight'].copy()
        block.feed_forward.linear1.bias = parameters[f'block_{i}.feed_forward.linear1.bias'].copy()
        block.feed_forward.linear2.weight = parameters[f'block_{i}.feed_forward.linear2.weight'].copy()
        block.feed_forward.linear2.bias = parameters[f'block_{i}.feed_forward.linear2.bias'].copy()
        
        # Layer norm parameters
        block.norm1.weight = parameters[f'block_{i}.norm1.weight'].copy()
        block.norm1.bias = parameters[f'block_{i}.norm1.bias'].copy()
        block.norm2.weight = parameters[f'block_{i}.norm2.weight'].copy()
        block.norm2.bias = parameters[f'block_{i}.norm2.bias'].copy()
    
    # Final layer norm
    model.ln_f.weight = parameters['ln_f.weight'].copy()
    model.ln_f.bias = parameters['ln_f.bias'].copy()
    
    # Output layer
    model.lm_head.weight = parameters['lm_head.weight'].copy()
    model.lm_head.bias = parameters['lm_head.bias'].copy()