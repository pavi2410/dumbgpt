import numpy as np
from typing import Dict, Any


class SGD:
    """
    Stochastic Gradient Descent optimizer.
    
    Simple gradient descent with momentum support.
    """
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        """
        Initialize SGD optimizer.
        
        Args:
            learning_rate: Learning rate for parameter updates
            momentum: Momentum factor for accelerated gradient descent
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {}
    
    def update_param(self, param: np.ndarray, grad: np.ndarray, param_name: str = "default") -> np.ndarray:
        """
        Update parameter using SGD.
        
        Args:
            param: Parameter array
            grad: Gradient array
            param_name: Name of parameter (for momentum tracking)
            
        Returns:
            Updated parameter
        """
        if self.momentum > 0:
            # Initialize velocity if not exists
            if param_name not in self.velocity:
                self.velocity[param_name] = np.zeros_like(param)
            
            # Update velocity: v = momentum * v + grad
            self.velocity[param_name] = self.momentum * self.velocity[param_name] + grad
            
            # Update parameter: param = param - lr * v
            updated_param = param - self.learning_rate * self.velocity[param_name]
        else:
            # Simple SGD: param = param - lr * grad
            updated_param = param - self.learning_rate * grad
        
        return updated_param
    
    def zero_grad(self):
        """Reset gradients (placeholder for interface consistency)."""
        pass
    
    def step(self, parameters: Dict[str, np.ndarray], gradients: Dict[str, np.ndarray]):
        """
        Update all parameters.
        
        Args:
            parameters: Dictionary of parameter arrays
            gradients: Dictionary of gradient arrays
        """
        for param_name in parameters:
            if param_name in gradients:
                parameters[param_name] = self.update_param(
                    parameters[param_name], 
                    gradients[param_name], 
                    param_name
                )


class Adam:
    """
    Adam optimizer with adaptive learning rates.
    
    Implements the Adam algorithm with bias correction.
    """
    
    def __init__(
        self, 
        learning_rate: float = 0.001, 
        beta1: float = 0.9, 
        beta2: float = 0.999, 
        epsilon: float = 1e-8
    ):
        """
        Initialize Adam optimizer.
        
        Args:
            learning_rate: Learning rate
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Moment estimates
        self.m = {}  # First moment (mean)
        self.v = {}  # Second moment (variance)
        self.t = 0   # Time step
    
    def update_param(self, param: np.ndarray, grad: np.ndarray, param_name: str = "default") -> np.ndarray:
        """
        Update parameter using Adam.
        
        Args:
            param: Parameter array
            grad: Gradient array
            param_name: Name of parameter
            
        Returns:
            Updated parameter
        """
        # Initialize moments if not exists
        if param_name not in self.m:
            self.m[param_name] = np.zeros_like(param)
            self.v[param_name] = np.zeros_like(param)
        
        # Update biased first moment estimate
        self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grad
        
        # Update biased second moment estimate
        self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (grad ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat = self.m[param_name] / (1 - self.beta1 ** self.t)
        
        # Compute bias-corrected second moment estimate
        v_hat = self.v[param_name] / (1 - self.beta2 ** self.t)
        
        # Update parameter
        updated_param = param - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return updated_param
    
    def zero_grad(self):
        """Reset gradients (placeholder for interface consistency)."""
        pass
    
    def step(self, parameters: Dict[str, np.ndarray], gradients: Dict[str, np.ndarray]):
        """
        Update all parameters.
        
        Args:
            parameters: Dictionary of parameter arrays
            gradients: Dictionary of gradient arrays
        """
        self.t += 1  # Increment time step
        
        for param_name in parameters:
            if param_name in gradients:
                parameters[param_name] = self.update_param(
                    parameters[param_name], 
                    gradients[param_name], 
                    param_name
                )