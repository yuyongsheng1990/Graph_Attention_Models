import math
from typing import Optional

import torch


class Swish(torch.nn.Module):
    """Swish activation module"""
    def __init__(self, beta: Optional[float] = None):
        """Initialize the module.

        Args:
            beta (Optional[float], optional): Shape parameter. If None, it's a learnable parameter. Defaults to None.
        """
        super(Swish, self).__init__()
        # If beta is None, make it a learnable parameter
        if beta is None:
            self.beta = torch.nn.Parameter(torch.ones(1))
        # Otherwise, set it to a fixed constant
        else:
            self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Activation tensor.
        """
        return x * torch.nn.functional.sigmoid(self.beta * x)

class SwiGLU(torch.nn.Module):
    """SwiGLU activation module."""
    def __init__(self, dim: int):
        """Initialize the module.

        Args:
            dim (int): Dimension of the input tensor.
        """
        super(SwiGLU, self).__init__()
        self.swish = Swish()
        self.W = torch.nn.Parameter(torch.randn(dim, dim) / dim ** 0.5)
        self.V = torch.nn.Parameter(torch.randn(dim, dim) / dim ** 0.5)
        self.b = torch.nn.Parameter(torch.zeros(dim))
        self.c = torch.nn.Parameter(torch.zeros(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Activation tensor.
        """
        return self.swish(x @ self.W + self.b) * (x @ self.V + self.c)
        
        
class GEGLU(torch.nn.Module):
    """GEGLU activation module."""
    def __init__(self, dim: int):
        """Initialize the module.

        Args:
            dim (int): Dimension of the input tensor.
        """
        super(GEGLU, self).__init__()
        self.W = torch.nn.Parameter(torch.randn(dim, dim) / dim ** 0.5)
        self.V = torch.nn.Parameter(torch.randn(dim, dim) / dim ** 0.5)
        self.b = torch.nn.Parameter(torch.zeros(dim))
        self.c = torch.nn.Parameter(torch.zeros(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Activation tensor.
        """
        return torch.nn.functional.gelu(x @ self.W + self.b) * (x @ self.V + self.c)
    
class FFNGLU(torch.nn.Module):
    """FFN GLU activation module."""
    def __init__(self, dim: int):
        """Initialize the module.

        Args:
            dim (int): Dimension of the input tensor.
        """
        super(FFNGLU, self).__init__()
        inner_dim = math.ceil(dim * 2 / 3)
        self.W1 = torch.nn.Parameter(torch.randn(dim, inner_dim) /  inner_dim ** 0.5)
        self.W2 = torch.nn.Parameter(torch.randn(inner_dim, dim) / dim ** 0.5)
        self.V = torch.nn.Parameter(torch.randn(dim, inner_dim) / inner_dim ** 0.5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Activation tensor.
        """
        return (torch.nn.functional.sigmoid(x @ self.W1) * (x @ self.V)) @ self.W2
    
class FFNGEGLU(torch.nn.Module):
    """FFN GELU activation module."""
    def __init__(self, dim: int):  # 2048
        """Initialize the module.

        Args:
            dim (int): Dimension of the input tensor.
        """
        super(FFNGEGLU, self).__init__()  # 全连接层 + GELU激活函数
        inner_dim = math.ceil(dim * 2 / 3)    # 1366
        # 全连接层权重W1 / inner_dim平方根。这个操作的目的是对权重进行缩放，以确保权重W1的方差不受输入维度dim、inner_dim的影响，
        # 以确保在训练过程中权重的更新具有相对较小的幅度，从而提高模型的稳定性。
        self.W1 = torch.nn.Parameter(torch.randn(dim, inner_dim) /  inner_dim ** 0.5)  # 除以inner_dim的平方根
        self.W2 = torch.nn.Parameter(torch.randn(inner_dim, dim) / dim ** 0.5)
        self.V = torch.nn.Parameter(torch.randn(dim, inner_dim) / inner_dim ** 0.5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.
            - x @ self.W1，执行tensor x与W1的矩阵乘法。
            - GELU激活函数。
            - x @ self.V，执行tensor x与V的矩阵乘法。
        Returns:
            torch.Tensor: Activation tensor.
        """
        return (torch.nn.functional.gelu(x @ self.W1) * (x @ self.V)) @ self.W2
    
class FFNSwiGLU(torch.nn.Module):
    """FFN SwiGLU activation module."""
    def __init__(self, dim: int):
        """Initialize the module.

        Args:
            dim (int): Dimension of the input tensor.
        """
        super(FFNSwiGLU, self).__init__()
        inner_dim = math.ceil(dim * 2 / 3)
        self.swish = Swish(beta=1)
        self.W1 = torch.nn.Parameter(torch.randn(dim, inner_dim) /  inner_dim ** 0.5)
        self.W2 = torch.nn.Parameter(torch.randn(inner_dim, dim) / dim ** 0.5)
        self.V = torch.nn.Parameter(torch.randn(dim, inner_dim) / inner_dim ** 0.5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Activation tensor.
        """
        return (self.swish(x @ self.W1) * (x @ self.V)) @ self.W2
    
class Abs(torch.nn.Module):
    """Absolute value activation module."""
    def __init__(self):
        """Initialize the module."""
        super(Abs, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Activation tensor.
        """
        return torch.abs(x)

# Importable container for available activations
ACTIVATIONS = {
    "relu": torch.nn.ReLU,
    "gelu": torch.nn.GELU,
    "swish": Swish,
    "swiglu": SwiGLU,
    "geglu": GEGLU,
    "ffnglu": FFNGLU,
    "ffngeglu": FFNGEGLU,
    "ffnswiglu": FFNSwiGLU,
    "abs": Abs
}
