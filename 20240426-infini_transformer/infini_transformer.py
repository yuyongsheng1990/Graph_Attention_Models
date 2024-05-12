import math
from typing import Optional, Tuple

import torch
from torch import nn

from compressive_memory import CompressiveMemory
from activations import ACTIVATIONS


class InfiniTransformer(nn.Module):
    """Transformer layer with compressive memory."""

    def __init__(
        self,  # MoDInfiniTransformer()
        dim_input: int,  # 512
        dim_hidden: int,  # 2048
        dim_key: int,  # 64
        dim_value: int,  # 64
        num_heads: int,  # 8
        activation: str,  # glu
        segment_len: int,  # 256
        update: str = "linear",
        causal: bool = False,
        dropout: float = 0.0  # 0.1
    ):
        """Initializes the module.

        Args:
            dim_input (int): Input dimension.
            dim_hidden (int): Hidden dimension for the MLP.
            dim_key (int): Key dimension for the CompressiveMemory.
            dim_value (int): Value dimension for the CompressiveMemory.
            num_heads (int): Number of attention heads for the CompressiveMemory.
            activation (str): Activation function to use for the MLP. Must be a key in the ACTIVATIONS dictionary.
            segment_len (int): Segment length for the CompressiveMemory.
            update (str, optional): Type of memory update rule to use for the CompressiveMemory ("linear" or "delta"). Defaults to "linear".
            causal (bool, optional): Whether to use causal attention masking for the CompressiveMemory. Defaults to False.
            dropout (float, optional): Dropout rate for the MLP. Defaults to 0.0.
        """
        super(InfiniTransformer, self).__init__()

        # Multi-head attention
        self.attn = CompressiveMemory(
            dim_input, dim_key, dim_value, num_heads, segment_len, update, causal)  # 512, 64, 64, 8, 256, 'linear', False
        # MLP
        if activation not in ACTIVATIONS:
            raise ValueError(f"Invalid activation function: {activation}")
        if activation in ["swiglu", "geglu", "ffnglu", "ffngeglu", "ffnswiglu"]:
            act = ACTIVATIONS[activation](dim_hidden)  # 2048
        else:
            act = ACTIVATIONS[activation]()
        self.mlp = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),  # (512, 2048)
            nn.Dropout(dropout),  # 0.1
            act,
            nn.Linear(dim_hidden, dim_input),  # (2048, 512)
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(dim_input)  # 层归一化 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim_input).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim_input).
        """

        # Apply multi-head attention, followed by MLP and layer normalization with residual connection.
        x_ = self.attn(x)
        x_ = self.mlp(x_)

        return self.layer_norm(x_ + x)


class MoDInfiniTransformer(InfiniTransformer):
    """Mixture-of-Depths Infini-Transformer Layer."""

    def __init__(
        self,
        dim_input: int,  # 512
        dim_hidden: int,  # 2048
        dim_key: int,  # 64
        dim_value: int,  # 64
        num_heads: int,  # 8
        activation: str,  # geglu(x)=x * sigmoid(x)
        segment_len: int,  # 2048
        sampling_factor: int,  # 8
        update="linear",
        causal: bool = False,
        dropout: float = 0.0  # 0.1
    ):
        """Instantiate module.

        Args:
            dim_input (int): Input dimension.
            dim_hidden (int): Hidden dimension for the MLP.
            dim_key (int): Key dimension for the CompressiveMemory.
            dim_value (int): Value dimension for the CompressiveMemory.
            num_heads (int): Number of attention heads for the CompressiveMemory.
            activation (str): Activation function to use for the MLP. Must be a key in the ACTIVATIONS dictionary.
            segment_len (int): Segment length for the CompressiveMemory.
            sampling_factor (int): Reciprocal of the sampling rate for the Mixture-of-Depths mechanism.
            update (str, optional): Type of memory update rule to use for the CompressiveMemory ("linear" or "delta"). Defaults to "linear".
            causal (bool, optional): Whether to use causal attention masking for the CompressiveMemory. Defaults to False.
            dropout (float, optional): Dropout rate for the MLP. Defaults to 0.0.

        Raises:
            ValueError: Segment length not divisible by sampling factor.  可分割的
        """
        # Initialize ordinary InfiniTransformer, but with segment length reduced by sampling_factor
        super(MoDInfiniTransformer, self).__init__(
            dim_input=dim_input,  # 512
            dim_hidden=dim_hidden,  # 2048
            dim_key=dim_key,  # 64
            dim_value=dim_value,  # 64
            num_heads=num_heads,  # 8
            activation=activation,  # geglu = x * sigmoid(x)
            segment_len=math.ceil(segment_len / sampling_factor),  # 256
            update=update,  # 'linear'
            causal=causal,  # False
            dropout=dropout  # 0.1
        )

        # Record additional init arguments for forward pass
        self.segment_len = math.ceil(segment_len / sampling_factor)  # 向上取整 ceil(2048/8)=256
        self.full_segment_len = segment_len  # 2048
        self.sampling_factor = sampling_factor  # 8
        self.dim_input = dim_input  # 512

        # Projection for tensor of logits when sampling
        self.proj_sampling = nn.Linear(dim_input, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass wrapper -- used to check at inference time whether to handle each observation individually.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim_input).  # seq_len=4096, dim_input=512

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim_input).
            torch.Tensor: Token selection mask of shape (batch_size * seq_len, 1) or None.
            torch.Tensor: Predicted token selection scores of shape (batch_size * seq_len, 1) or None.
        """
        if self.train:
            return self.forward_(x)
        else:
            out = []
            for ix in range(x.size(0)):
                obs_out, _, _ = self.forward_(x[ix:ix+1])
                out.append(obs_out)
                
            return torch.cat(out, dim=0), None, None
                
    def forward_(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim_input).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim_input).
            torch.Tensor: Token selection mask of shape (batch_size * seq_len, 1).
            torch.Tensor: Predicted token selection scores of shape (batch_size * seq_len, 1) or None.
        """
        # Calculate number of total segments, samples
        batch_size, seq_len, _ = x.shape  # 随机张量x (batch_size, seq_len, dim_input) = (2; 4096; 512)
        num_segments, rem = divmod(seq_len, self.full_segment_len)  # 4096/2048 divmod 取商 和 余数
        num_segments += 1 if rem > 0 else 0

        # Initialize list of token sample masks
        sample_masks = []

        # Use linear embedding for sample scores
        sample_scores = self.proj_sampling(x).squeeze(-1)  # x -> (input_dim, 1) (2,4095, 1)-> squeeze (2, 4096), 移除tensor中最后一个dim=1的维度。

        # For each segment, sample the tokens with the highest scores
        for seg_num in range(num_segments):
            # Compute segment indices
            ix_lo = seg_num * self.full_segment_len  # 2048
            ix_hi = ix_lo + self.full_segment_len

            if self.train:
                # During training, take the top-k tokens by score
                # Argsort by sample scores to get indices of tokens to keep
                sort_ixs = torch.argsort(
                    sample_scores[:, ix_lo:ix_hi], dim=1, descending=True)  # 用于返回张量中元素在dim上按大小排序后的索引indices，descending=True降序, (2, 2048)

                # Convert token indices to a binary mask，创建掩码
                sample_mask_seg = torch.zeros_like(  # 全零张量，(2, 2048)
                    sample_scores[:, ix_lo:ix_hi], device=x.device)
                sample_mask_seg.scatter_(  # scatter_函数根据给定的索引，在指定维度dim=1上填充指定的value=1.0。
                    dim=1, index=sort_ixs[:, :self.segment_len], value=1.0)  # sort_ixs指定要填充的index，为降序排列的前segment_len=256个索引。
            else:
                # During inference, take the tokens with score greater than zero
                sample_mask_seg = (sample_scores[:, ix_lo:ix_hi] > 0.0).float()  # > 0.0创建一个bool tensor；float()将bool tensor转化为浮点型，true为1.0，falso为0

            sample_masks.append(sample_mask_seg)  # 获取掩码

        # Combine segment masks into a single mask
        sample_mask = torch.cat(sample_masks, dim=1).bool()  # dim=1拼接，(2, 4096)，但是其中selected mask true为512, (2, 512)

        # Apply multi-head attention to sample, followed by MLP
        # 通过sample_mask对tensor x进行过滤
        sample_shape = (batch_size, self.segment_len * num_segments, self.dim_input)  # (2, 256*2=512, 512); 262144
        # unsqueeze在sample_mask上添加一个-1维，(2,4096,1) -> 对sample_mask进行复制，将其在第三个维度复制self.dim_input次，(2,4096,512)
        # x(batch_size, seq_len, dim_input) = (2; 4096; 512) -》因为sampling mask sorted num为512
        x_ = x[sample_mask.unsqueeze(-1).repeat((1, 1, self.dim_input))].view(sample_shape)  # (2, 512, 512)
        x_ = self.attn(x_)  # (2,512,512)
        x_ = self.mlp(x_)  # (2,512,512)

        # Add result of attended tokens to the result (equivalent to making the result 
        # for non-attended tokens zero)
        '''
            x (batch_size, seq_len, dim_input) = (2; 4096; 512) 
            sample_mask, (2,4096)-> unsqueeze,(2, 4096, 1) -> repeat,(2,4096,512)
            x_, (2, 512, 512) -> view(-1), (524288,)
            对于tensor shape不匹配的情况，pytorch会进行broadcast操作，将较小的tensor沿着维度进行复制，使其形状与较大的tensor相匹配。
        '''
        # x_1_1 = sample_mask.unsqueeze(-1)  # (2, 4096, 1)
        # x_1_2 = sample_mask.unsqueeze(-1).repeat((1, 1, self.dim_input))  # (2,4096,512)
        x_1 = x[sample_mask.unsqueeze(-1).repeat((1, 1, self.dim_input))]  # sample_mask中true_num为512，所以能取出的mask_true_element为(524288，)
        x_2 = x_.view(-1)  # 524288
        x[sample_mask.unsqueeze(-1).repeat((1, 1, self.dim_input))] += x_2  # (2,4096,512)

        # Flatten sample scores and concatenation of top-k masks for auxillary training task
        sample_scores = sample_scores.view((-1, 1))  # 调整tensor shape为(8192,1)
        sample_mask = sample_mask.view((-1, 1)).float()  # 调整tensor shape为(8192,1)

        # layer_norm, 512
        return self.layer_norm(x), sample_mask, sample_scores


if __name__=='__main__':
# def demo_mod_infini_transformer():
    """
    Demonstrates the usage of the MoDInfiniTransformer class.
    """
    # Define the model parameters
    dim_input = 512
    dim_hidden = 2048
    dim_key = 64
    dim_value = 64
    num_heads = 8
    activation = "ffngeglu"
    segment_len = 2048
    sampling_factor = 8
    update = "linear"
    dropout = 0.1
    

    # Define batch dimensions
    seq_len = 4096
    batch_size = 2

    # Create the MoDInfiniTransformer layer
    layer = MoDInfiniTransformer(
        dim_input=dim_input,
        dim_hidden=dim_hidden,
        dim_key=dim_key,
        dim_value=dim_value,
        num_heads=num_heads,
        activation=activation,
        segment_len=segment_len,
        sampling_factor=sampling_factor,
        update=update,
        dropout=dropout
    )

    # Generate dummy batch
    x = torch.randn(batch_size, seq_len, dim_input)  # 随机张量，2; 4096; 512

    # Test outputs for the case where the net is training
    layer.train()
    x_att, sample_mask, sample_scores_pred = layer(x)

    # Test output for the case where the net is not training
    layer.eval()
    x_att, sample_mask, sample_scores_pred = layer(x)
