from typing import Optional
import warnings

import torch
from torch import nn


class CompressiveMemory(nn.Module):
    """Implements the Compressive Transformer memory module."""

    def __init__(  # 512, 64, 64, 8, 256, 'linear', False
        self, 
        dim_input: int, 
        dim_key: int, 
        dim_value: int, 
        num_heads: int, 
        segment_len: int, 
        update: str = "linear",
        causal: bool = False
    ):
        """Initialize module.

        Args:
            dim_input (int): Input dimension.
            dim_key (int): Key dimension.
            dim_value (int): Value dimension.
            num_heads (int): Number of attention heads.
            segment_len (int): Segment length (must be a factor of the input sequence length).
            update (str, optional): Type of memory update rule to use ("linear" or "delta"). Defaults to "linear".
            causal (bool, optional): Whether to use causal attention masking. Defaults to False.
        """
        super(CompressiveMemory, self).__init__()

        # Record input parameters
        self.num_heads = num_heads
        self.segment_len = segment_len  # 256

        self.dim_input = dim_input  # 512
        self.dim_key = dim_key
        self.dim_value = dim_value

        self.update = update
        self.causal = causal

        # Projections for stacked SDP attention
        self.proj_k = nn.Linear(dim_input, num_heads * dim_key, bias=False)  # (512, 512)
        self.proj_v = nn.Linear(dim_input, num_heads * dim_value, bias=False)  # (512, 512)
        self.proj_q = nn.Linear(dim_input, num_heads * dim_key, bias=False)  # (512, 512)

        # Initialize betas for weighted average of dot-product and memory-based attention
        self.betas = nn.Parameter(torch.randn(1, num_heads, 1, dim_value))  # random tensor, shape=(1, 8, 1, 64)

        # Projection for output
        self.proj_out = nn.Linear(num_heads * dim_value, dim_input, bias=False)  # (512, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # tensor x_, (2, 512, 512)
        """
        Applies Scaled Dot-Product Attention to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim_input).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim_input).
        """
        batch_size, seq_len, _ = x.shape  # batch_size=2, seq_len=512

        num_segments, rem = divmod(seq_len, self.segment_len)  # 求商取余，512/256=2
        num_segments += 1 if rem > 0 else 0

        out = []

        # Initialize mem and normalization, 初始化memory 和 归一化项z_s
        # !!! Initialization was never specified in the paper, so this is an educated guess
        mem = torch.zeros(1, self.num_heads, self.dim_key, self.dim_value)  # (1, 8, 64, 64)
        z = torch.zeros(batch_size, self.num_heads, self.dim_key, 1)  # (2, 8, 64, 1)
        
        # Project the input tensor to get the key, value, and query tensors
        k_full = self.proj_k(x).unsqueeze(1).view(  # 扩展dim=1，(2,1,512,512) -> view, (2, 8, 512, 64)
            (batch_size, self.num_heads, x.size(1), self.dim_key))
        v_full = self.proj_v(x).unsqueeze(1).view(
            (batch_size, self.num_heads, x.size(1), self.dim_value))
        q_full = self.proj_q(x).unsqueeze(1).view(
            (batch_size, self.num_heads, x.size(1), self.dim_key))
        
        for ix in range(num_segments):
            ix_lo = ix * self.segment_len  # 256
            ix_hi = min(ix_lo + self.segment_len, x.size(1))
            seg_len = ix_hi - ix_lo  # 256

            # Extract segment from key, value and query tensors
            k = k_full[:, :, ix_lo:ix_hi, :]  # (2, 8, 0:256, 64)
            v = v_full[:, :, ix_lo:ix_hi, :]
            q = q_full[:, :, ix_lo:ix_hi, :]
            
            # Pre-calculate sigma(q) for updating memory and calculating attention
            # shape: (batch_size, num_heads, segment_len, dim_key)
            sigma_q = (nn.functional.elu(q) + 1.0)  # For memory retrieval

            # Apply normalization term update. keepdim保持求和后的tensor维度和原始维度相同，只是对应维度变为1，(2,1,0:256,64) -> transpose, (2,64,0:256,1)
            z = z + (nn.functional.elu(k) + 1.0).sum(dim=-2, keepdim=True).transpose(-2, -1)  # For memory update

            # Apply SDP attention, Scaled Dot-product attention, 点积attention
            scores = q @ k.transpose(-2, -1) / self.dim_key ** 0.5  # (2, 8, 0:256, 64) * (2, 8, 64, 0:256) / 放缩因子 8 = (2,8,256,256)

            # If causal mask specified 指定因果关系, calculate and apply it
            if self.causal:
                # 创建下三角形矩阵作为causal mask，确保每个位置只能看到之前的位置，避免未来信息的泄露，矩阵对角线及以上的元素为true，其余元素为false，seg_len=(256, 256)
                mask = torch.tril(torch.ones((seg_len, seg_len), dtype=torch.bool), diagonal=0)
                # 对掩码进行维度扩展unsqueeze，(1,1,256,256); -> repeat, (2,8,256,256)
                mask = mask.unsqueeze(0).unsqueeze(0).repeat((batch_size, self.num_heads, 1, 1))
                scores.masked_fill_(torch.logical_not(mask), float('-inf'))  # 将mask位置替换为'-inf'

            # Calculate SDP attention
            att_dot = nn.functional.softmax(scores, dim=-1) @ v  # (2,8,256,256) * (2,8,256,64) = (2,8,256,64)

            # Calculate normalized linear attention, 计算记忆attention
            # shape: (batch_size, num_heads, segment_len, dim_value)
            att_mem_1 = (sigma_q @ mem)  # (2,8,256,64) * (1, 8, 64, 64) = (2,8,256,64)
            att_mem_2 = (sigma_q @ z)  # (2,8,256,64) * (2, 8, 64, 1) = (2,8,256,1)
            att_mem = att_mem_1 / att_mem_2  # (2,8,256,64)

            # Apply memory update
            sigma_k = nn.functional.elu(k) + 1.0  # (2,8,256,64)
            if self.update == "linear":
                mem = mem + sigma_k.transpose(-2, -1) @ v  # (1,8,64,64) + (2,8,64,256) * (2,8,256,64) = (2,8,64,64)
            elif self.update == "delta":
                mem = mem + \
                    sigma_k.transpose(-2, -1) @ (v - (sigma_k @ mem) / (sigma_k @ z))

            # Calculate weighted average of dot-product and memory-based attention
            att = nn.functional.sigmoid(
                self.betas) * att_mem + (1 - nn.functional.sigmoid(self.betas)) * att_dot  # (2,8,256,64)
            att = att.view((batch_size, seg_len,
                        self.num_heads * self.dim_value))  # (2,256,512)

            # Append output to buffer
            out.append(self.proj_out(att))

        # Return concatenated full sequence from buffer
        out = torch.concat(out, dim=1)  # (2,512,512)

        return out


def test_compressive_memory(
    short_seq_len: bool = False, 
    even_seq_len: bool = True, 
    causal_masking: bool = False, 
    update: str = "linear"
) -> None:
    # Set example module parameters
    dim_input = 512
    dim_key = 64
    dim_value = 64
    num_heads = 8
    segment_len = 32
    causal = causal_masking
    
    # Set dummy input dimensions
    batch_size = 4
    
    # Handle sequence length based on test case
    if short_seq_len:
        seq_len = 16
    else:
        if even_seq_len:
            seq_len = 128
        else:
            seq_len = 144

    # Initialize module
    model = CompressiveMemory(
        dim_input, dim_key, dim_value, num_heads, segment_len, update, causal)

    # Generate random input
    batch = torch.randn(batch_size, seq_len, dim_input)

    # Apply the CompressiveMemory module
    model(batch)


if __name__ == "__main__":
    # Test all cases with short sequence lengths
    print("Testing with short sequence lengths:")
    
    short_seq_len = True
    # In this case even_seq_len doesn't matter -- arbitrarily setting it to True
    even_seq_len = True
    
    for causal_masking in [True, False]:
        for update in ["linear", "delta"]:
            print(f"  Testing with causal_masking={causal_masking} and update={update}")
            test_compressive_memory(
                short_seq_len=short_seq_len,
                even_seq_len=even_seq_len,
                causal_masking=causal_masking,
                update=update
            )
            
    # Test all cases with short sequence lengths
    print("Testing with non-short sequence lengths:")
    
    short_seq_len = False
    
    for even_seq_len in [True, False]:
        for causal_masking in [True, False]:
            for update in ["linear", "delta"]:
                print(f"  Testing with even_seq_len={even_seq_len}, causal_masking={causal_masking} and update={update}")
                test_compressive_memory(
                    short_seq_len=short_seq_len,
                    even_seq_len=even_seq_len,
                    causal_masking=causal_masking,
                    update=update
                )