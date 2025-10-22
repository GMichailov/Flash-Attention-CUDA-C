import torch
import torch.nn.functional as F

def multi_head_attention(Q, K, V, num_heads):
    """
    Compute Multi-Head Attention for Q, K, V.
    Q, K, V: tensors of shape (batch, seq_len, d_model)
    num_heads: number of attention heads
    """
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads  # dimension per head

    # Reshape for multi-heads
    Q = Q.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
    K = K.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
    V = V.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)

    # Scaled dot-product attention
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, V)

    # Concatenate heads
    output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
    return output, attn


if __name__ == "__main__":
    # Example setup
    batch_size = 1
    seq_len = 4
    d_model = 8
    num_heads = 2

    # Initialize Q, K, V with all ones
    Q = torch.ones((batch_size, seq_len, d_model))
    K = torch.ones((batch_size, seq_len, d_model))
    V = torch.ones((batch_size, seq_len, d_model))

    output, attn = multi_head_attention(Q, K, V, num_heads)

    print("Attention Weights:\n", attn)
    print("Output:\n", output)