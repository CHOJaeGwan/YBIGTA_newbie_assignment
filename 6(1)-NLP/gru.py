import torch
from torch import nn, Tensor


class GRUCell(nn.Module):
    """
    Gated Recurrent Unit (GRU) Cell.
    """
    def __init__(self, input_size: int, hidden_size: int) -> None:
        """
        Initialize the GRU cell.
        Args:
            input_size (int): _size of the input features
            hidden_size (int): _size of the hidden state
        """
        super().__init__()
        self.hidden_size = hidden_size
       
        self.W_z = nn.Linear(input_size, hidden_size, bias=False)
        self.U_z = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_r = nn.Linear(input_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_h = nn.Linear(input_size, hidden_size, bias=False)
        self.U_h = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        """
        Forward pass for the GRU cell.
        Args:
            x (Tensor): _input tensor of shape (batch_size, input_size)
            h (Tensor): _previous hidden state of shape (batch_size, hidden_size)

        Returns:
            Tensor: _next hidden state of shape (batch_size, hidden_size)
        """
        # 구현하세요!
        z = torch.sigmoid(self.W_z(x) + self.U_z(h))
        r = torch.sigmoid(self.W_r(x) + self.U_r(h))
        h_tilde = torch.tanh(self.W_h(x) + self.U_h(r * h))
        h_next = (1 - z) * h + z * h_tilde
        return h_next


class GRU(nn.Module):
    """
    Gated Recurrent Unit (GRU) model."""
    def __init__(self, input_size: int, hidden_size: int) -> None:
        """
        Initialize the GRU model.
        Args:
            input_size (int): _size of the input features
            hidden_size (int): _size of the hidden state
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = GRUCell(input_size, hidden_size)
        # 구현하세요!

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Forward pass for the GRU model.

        Args:
            inputs (Tensor): _input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Tensor: _output tensor of shape (batch_size, seq_len, hidden_size)
        """
        # 구현하세요!
        batch_size, seq_len, _ = inputs.size()
        h = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        outputs = []
        for t in range(seq_len):
            x_t = inputs[:, t, :]
            h = self.cell(x_t, h)
            outputs.append(h.unsqueeze(1))

        return torch.cat(outputs, dim=1) 