from torch import nn, Tensor, LongTensor
from gru import GRU

class MyGRULanguageModel(nn.Module):
    """
    A simple GRU-based language model for text classification.
    """
    def __init__(
        self,
        d_model: int,
        hidden_size: int,
        num_classes: int,
        embeddings: int | Tensor,
        padding_idx: int = 0 
    ) -> None:
        """
        Initialize the GRU language model.

        Args:
            d_model (int): _size of the input features (embedding dimension)
            hidden_size (int): _size of the hidden state
            num_classes (int): _number of output classes for classification
            embeddings (int | Tensor): number of embeddings or pre-trained embeddings
            padding_idx (int, optional): _index of the padding token in the embeddings. Defaults to 0.
        """
        super().__init__()
        # Embedding layer: pad tokens will have zero vectors and won't be updated
        if isinstance(embeddings, int):
            self.embeddings = nn.Embedding(
                num_embeddings=embeddings,
                embedding_dim=d_model,
                padding_idx=padding_idx
            )
        else:
            self.embeddings = nn.Embedding.from_pretrained(
                embeddings,
                freeze=False,          # allow fine-tuning
                padding_idx=padding_idx
            )

        self.gru = GRU(d_model, hidden_size)
        self.dropout = nn.Dropout(p=0.5)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids: LongTensor) -> Tensor:
        """
        Forward pass for the GRU language model.
        Args:
            input_ids (LongTensor): _input tensor of shape (batch_size, seq_len)
        Returns:
            Tensor: _output tensor of shape (batch_size, num_classes)
        """
        x = self.embeddings(input_ids)        # (batch_size, seq_len, d_model)
        hs = self.gru(x)                      # (batch_size, seq_len, hidden_size)
        last_hidden = hs[:, -1, :]            # (batch_size, hidden_size)
        last_hidden = self.dropout(last_hidden)
        logits = self.head(last_hidden)       # (batch_size, num_classes)
        return logits
