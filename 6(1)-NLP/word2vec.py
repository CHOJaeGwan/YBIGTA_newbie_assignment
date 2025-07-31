import torch
from torch import nn, Tensor
from torch.optim import Adam

from transformers import PreTrainedTokenizer
from tqdm import tqdm  # type: ignore[import]
from typing import Literal


class Word2Vec(nn.Module):
    """
    Word2Vec model for training word embeddings."""
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        window_size: int,
        method: Literal["cbow", "skipgram"]
    ) -> None:
        """
        Initialize the Word2Vec model.

        Args:
            vocab_size (int): _size of the vocabulary
            d_model (int): _size of the word embeddings
            window_size (int): _size of the context window
            method (Literal[&quot;cbow&quot;, &quot;skipgram&quot;]): _which training method to use
        """
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.weight = nn.Linear(d_model, vocab_size, bias=False)
        self.window_size = window_size
        self.method = method

    def embeddings_weight(self) -> Tensor:
        """
        Get the embeddings weight.

        Returns:
            Tensor: _embeddings weight tensor of shape (vocab_size, d_model)
        """
        return self.embeddings.weight.detach()

    def fit(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        lr: float,
        num_epochs: int
    ) -> None:
        """
        Train the Word2Vec model.

        Args:
            corpus (list[str]): _list of sentences in the corpus_
            tokenizer (PreTrainedTokenizer): _tokenizer to convert sentences to token IDs
            lr (float): _learning rate for the optimizer
            num_epochs (int): _number of training epochs
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=lr)
        token_ids = tokenizer(corpus, padding=False, truncation=True)["input_ids"]
        device = self.embeddings.weight.device

        for epoch in range(num_epochs):
            total_loss = torch.tensor(0.0, device=device)
            for sent in tqdm(token_ids, desc=f"Epoch {epoch+1}"):
                tokens = [t for t in sent if t != tokenizer.pad_token_id]
                if len(tokens) < 2 * self.window_size + 1:
                    continue

                if self.method == "cbow":
                    loss = self._train_cbow(tokens, criterion)
                elif self.method == "skipgram":
                    loss = self._train_skipgram(tokens, criterion)
                else:
                    raise ValueError(f"Unknown method: {self.method}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}: loss = {total_loss:.4f}")

    def _train_cbow(
        self,
        tokens: list[int],
        criterion: nn.Module
    ) -> torch.Tensor:
        """
        Train the CBOW model.

        Args:
            tokens (list[int]): _list of token IDs in the sentence
            criterion (nn.Module): _loss function to compute the loss

        Returns:
            torch.Tensor: _loss tensor
        """
        device = self.embeddings.weight.device
        loss = torch.tensor(0.0, device=device)

        for center in range(self.window_size, len(tokens) - self.window_size):
            context = tokens[center - self.window_size:center] + tokens[center + 1:center + self.window_size + 1]
            target = tokens[center]

            context_tensor = torch.tensor(context, dtype=torch.long, device=device)
            target_tensor = torch.tensor([target], dtype=torch.long, device=device)

            context_embeddings = self.embeddings(context_tensor)
            context_mean = context_embeddings.mean(dim=0, keepdim=True)

            logits = self.weight(context_mean)
            loss += criterion(logits, target_tensor)
        return loss

    def _train_skipgram(
        self,
        tokens: list[int],
        criterion: nn.Module
    ) -> torch.Tensor:
        """
        Train the Skip-gram model.

        Args:
            tokens (list[int]): _list of token IDs in the sentence
            criterion (nn.Module): _loss function to compute the loss

        Returns:
            torch.Tensor: _loss tensor
        """
        device = self.embeddings.weight.device
        loss = torch.tensor(0.0, device=device)

        for center in range(self.window_size, len(tokens) - self.window_size):
            target = tokens[center]
            context_indices = list(range(center - self.window_size, center)) + \
                              list(range(center + 1, center + self.window_size + 1))

            context_tensor = torch.tensor(context_indices, dtype=torch.long, device=device)
            target_tensor = torch.tensor([target] * len(context_indices), dtype=torch.long, device=device)

            context_embeddings = self.embeddings(context_tensor)
            logits = self.weight(context_embeddings)
            loss += criterion(logits, target_tensor)
        return loss
