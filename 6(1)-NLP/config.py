from typing import Literal
import torch

device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

d_model = 256

# Word2Vec
window_size = 7
method: Literal["cbow", "skipgram"] = "skipgram"
lr_word2vec = 5e-03
num_epochs_word2vec = 20

# GRU
hidden_size = 256
num_classes = 4
lr = 5e-03
num_epochs = 60
batch_size = 32