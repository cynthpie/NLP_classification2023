import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class FFNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_classes):
        # embedding_dim: embedding vector dim
        # hidden_dim: nb of nerons in hidden layer
        # num_class: number out predicted outcome
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        sent_lens = x.ne(0).sum(dim=1, keepdims=True)
        averged = embedded.sum(dim=1)/sent_lens
        out = self.fc1(averged)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

