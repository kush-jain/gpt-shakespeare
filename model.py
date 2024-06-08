import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim=None):
        super(BigramModel, self).__init__()

        if embedding_dim is None:
            embedding_dim = vocab_size

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, idx, targets=None):

        # Idx should be Batch X Time dimesions
        logits = self.embeddings(idx) # this should return Batch X Time X Embedding (channel)

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens=1):
        """
        Generate new tokens based on the input idx
        """

        # IDX is in Batch X Time

        for _ in range(max_new_tokens):

            # Get the logits for the last token
            logits, loss = self(idx)

            # Focus only on last stuff
            logits = logits[:, -1, :]

            # Multiple ways of generating new stuff
            probs = F.softmax(logits, dim=1)

            # 1. Get the most likely token
            # idx_next = torch.max(probs, dim=1)
            # 2. Pick one from distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append the new token to the idx
            idx = torch.cat([idx, idx_next], dim=1)

        return idx
