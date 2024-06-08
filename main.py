
import torch
import torch.nn as nn
from torch.nn import functional as F

# ----------------------

## Hyper Parameter settings

block_size = 128 # Time dimension
batch_size = 32 # This is parallelism
embedding_dim = 64
learning_rate = 3e-4
max_iters = 5000

heads = 8
transformer_blocks = 3
dropout = 0.2

eval_iter = 1000

# ----------------------

## Other settings

_input_file_path = '/Users/UI0627/Projects/genai/input.txt'
model_path = '/Users/UI0627/Projects/genai/model.pth'

# ----------------------

## Helper functions

def read_file(_file):
    try:
        with open(_file, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return None


def get_vocab(contents):
    uniq_chars = set(contents)
    vocab = sorted(list(uniq_chars))
    return vocab


def construct_vocab_map(vocab):
    # Alternatives: Open-source variants. For example, GPT uses tik-token library
    encoder_map = { ch: i for i, ch in enumerate(vocab) }
    decoder_map = { i: ch for i, ch in enumerate(vocab) }
    return encoder_map, decoder_map


def encoder(str):
    return [ encoder_map[ch] for ch in str ]

def decoder(idx_arr):
    return "".join([ decoder_map[idx] for idx in idx_arr ])

# ----------------------

def split_data(data, split_ratio=0.9):
    split_point = int(len(data) * split_ratio)
    train_data = data[:split_point]
    test_data = data[split_point:]
    return train_data, test_data


def get_batches_v2(data):
    random_idx = torch.randint(len(data) - block_size, size=(batch_size,))
    x = torch.stack( [data[ix:ix+block_size] for ix in random_idx] )
    y = torch.stack( [data[ix+1:ix+block_size+1] for ix in random_idx] )
    return x, y

def get_batch(split):
    data = train_data if split == "train" else test_data
    return get_batches_v2(data)


# ----------------------


class BigramModel(nn.Module):

    def __init__(self):
        super(BigramModel, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embeddings = nn.Embedding(block_size, embedding_dim)
        self.transformer_blocks = nn.Sequential(
            *[BlockModel() for _ in range(transformer_blocks)] +
            [
                nn.LayerNorm(embedding_dim)
            ]
        )
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx, targets=None):

        B, T = idx.shape

        # Idx should be Batch X Time dimesions
        tok_embedding = self.embeddings(idx) # this should return Batch X Time X Embedding (channel)
        pos_embedding = self.positional_embeddings(torch.arange(T))
        x = tok_embedding + pos_embedding
        x = self.transformer_blocks(x)
        logits = self.lm_head(x)  # this should return Batch X Time X Vocab

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

            # Crop the sequence to the block size
            idx_c = idx[:, -block_size:]

            # Get the logits for the last token
            logits, loss = self(idx_c)

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


class BlockModel(nn.Module):
    """
    Abstracts out a block of the transformer containing self-attention and feed-forward
    Self-Attention is communication,
    while feed-forward is computation
    """

    def __init__(self):
        super(BlockModel, self).__init__()

        self.sa_head = MultiHeadModel(num_heads=heads)
        self.ff_head = FeedForwardModel()

        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # Creating skip connections by additive factors
        # In original paper, layer norms were applied after, but now they are done prior
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ff_head(self.ln2(x))
        return x


class SingleHeadModel(nn.Module):
    """
    Compute self-attention using a single head
    """

    def __init__(self, head_size=None):
        super(SingleHeadModel, self).__init__()

        self.head_size = head_size or embedding_dim

        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        keys = self.key(x)          # B x T x H
        queries = self.query(x)     # B x T x H
        values = self.value(x)      # B x T x H

        # Compute the attention
        logits = torch.bmm(queries, keys.transpose(1, 2)) / (self.head_size ** 0.5)      # B x T x T
        logits = logits.masked_fill(self.tril[:T, :T] == 0, float("-inf"))          # B x T x T
        probs = F.softmax(logits, dim=-1)                                           # B x T x T
        probs = self.dropout(probs)

        # Compute the output
        out = torch.bmm(probs, values)                                              # B x T x H

        return out


class MultiHeadModel(nn.Module):
    """
    Run multiple heads in parallel
    """

    def __init__(self, num_heads=8):
        super(MultiHeadModel, self).__init__()

        # Embedding_dimension = num_heads * head_size
        head_size = embedding_dim // num_heads
        self.heads = nn.ModuleList([SingleHeadModel(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.projection(out))


class FeedForwardModel(nn.Module):
    """
    Feed forward model
    """

    def __init__(self):
        super(FeedForwardModel, self).__init__()

        # Multiplied by 4 as the original paper suggests that inner layer be 4 times the outer layer
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4*embedding_dim),
            nn.ReLU(),
            nn.Linear(4*embedding_dim, embedding_dim),       # This is residual connection
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


def train():

    for steps in range(max_iters):
        xb, yb = get_batch("train")

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if steps % eval_iter == 0:
            print(loss.item(), steps)

    return loss


if __name__ == "__main__":

    contents = read_file(_input_file_path)

    vocab = get_vocab(contents)
    vocab_size = len(vocab)
    encoder_map, decoder_map = construct_vocab_map(vocab)

    data = torch.tensor(encoder(contents))

    train_data, test_data = split_data(data)

    xb, yb = get_batch("train")

    # Train Code
    # model = BigramModel()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # loss = train()
    # print(loss.item())

    # torch.save(model.state_dict(), model_path)

    # Direct Load
    model = BigramModel()
    model.load_state_dict(torch.load(model_path))

    # Eval Code
    test_d = torch.zeros( (1, 1), dtype=torch.long)
    print(decoder(model.generate(test_d, 500)[0].tolist()))
