import torch
import torch.nn as nn
import torch.nn.functional as F

class ImaginationCell(nn.Module):
    """A cell inspired by Bachelard's interior/exterior dialectic."""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.attic_gate = nn.Linear(dim, hidden_dim)
        self.cellar_gate = nn.Linear(dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim * 2, hidden_dim)

    def forward(self, emb, hidden):
        attic = torch.tanh(self.attic_gate(emb))            # the heights of thought
        cellar = torch.tanh(self.cellar_gate(-emb))         # the depths of memory
        combined = torch.cat([attic, cellar], dim=-1)
        hidden = self.rnn(combined, hidden)
        return hidden

class BachelardianImaginationSpace(nn.Module):
    """Generative framework drawing on Gaston Bachelard's poetics of space."""
    def __init__(self, vocab_size, dim, hidden_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, dim)
        self.cell = ImaginationCell(dim, hidden_dim)
        self.to_logits = nn.Linear(hidden_dim, vocab_size)

    def init_hidden(self, batch_size, device=None):
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, tokens, hidden=None):
        # tokens: (B, T)
        B, T = tokens.shape
        device = tokens.device
        if hidden is None:
            hidden = self.init_hidden(B, device=device)
        logits_out = []
        for t in range(T):
            emb = self.embedding(tokens[:, t])
            hidden = self.cell(emb, hidden)
            logits = self.to_logits(hidden)
            logits_out.append(logits)
        return torch.stack(logits_out, dim=1), hidden

    @torch.no_grad()
    def generate(self, start_tokens, steps, temperature=1.0):
        # start_tokens: (B,)
        B = start_tokens.shape[0]
        device = start_tokens.device
        hidden = self.init_hidden(B, device=device)
        tokens = start_tokens
        outputs = []
        for _ in range(steps):
            logits, hidden = self.forward(tokens.unsqueeze(1), hidden)
            logits = logits[:, -1, :]
            probs = F.softmax(logits / temperature, dim=-1)
            tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            outputs.append(tokens)
        return torch.stack(outputs, dim=1)
