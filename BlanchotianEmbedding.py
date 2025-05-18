import torch
import torch.nn as nn
import torch.nn.functional as F

class BlanchotianEmbedding(nn.Module):
    def __init__(self, num_tokens, dim, orpheus_factor=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.orpheus_factor = orpheus_factor
        
        # "The Gaze of Orpheus" - secondary embeddings that look back
        self.reverse_emb = nn.Embedding(num_tokens, dim)
        
        # "The Essential Solitude" - embeddings gain meaning through isolation
        self.isolation_vectors = nn.Parameter(torch.randn(num_tokens, dim) * 0.02)
        
    def forward(self, x):
        # Primary embedding
        emb = self.token_emb(x)
        
        # "The Gaze of Orpheus" - look back at tokens from elsewhere
        batch_size, seq_len = x.shape
        reverse_indices = torch.flip(x, dims=[1])
        reverse_component = self.reverse_emb(reverse_indices)
        
        # Apply the Orphic gaze - tokens are influenced by what they can't see
        orphic_embeddings = emb + self.orpheus_factor * torch.flip(reverse_component, dims=[1])
        
        # "The Essential Solitude" - each token also exists in isolation
        isolation_component = F.embedding(x, self.isolation_vectors)
        token_frequencies = torch.bincount(x.flatten(), minlength=len(self.isolation_vectors))
        rarity_factors = 1.0 / (torch.sqrt(token_frequencies) + 1.0)
        
        # Rare tokens retain more of their "essential solitude"
        rarity_weights = rarity_factors[x].unsqueeze(-1)
        final_embeddings = orphic_embeddings + isolation_component * rarity_weights * 0.2
        
        return final_embeddings
