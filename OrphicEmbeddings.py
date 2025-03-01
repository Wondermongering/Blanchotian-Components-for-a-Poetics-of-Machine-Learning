import torch
import torch.nn as nn
import torch.nn.functional as F

class OrphicEmbeddings(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_window, alpha=0.5, num_negative_samples=5):
        super(OrphicEmbeddings, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_window = context_window
        self.alpha = alpha
        self.num_negative_samples = num_negative_samples

        self.forward_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.reverse_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.isolation_vectors = nn.Embedding(vocab_size, embedding_dim)

        # Initialize embeddings (optional, could use pre-trained)
        nn.init.xavier_uniform_(self.forward_embeddings.weight)
        nn.init.xavier_uniform_(self.reverse_embeddings.weight)
        nn.init.normal_(self.isolation_vectors.weight, mean=0, std=0.02) # Small initial values


    def forward(self, target_tokens, context_tokens):
        # target_tokens: (batch_size,)
        # context_tokens: (batch_size, 2 * context_window)

        batch_size = target_tokens.shape[0]

        # 1. Forward and Reversed Contextual Embeddings
        forward_embeds = self.forward_embeddings(target_tokens)  # (batch_size, embedding_dim)
        reverse_embeds = self.reverse_embeddings(target_tokens) # (batch_size, embedding_dim)
        contextual_embeds = self.alpha * forward_embeds + (1 - self.alpha) * reverse_embeds

        # 2. Isolation Vectors (Scaling based on frequency - needs to be pre-calculated)
        # Assuming we have a pre-calculated tensor 'token_frequencies' of shape (vocab_size,)
        token_frequencies = torch.tensor([1000, 10, 500, ...], dtype=torch.float) # Example Frequencies
        scaling_factors = 1.0 / (1.0 + torch.log(token_frequencies + 1e-6)) #Adding a small number for stability
        scaled_isolation_vectors = self.isolation_vectors(target_tokens) * scaling_factors[target_tokens].unsqueeze(-1)

        # 3. Orphic Embedding
        orphic_embeds = contextual_embeds + scaled_isolation_vectors

        # 4. Calculate Loss (Skip-gram with Negative Sampling and Absence Augmentation)
        context_embeds_positive = self.forward_embeddings(context_tokens) # Use forward embeddings for context
        positive_scores = torch.bmm(orphic_embeds.unsqueeze(1), context_embeds_positive.transpose(1, 2)).squeeze(1) # (batch_size, 2*window)
        positive_loss = -torch.log(torch.sigmoid(positive_scores) + 1e-6).mean()

        # Negative Sampling and Absence Augmentation
        negative_loss = 0
        for i in range(batch_size):
            target = target_tokens[i].item()
            context = context_tokens[i].tolist()

            # Create potential context (all words in vocab)
            potential_context = list(range(self.vocab_size))

            # Create absence set
            absence_set = [w for w in potential_context if w not in context and w != target]

            # Sample absent tokens
            if len(absence_set) > self.num_negative_samples:
                negative_samples = torch.tensor(random.sample(absence_set, self.num_negative_samples), dtype=torch.long)
            elif len(absence_set) > 0:
                 negative_samples = torch.tensor(absence_set, dtype=torch.long)
            else:
                # Handle case where absence_set might be small or empty.  Sample from the whole vocab instead
                negative_samples = torch.randint(0, self.vocab_size, (self.num_negative_samples,), dtype=torch.long)

            negative_embeds = self.forward_embeddings(negative_samples) #Use forward embeddings
            negative_scores = torch.matmul(orphic_embeds[i], negative_embeds.transpose(0, 1)) # (num_negative_samples,)
            negative_loss -= torch.log(torch.sigmoid(-negative_scores) + 1e-6).mean()

        total_loss = positive_loss + negative_loss
        return total_loss
