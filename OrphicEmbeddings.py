import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class OrphicEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_window, alpha=0.5,
                 isolation_factor=0.3, absence_factor=0.1, use_explicit_absence=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_window = context_window
        self.alpha = alpha
        self.isolation_factor = isolation_factor
        self.absence_factor = absence_factor
        self.use_explicit_absence = use_explicit_absence

        self.forward_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.reverse_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.isolation_vectors = nn.Parameter(torch.randn(vocab_size, embed_dim) * 0.02)
        self.absence_proj = nn.Linear(embed_dim, embed_dim, bias=False)  # For learned absence

        self.register_buffer('token_counts', torch.zeros(vocab_size))

    def forward(self, target_tokens):
        # target_tokens: (batch_size,)  or (batch_size, seq_len)

        batch_size = target_tokens.shape[0]

        # Update token frequencies (only if 1D input)
        if target_tokens.ndim == 1:
            counts = torch.bincount(target_tokens.flatten(), minlength=len(self.token_counts))
            self.token_counts += counts

        # Calculate rarity weights
        frequencies = self.token_counts / (self.token_counts.sum() + 1e-6)
        rarity = torch.sqrt(1.0 / (frequencies + 1e-6))

        # 1. Forward and Reversed Contextual Embeddings
        forward_embeds = self.forward_embeddings(target_tokens)
        reverse_embeds = self.reverse_embeddings(target_tokens)
        # Handle 2D input for sequences
        if target_tokens.ndim == 2:
            reverse_embeds = torch.flip(reverse_embeds, dims=[1])

        contextual_embeds = self.alpha * forward_embeds + (1 - self.alpha) * reverse_embeds

        # 2. Isolation Vectors
        isolation = self.isolation_vectors[target_tokens]
        isolation_strength = rarity[target_tokens].unsqueeze(-1) * self.isolation_factor
        isolated_emb = contextual_embeds + isolation * isolation_strength

        # 3. Absence (Optional)
        if self.use_explicit_absence:
             #For explicit absence, we don't transform in the forward, it is done during training
            final_emb = isolated_emb
        else: #Learned absence
            absence = self.absence_proj(isolated_emb)
            presence_mask = torch.sigmoid(rarity[target_tokens].unsqueeze(-1) * 2.0) # Or some other scaling
            final_emb = (isolated_emb * presence_mask) - (absence * (1 - presence_mask) * self.absence_factor)

        return final_emb

    def get_context_tokens(self, input_sequence, target_index):
      """Helper function to get context tokens for training."""
      window = self.context_window
      start = max(0, target_index - window)
      end = min(len(input_sequence), target_index + window + 1)
      context = [input_sequence[i] for i in range(start,end) if i != target_index]
      # Pad if necessary
      padding_needed = 2 * window - len(context)
      if padding_needed > 0:
            context.extend([0] * padding_needed) # Pad with 0 (assuming 0 is a padding token)
      return context
    
    def get_absence_set(self, input_sequence, target_index):
        """Helper function to get the absence set for a token."""
        context_tokens = self.get_context_tokens(input_sequence, target_index)
        potential_context = list(range(self.vocab_size))
        absence_set = [w for w in potential_context if w not in context_tokens and w != input_sequence[target_index]]
        return absence_set


# Example Training Loop (using explicit absence)
def train_orphic_embeddings(model, data_loader, epochs, optimizer, num_negative_samples):
    for epoch in range(epochs):
        total_loss = 0
        for batch in data_loader: # Assuming a data loader that provides sequences
            for sequence in batch:  # Iterate through sequences in the batch
                sequence = sequence.tolist() # Convert to list for easier indexing
                for i in range(len(sequence)):
                    target_token = torch.tensor([sequence[i]], dtype=torch.long)
                    context_tokens = torch.tensor(model.get_context_tokens(sequence, i), dtype=torch.long)

                    #Get Orphic embedding
                    orphic_embed = model(target_token)
                    context_embeds_positive = model.forward_embeddings(context_tokens)
                    positive_scores = torch.matmul(orphic_embed, context_embeds_positive.transpose(0,1))
                    positive_loss = -torch.log(torch.sigmoid(positive_scores) + 1e-6).mean()

                    # Negative Sampling from Absence Set
                    absence_set = model.get_absence_set(sequence,i)
                    if len(absence_set) > num_negative_samples:
                        negative_samples = torch.tensor(random.sample(absence_set, num_negative_samples), dtype=torch.long)
                    elif len(absence_set) > 0:
                        negative_samples = torch.tensor(absence_set, dtype=torch.long)
                    else:
                        negative_samples = torch.randint(0, model.vocab_size, (num_negative_samples,), dtype=torch.long)
                    negative_embeds = model.forward_embeddings(negative_samples)
                    negative_scores = torch.matmul(orphic_embed, negative_embeds.transpose(0, 1))
                    negative_loss = -torch.log(torch.sigmoid(-negative_scores) + 1e-6).mean()


                    loss = positive_loss + negative_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss}")

if __name__ == "__main__":
    # Example Usage
    vocab_size = 10000  # Example
    embedding_dim = 128
    context_window = 5
    model = OrphicEmbedding(vocab_size, embedding_dim, context_window)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create a dummy data loader (replace with your actual data)
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_sequences, seq_len):
            self.num_sequences = num_sequences
            self.seq_len = seq_len

        def __len__(self):
            return self.num_sequences

        def __getitem__(self, idx):
            return torch.randint(0, vocab_size, (self.seq_len,))  # Random sequence

    dataset = DummyDataset(100, 20)  # 100 sequences of length 20
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32)

    # Train
    train_orphic_embeddings(
        model,
        data_loader,
        epochs=10,
        optimizer=optimizer,
        num_negative_samples=5,
    )
    print("Finished Training Orphic Embeddings")
