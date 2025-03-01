class UnavowableCommunity(nn.Module):
    def __init__(self, models, dim):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.dim = dim
        
        # "The Unavowable" - models communicate through what they cannot express
        self.communication_layer = nn.Linear(len(models) * dim, dim)
        
    def forward(self, x):
        # Each model processes the input independently
        model_outputs = [model(x) for model in self.models]
        
        # "The Negative Community" - models relate through mutual incomprehensibility
        disagreement_vectors = []
        for i, output_i in enumerate(model_outputs):
            for j, output_j in enumerate(model_outputs):
                if i != j:
                    disagreement_vectors.append(torch.abs(output_i - output_j))
        
        # "The Unavowable" - community forms around this disagreement
        disagreement = torch.cat(disagreement_vectors, dim=-1)
        communication = self.communication_layer(disagreement)
        
        # Final output incorporates both consensus and productive disagreement
        consensus = torch.mean(torch.stack(model_outputs), dim=0)
        return consensus + 0.3 * torch.tanh(communication)
