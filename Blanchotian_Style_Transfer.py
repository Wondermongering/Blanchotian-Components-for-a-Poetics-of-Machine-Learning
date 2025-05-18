import importlib.machinery
import importlib.util
import torch

attn_loader = importlib.machinery.SourceFileLoader('blanchot_attn', 'Blanchotian Attention Mechanism.py')
attn_spec = importlib.util.spec_from_loader('blanchot_attn', attn_loader)
attn_mod = importlib.util.module_from_spec(attn_spec)
attn_loader.exec_module(attn_mod)
BlanchotianAttention = attn_mod.BlanchotianAttention

xfm_loader = importlib.machinery.SourceFileLoader('blanchot_xfm', 'The Infinite Conversation.py')
xfm_spec = importlib.util.spec_from_loader('blanchot_xfm', xfm_loader)
xfm_mod = importlib.util.module_from_spec(xfm_spec)
xfm_loader.exec_module(xfm_mod)
BlanchotianTransformer = xfm_mod.BlanchotianTransformer

from BlanchotianEmbedding import BlanchotianEmbedding

def blanchotian_style_transfer(token_ids, vocab_size, style_vector):
    """Return transformed embeddings influenced by style_vector."""
    dim = style_vector.shape[-1]
    embed = BlanchotianEmbedding(vocab_size, dim)
    model = BlanchotianTransformer(dim=dim, depth=2, heads=4, mlp_dim=dim*2)

    tokens = torch.tensor(token_ids).unsqueeze(0)
    embeddings = embed(tokens)
    styled = embeddings + style_vector.view(1,1,-1)
    out = model(styled)
    return out.squeeze(0)

if __name__ == '__main__':
    vocab = 128
    text = [1,5,9,2,7]
    style_vec = torch.randn(32)
    result = blanchotian_style_transfer(text, vocab, style_vec)
    print(result)
