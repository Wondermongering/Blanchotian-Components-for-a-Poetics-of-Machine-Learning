import importlib.machinery
import importlib.util

module_name = "recursive_orphic_tokenizer"
file_path = "Recursive Orphic Tokenizer.py"
loader = importlib.machinery.SourceFileLoader(module_name, file_path)
spec = importlib.util.spec_from_loader(module_name, loader)
tokenizer_module = importlib.util.module_from_spec(spec)
loader.exec_module(tokenizer_module)

RecursiveOrphicTokenizer = tokenizer_module.RecursiveOrphicTokenizer

def test_absence_set_includes_unseen_token():
    texts = ["a b c", "a b d", "e b c"]
    tok = RecursiveOrphicTokenizer(max_context=1, top_absent=2)
    tok.train(texts)
    enc = tok.encode("a b c")
    token_a = tok.vocab["a"]
    assert token_a in enc[2]["absence"]

def test_encode_and_decode_roundtrip():
    texts = ["x y z"]
    tok = RecursiveOrphicTokenizer(max_context=1)
    tok.train(texts)
    enc = tok.encode("x y z")
    out = tok.decode(enc)
    assert out == "x y z"
