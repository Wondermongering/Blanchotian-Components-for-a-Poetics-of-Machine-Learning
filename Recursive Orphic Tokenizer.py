import collections
from typing import Iterable, List, Dict, Tuple

class RecursiveOrphicTokenizer:
    """Tokenizer that encodes presence and conspicuous absence.

    Parameters
    ----------
    max_context : int, default 1
        How many preceding tokens constitute the context.
    top_absent : int, default 3
        Number of high‑frequency absent tokens to record.
    """

    def __init__(self, max_context: int = 1, top_absent: int = 3) -> None:
        self.max_context = max_context
        self.top_absent = top_absent
        self.vocab: Dict[str, int] = {}
        self.id_to_token: List[str] = []
        self.context_counts: Dict[Tuple[int, ...], collections.Counter] = collections.defaultdict(collections.Counter)
        self.global_counts: collections.Counter = collections.Counter()

    def build_vocab(self, texts: Iterable[str]) -> None:
        tokens = set()
        for text in texts:
            tokens.update(text.split())
        self.id_to_token = sorted(tokens)
        self.vocab = {tok: i for i, tok in enumerate(self.id_to_token)}

    def train(self, texts: Iterable[str]) -> None:
        if not self.vocab:
            self.build_vocab(texts)
        for text in texts:
            token_ids = [self.vocab[t] for t in text.split() if t in self.vocab]
            self.global_counts.update(token_ids)
            for idx, tid in enumerate(token_ids):
                for ctx_len in range(1, self.max_context + 1):
                    if idx - ctx_len < 0:
                        break
                    ctx = tuple(token_ids[idx - ctx_len: idx])
                    self.context_counts[ctx][tid] += 1

    def _top_absent_tokens(self, context: Tuple[int, ...]) -> List[int]:
        present = set(self.context_counts.get(context, {}))
        ranked = [tid for tid, _ in self.global_counts.most_common() if tid not in present]
        return ranked[: self.top_absent]

    def encode(self, text: str) -> List[Dict[str, List[int]]]:
        token_ids = [self.vocab.get(t) for t in text.split() if t in self.vocab]
        encoded = []
        for i, tid in enumerate(token_ids):
            chosen_ctx = ()
            for ctx_len in range(self.max_context, 0, -1):
                if i - ctx_len < 0:
                    continue
                ctx = tuple(token_ids[i - ctx_len: i])
                if ctx in self.context_counts:
                    chosen_ctx = ctx
                    break
            absence = self._top_absent_tokens(chosen_ctx)
            encoded.append({"token": tid, "absence": absence})
        return encoded

    def decode(self, encoded: Iterable[Dict[str, List[int]]]) -> str:
        tokens = [self.id_to_token[item["token"]] for item in encoded]
        return " ".join(tokens)
