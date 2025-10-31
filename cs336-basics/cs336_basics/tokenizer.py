from typing import Iterable, Iterator
import regex as re


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.token_to_id = {}
        for id, token in self.vocab.items():
            self.token_to_id[token] = id

    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        pass

    def encode(self, text: str) -> list[int]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        pre_tokens = []
        if self.special_tokens:
            specials_sorted = sorted(self.special_tokens, key=len, reverse=True)
            specials_pat = "|".join(re.escape(s) for s in specials_sorted)

            parts = re.split(f"({specials_pat})", text)
            for part in parts:
                if part in self.special_tokens:
                    pre_tokens.append([part.encode("utf-8")])
                else:
                    for pt in re.finditer(PAT, part):
                        token_text = pt.group()
                        pre_tokens.append([bytes([b]) for b in token_text.encode("utf-8")])
        else:
            for pt in re.finditer(PAT, text):
                token_text = pt.group()
                pre_tokens.append([bytes([b]) for b in token_text.encode("utf-8")])
        

        encode_ids = []
        for pre_token in pre_tokens:
            if len(pre_token) == 1:
                encode_ids.append(self.token_to_id[pre_token[0]])
                continue

            # For each merge (in order), replace all occurrences of that pair
            # within `pre_token` with their concatenation.
            for merge in self.merges:
                if len(pre_token) < 2:
                    break
                i = 0
                while i + 1 < len(pre_token):
                    if pre_token[i] == merge[0] and pre_token[i + 1] == merge[1]:
                        pre_token[i:i+2] = [merge[0] + merge[1]]
                    else:
                        i += 1

            for token in pre_token:
                encode_ids.append(self.token_to_id[token])

        return encode_ids


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            for id_ in self.encode(chunk):
                yield id_

    def decode(self, ids: list[int]) -> str:
        out_bytes = b"".join(self.vocab[id] for id in ids)
        return out_bytes.decode("utf-8", errors="replace")