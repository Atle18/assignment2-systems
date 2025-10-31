import regex as re

def is_greater(merge1, merge2):
    p1, p2 = 0, 0
    while p1 < len(merge1) and p2 < len(merge2):
        if merge1[p1] > merge2[p2]:
            return True
        elif merge1[p1] == merge2[p2]:
            p1 += 1
            p2 += 1
        else:
            return False
    return p2 == len(merge2)


def pre_tokenizer(text_chunks):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_token_freq = {}
    for chunk in text_chunks:
        for pt in re.finditer(PAT, chunk):
            pre_token_encode = pt.group().encode('utf-8')
            key = tuple(bytes([b]) for b in pre_token_encode)
            pre_token_freq[key] = pre_token_freq.get(key, 0) + 1
    return pre_token_freq


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    vocab = {i: bytes([i]) for i in range(256)}

    for sp in special_tokens:
        sp_encode = sp.encode('utf-8')
        vocab[len(vocab)] = sp_encode

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    if special_tokens:
        pattern = "|".join(re.escape(t) for t in special_tokens)
        text_chunks = re.split(pattern, text)
    else:
        text_chunks = [text]

    pre_token_freq = pre_tokenizer(text_chunks)

    merges = []
    while len(vocab) < vocab_size:
        freq = {}
        for pre_token, pre_token_cnt in pre_token_freq.items():
            for i in range(1, len(pre_token)):
                curr_merge = (pre_token[i-1], pre_token[i])
                if curr_merge not in freq:
                    freq[curr_merge] = [0, set()]
                freq[curr_merge][0] += pre_token_cnt
                freq[curr_merge][1].add(pre_token)
        
        if len(freq) == 0:
            break

        max_cnt = 0
        max_merge = None
        for merge, cnt in freq.items():
            cnt = cnt[0]
            if cnt > max_cnt or (cnt == max_cnt and is_greater(merge, max_merge)):
                max_cnt = cnt
                max_merge = merge
        
        a, b = max_merge
        merges.append(max_merge)
        pre_token_set = freq[max_merge][1]

        max_merge = b"".join(max_merge)
        vocab[len(vocab)] = max_merge

        for pre_token in pre_token_set:
            i = 0
            curr_ = []
            while i < len(pre_token):
                if i+1 < len(pre_token) and (pre_token[i] == a and pre_token[i+1] == b):
                    curr_.append(max_merge)
                    i += 2
                else:
                    curr_.append(pre_token[i])
                    i += 1
            
            cnt_ = pre_token_freq[pre_token]
            del pre_token_freq[pre_token]
            pre_token_freq[tuple(curr_)] = pre_token_freq.get(tuple(curr_), 0) + cnt_

    return vocab, merges


if __name__ == "__main__":
    input_path = "../data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 1000
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    print(vocab)