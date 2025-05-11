import unicodedata as ud

def get_frequencies(text, counts=None):
    """
    Returns a dictionary with the frequency of each character in the text.
    If counts is provided, it will be used to update the frequencies.
    """
    counts = counts if counts is not None else {}
    for pair in zip(text, text[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(tokens, pair, new_token):
    """
    Merges the pair of tokens into a new token and returns the updated list of tokens.
    """
    new_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and pair[0] == tokens[i] and pair[1] == tokens[i+1]:
            new_tokens.append(new_token)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
            
    return new_tokens

def replace_control_chars(s: str) -> str:
    """ Replaces control characters in the text with their unicode escape sequences. """
    chars = []
    for c in s:
        if ud.category(c)[0] != 'C':
            chars.append(c)
        else:
            chars.append(f"\\u{ord(c):04x}")

def render_token(t: bytes) -> str:
    """ Renders a token as a string. """
    if isinstance(t, bytes):
        return t.decode('utf-8', errors='replace')
    elif isinstance(t, str):
        return t
    else:
        raise TypeError(f"Expected bytes or str, got {type(t)}")
    

class Tokenizer:
    """ A Base class for tokenizers. """
    def __init__(self):
        self.merges = {}
        self.special_tokens = {}
        self.vocab = self._build_vocab()
        self.pattern = ""

    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def encode(self, text):
      raise NotImplementedError("Subclasses must implement this method.")
    
    def decode(self, tokens):
      raise NotImplementedError("Subclasses must implement this method.")

    def _build_vocab(self):
        """ Builds the vocabulary from the special tokens. """
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p, q), idx in self.merges.items():
            vocab[idx] = vocab[p] + vocab[q]
        for special_token, idx in self.special_tokens.items():
            vocab[idx] = special_token.encode('utf-8')
        return vocab

    def save(self, file_path_prefix):
        """ Saves the tokenizer (merges and vocab) to a files. """
        path = file_path_prefix + ".model"
        with open(path, 'wb') as f:
            f.write("myBPE v1\n")
            f.write(f"{self.pattern}\n")
            f.write(f"{len(self.special_tokens)}\n")
            for special_token, idx in self.special_tokens.items():
                f.write(f"{special_token} {idx}\n")
            for p, q in self.merges:
                f.write(f"{p} {q}\n")

        vocab_file = file_path_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for idx, token in self.vocab.items():
                s = render_token(token)
                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"{s} {idx}\n")
    def load(self, model_path):
        """ Loads the tokenizer from a model file. """
        assert model_path.endswith(".model")
        merges = {}
        idx = 256
        special_tokens = {}
        with open(model_path) as f:
            version = f.readline().strip()
            assert version == "minbpe v1"
            self.pattern = f.readline().strip()
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()