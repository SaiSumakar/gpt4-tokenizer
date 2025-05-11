""" Tokenizer handles optional regex splitting and special tokens. """

import regex as re
from base import Tokenizer, get_frequencies, merge

GPT4_PAT = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(Tokenizer):
    def __init__(self, regex_pattern=None):
      super().__init__()
      self.pattern = GPT4_PAT if regex_pattern is None else regex_pattern
      self.complied_patttern = re.compile(self.pattern)
      self.special_tokens = {}
      self.inverse_special_tokens = {}

    def train(self, text, vocab_size, verbose=False):
      assert vocab_size >= 256
      num_merges = vocab_size - 256
      text_chunks = re.findall(self.complied_patttern, text)

      ids = [list(ch.encode('utf-8')) for ch in text_chunks]

      merges = {}
      vocab = {i: bytes([i]) for i in range(256)}

      for i in range(num_merges):
        freqs = {}
        for chunk in text_chunks:
          get_frequencies(chunk, freqs)

        pair = max(freqs, key=freqs.get)
        idx = 256 + i
        ids = [merge(chunk_id, pair, idx) for chunk_id in ids]
        merges[pair] = idx
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        if verbose:
          print(f"Merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) has {freqs[pair]} occurrences")

      self.merges = merges
      self.vocab = vocab  

    def get_special_token(self, special_tokens):
      self.special_tokens = special_tokens
      self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def _encode_chunk(self, text_bytes):
      ids = list(text_bytes)
      while len(ids) > 2:
        freqs = get_frequencies(ids)
        pair = min(freqs, key=lambda x: self.merges.get(x, float('inf')))
        if pair not in self.merges:
          break
        idx = self.merges[pair]
        ids = merge(ids, pair, idx)
      return ids
    

    def encode_ordinary_text(self, text):
      text_chunks = re.findall(self.compiled_pattern, text)
      ids = []
      for chunk in text_chunks:
          chunk_bytes = chunk.encode("utf-8") # raw bytes
          chunk_ids = self._encode_chunk(chunk_bytes)
          ids.extend(chunk_ids)
      return ids

    def encode(self, text, allowed_special="none_raise"): 
      special = None
      if allowed_special == "all":
        special = self.special_tokens
      elif allowed_special == "none":
        special = {}
      elif allowed_special == "none_raise":
        special = {}
        assert all(token not in text for token in self.special_tokens)
      elif isinstance(allowed_special, set):
        special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
      else:
        raise ValueError(f"allowed_special={allowed_special} not understood")
      if not special:
        return self.encode_ordinary_text(text)
      
      special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
      special_chunks = re.split(special_pattern, text)
      ids = []
      for part in special_chunks:
        if part in special:
          ids.append(special[part])
        else:
          ids.extend(self.encode_ordinary_text(part))
      return ids

    def decode(self, ids):
      part_bytes = []
      for idx in ids:
        if idx in self.vocab:
          part_bytes.append(self.vocab[idx])
        elif idx in self.inverse_special_tokens:
          part_bytes.append(self.inverse_special_tokens[idx].encode('utf-8'))
        else:
          raise ValueError(f"Unknown token ID: {idx}")
      text_bytes = b''.join(part_bytes)
      text = text_bytes.decode('utf-8', errors='replace')
      return text