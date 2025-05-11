""" A Basic Byte pair encoding tokenizer, doesn't support special tokens and regular expressions. """

from base import Tokenizer, get_frequencies, merge

class BasicTokenizer(Tokenizer):
    def __init__(self):
      super().__init__()

    def train(self, text, vocab_size, verbose=False):
      """ Trains the tokenizer on the given text. """
      assert vocab_size >= 256
      num_merges = vocab_size - 256
      text_bytes = text.encode('utf-8', errors='replace')
      ids = list(text_bytes)
      merges = {}
      vocab = {i: bytes([i]) for i in range(256)}
      for i in range(num_merges):
        freqs = get_frequencies(ids)
        pair = max(freqs, key=freqs.get)
        idx = 256 + i
        ids = merge(ids, pair, idx)
        merges[pair] = idx
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        if verbose:
          print(f"Merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {freqs[pair]} occurrences")

      self.merges = merges
      self.vocab = vocab


    def decode(self, ids):
      """ Decodes the given tokens. """
      text_bytes = b''.join(self.vocab[idx] for idx in ids)
      text = text_bytes.decode('utf-8', errors='replace')
      return text
  
    def encode(self, text):
      """ Encodes the given text. """
      text_bytes = text.encode('utf-8', errors='replace')
      ids = list(text_bytes)
      while len(ids) >= 2:
        freqs = get_frequencies(ids)
        pair = min(freqs, key = lambda x: self.merges.get(x, float('inf')))
        if pair not in self.merges:
          break
        idx = self.merges[pair]
        ids = merge(ids, pair, idx)
      return ids