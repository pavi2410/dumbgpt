class CharTokenizer:
    pad_token = "<pad>"
    unk_token = "<unk>"
    bos_token = "<bos>"
    eos_token = "<eos>"

    special_tokens = [
        pad_token,
        unk_token,
        bos_token,
        eos_token,
    ]

    def __init__(self):
        self.vocab: dict[str, int] = {v: k for k, v in enumerate(self.special_tokens)}
        self.reverse_vocab: dict[int, str] = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def build_vocab(self, texts: list[str]) -> None:
        chars = set("".join(texts))
        for char in chars:
            if char not in self.vocab.values():
                self.vocab[char] = self.vocab_size
                self.reverse_vocab[self.vocab_size] = char
                self.vocab_size += 1

    def encode(self, text: str) -> list[int]:
        unk_id = self.vocab[self.unk_token]

        encoding: list[int] = []
        for char in text:
            token = self.vocab.get(char, unk_id)
            encoding.append(token)
        return encoding

    def decode(self, tokens: list[int]) -> str:
        decoded_text = ""
        for token in tokens:
            decoded_text += self.reverse_vocab.get(token, self.unk_token)
        return decoded_text
