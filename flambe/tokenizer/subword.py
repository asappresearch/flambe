from typing import List

from flambe.tokenizer import Tokenizer
import fastBPE

class BPETokenizer(Tokenizer):
    """Implement a subword level tokenizer using
       byte pair encoding """

    def __init__(self, codes_path: str) -> None:
        self.bpe = fastBPE.fastBPE(codes_path)

    def tokenize(self, example: str) -> List[str]:
        return self.bpe.apply([example])[0].split(" ")
