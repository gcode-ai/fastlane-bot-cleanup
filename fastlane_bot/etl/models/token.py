from dataclasses import dataclass

@dataclass
class Token:
    """
    Token class
    """

    symbol: str
    address: str
    decimals: int

    def __eq__(self, other):
        return self.address == other.address if isinstance(other, Token) else False

    def __hash__(self):
        return hash(self.address)


