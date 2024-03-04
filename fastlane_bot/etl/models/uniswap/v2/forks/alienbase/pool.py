from dataclasses import dataclass

from fastlane_bot.etl.models.exchanges.uniswap.v2.pool import UniswapV2Pool


@dataclass
class AlienbasePool(UniswapV2Pool):
    """
    Class representing a pool in Alienbase
    """