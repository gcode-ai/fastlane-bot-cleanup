from dataclasses import dataclass

from fastlane_bot.etl.models.exchanges.uniswap.v3.pool import UniswapV3Pool


@dataclass
class SushiswapV3Pool(UniswapV3Pool):
    """
    Class representing a pool in Sushiswap V3
    """
