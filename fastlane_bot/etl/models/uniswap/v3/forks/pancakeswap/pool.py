from dataclasses import dataclass

from fastlane_bot.etl.models.exchanges.uniswap.v3.pool import UniswapV3Pool


@dataclass
class PancakeswapV3Pool(UniswapV3Pool):
    """
    Class representing a pool in Pancakeswap V3
    """
