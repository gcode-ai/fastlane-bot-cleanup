from dataclasses import dataclass

from fastlane_bot.etl.models.exchanges.uniswap.v3.exchange import UniswapV3Exchange


@dataclass
class PancakeswapV3Exchange(UniswapV3Exchange):
    """
    Class representing the Pancakeswap V3 exchange.
    """

    name: str = "pancakeswap_v3"
