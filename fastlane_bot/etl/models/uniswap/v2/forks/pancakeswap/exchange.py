from dataclasses import dataclass

from fastlane_bot.etl.models.exchanges.uniswap.v2.exchange import UniswapV2Exchange


@dataclass
class PancakeswapV2Exchange(UniswapV2Exchange):
    """
    Class representing the Pancakeswap V2 exchange.
    """

    name: str = "pancakeswap_v2"
