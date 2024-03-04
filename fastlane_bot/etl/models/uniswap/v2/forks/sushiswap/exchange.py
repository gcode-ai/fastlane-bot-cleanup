from dataclasses import dataclass

from fastlane_bot.etl.models.exchanges.uniswap.v2.exchange import UniswapV2Exchange


@dataclass
class SushiswapV2Exchange(UniswapV2Exchange):
    """
    Class representing the Sushiswap V2 exchange.
    """

    name: str = "sushiswap_v2"
