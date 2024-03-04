from dataclasses import dataclass

from fastlane_bot.etl.models.exchanges.uniswap.v3.exchange import UniswapV3Exchange


@dataclass
class SushiswapV3Exchange(UniswapV3Exchange):
    """
    Class representing the Sushiswap V3 exchange.
    """

    name: str = "sushiswap_v3"
