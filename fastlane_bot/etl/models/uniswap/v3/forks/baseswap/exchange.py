from dataclasses import dataclass

from fastlane_bot.etl.models.exchanges.uniswap.v3.exchange import UniswapV3Exchange


@dataclass
class BaseswapV3Exchange(UniswapV3Exchange):
    """
    Class representing the Baseswap V3 exchange.
    """

    name: str = "baseswap_v3"
