from dataclasses import dataclass

from fastlane_bot.etl.models.exchanges.uniswap.v2.exchange import UniswapV2Exchange


@dataclass
class BaseswapV2Exchange(UniswapV2Exchange):
    """
    Class representing the Baseswap V2 exchange.
    """

    name: str = "baseswap_v2"
