from dataclasses import dataclass

from fastlane_bot.etl.models.exchanges.uniswap.v2.exchange import UniswapV2Exchange


@dataclass
class AlienbaseExchange(UniswapV2Exchange):
    """
    Class representing the Alienbase exchange.
    """

    name: str = "alienbase_v2"
