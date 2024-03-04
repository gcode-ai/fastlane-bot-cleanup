"""
This file contains the constants used in the fastlane_bot package.

(c) Copyright Bprotocol foundation 2024.
Licensed under MIT License.
"""
import os
from dataclasses import dataclass
from decimal import Decimal

import pandas as pd

from fastlane_bot.abi import (
    EQUALIZER_V2_POOL_ABI,
    SCALE_V2_FACTORY_ABI,
    SOLIDLY_V2_FACTORY_ABI,
    SOLIDLY_V2_POOL_ABI,
    VELOCIMETER_V2_FACTORY_ABI,
    VELOCIMETER_V2_POOL_ABI,
)

BASE_PATH = os.path.normpath("staticdata/blockchain_data/{{blockchain}}")
MULTICHAIN_ADDRESS_PATH = os.path.normpath("staticdata/multichain_addresses.csv")
SANITIZATION_PATH = os.path.normpath(
    "staticdata/data_sanitization_center/sanitary_data.csv"
)
TOKENS_PATH = os.path.normpath("staticdata/blockchain_data/{{blockchain}}/tokens.csv")
UNISWAP_V2_EVENT_MAPPINGS_PATH = os.path.normpath(
    "staticdata/blockchain_data/{{blockchain}}/uniswap_v2_event_mappings.csv"
)
UNISWAP_V3_EVENT_MAPPINGS_PATH = os.path.normpath(
    "staticdata/blockchain_data/{{blockchain}}/uniswap_v3_event_mappings.csv"
)
SOLIDLY_V2_EVENT_MAPPINGS_PATH = os.path.normpath(
    "staticdata/blockchain_data/{{blockchain}}/solidly_v2_pool_mappings.csv"
)
CHAIN_SPECIFIC_INFO_PATH = os.path.normpath("staticdata/chain_specific_info.csv")
CHAIN_SPECIFIC_INFO = pd.DataFrame(pd.read_csv(CHAIN_SPECIFIC_INFO_PATH))
MULTICALLABLE_EXCHANGES = ["bancor_v3", "bancor_pol", "balancer_v1"]

TENDERLY_FORK_ID = None
TENDERLY_EVENT_EXCHANGES = "pancakeswap_v2,pancakeswap_v3"
ONE = 2**48
DEFAULT_BLOCKTIME_DEVIATION = 13 * 500 * 100
DEFAULT_GAS = 950_000
DEFAULT_GAS_PRICE = 0
DEFAULT_GAS_PRICE_OFFSET = 1.09
DEFAULT_GAS_SAFETY_OFFSET = 25_000
DEFAULT_MIN_PROFIT_GAS_TOKEN = Decimal("0.02")
EXPECTED_GAS_MODIFIER = "0.85"

BINANCE14_WALLET_ADDRESS = "0x28c6c06298d514db089934071355e5743bf21d60"

ETHEREUM = "ethereum"
POLYGON = "polygon"
POLYGON_ZKEVM = "polygon_zkevm"
ARBITRUM_ONE = "arbitrum_one"
OPTIMISM = "optimism"
BASE = "coinbase_base"
FANTOM = "fantom"

SOLIDLY_EXCHANGE_INFO = {
    "velocimeter_v2": {
        "decimals": 5,
        "factory_abi": VELOCIMETER_V2_FACTORY_ABI,
        "pool_abi": VELOCIMETER_V2_POOL_ABI,
    },
    "equalizer_v2": {
        "decimals": 5,
        "factory_abi": SCALE_V2_FACTORY_ABI,
        "pool_abi": EQUALIZER_V2_POOL_ABI,
    },
    "aerodrome_v2": {
        "decimals": 5,
        "factory_abi": SOLIDLY_V2_FACTORY_ABI,
        "pool_abi": SOLIDLY_V2_POOL_ABI,
    },
    "velodrome_v2": {
        "decimals": 5,
        "factory_abi": SOLIDLY_V2_FACTORY_ABI,
        "pool_abi": SOLIDLY_V2_POOL_ABI,
    },
    "scale_v2": {
        "decimals": 18,
        "factory_abi": SCALE_V2_FACTORY_ABI,
        "pool_abi": SOLIDLY_V2_POOL_ABI,
    },
}


@dataclass
class CommonEthereumTokens:
    """
    Common Ethereum tokens
    """

    ETH: str = "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE"
    WETH: str = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
    USDC: str = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
    USDT: str = "0xdAC17F958D2ee523a2206206994597C13D831ec7"
    WBTC: str = "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"
    BNT: str = "0x1F573D6Fb3F13d689FF844B4cE37794d79a7FF1C"
    LINK: str = "0x514910771AF9Ca656af840dff83E8264EcF986CA"
    DAI: str = "0x6B175474E89094C44Da98b954EedeAC495271d0F"
