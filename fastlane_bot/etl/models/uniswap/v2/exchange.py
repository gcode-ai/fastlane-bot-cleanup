# coding=utf-8
"""
Contains the exchange class for UniswapV2. This class is responsible for handling UniswapV2 exchanges and updating the state of the pools.

(c) Copyright Bprotocol foundation 2023.
Licensed under MIT
"""
from dataclasses import dataclass, field
from typing import Any, List, Tuple, Type

from web3.contract import AsyncContract, Contract
from web3.datastructures import AttributeDict

from fastlane_bot.constants import UNSWAP_V2_POOL_MAPPINGS
from fastlane_bot.etl.models.exchanges.exchange import BaseExchange
from fastlane_bot.etl.models.exchanges.uniswap.v2 import abi
from fastlane_bot.etl.models.exchanges.uniswap.v2.pool import UniswapV2Pool


@dataclass
class UniswapV2Exchange(BaseExchange):
    """
    UniswapV2 exchange class
    """

    base_name: str = "uniswap_v2"
    name: str = "uniswap_v2"
    id: int = 3
    static_pools: dict = field(default_factory=dict)

    def __post_init__(self):
        self.static_pools = UNSWAP_V2_POOL_MAPPINGS

    @staticmethod
    def get_abi():
        return abi.UNISWAP_V2_POOL_ABI

    @staticmethod
    def get_pool_abi() -> list:
        pass

    @staticmethod
    def get_factory_abi():
        return abi.UNISWAP_V2_FACTORY_ABI

    @classmethod
    def event_matches_format(cls, event: AttributeDict) -> bool:
        """
        See base class.
        """
        event_args = event["args"]
        return (
            "reserve0" in event_args
            and event["address"] in cls.static_pools[f"{cls.name}_pools"]
        )

    def get_events(self, contract: Contract) -> List[Type[Contract]]:
        return [contract.events.Sync] if self.exchange_initialized else []

    async def async_get_fee(
        self,
        address: str,
        contract: AsyncContract,
        factory_contract: AsyncContract = None,
    ) -> Tuple[str, float]:
        return self.fee, self.fee_float

    @staticmethod
    async def async_get_tkn0(address: str, contract: AsyncContract, event: Any) -> str:
        return await contract.functions.token0().call()

    @staticmethod
    async def async_get_tkn1(address: str, contract: AsyncContract, event: Any) -> str:
        return await contract.functions.token1().call()

    def add_pool(self, pool: UniswapV2Pool):
        self.pools[pool.state[UniswapV2Pool.unique_key()]] = pool
