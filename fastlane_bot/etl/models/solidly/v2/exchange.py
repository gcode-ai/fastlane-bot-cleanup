from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Tuple, Type

import pandas as pd
from web3.contract import AsyncContract, Contract
from web3.datastructures import AttributeDict

from fastlane_bot.constants import SOLIDLY_V2_EVENT_MAPPINGS_PATH
from fastlane_bot.etl.models.exchange import BaseExchange
from fastlane_bot.etl.models.solidly.v2.pool import SolidlyV2Pool


@dataclass
class SolidlyV2Exchange(BaseExchange):
    """
    SolidlyV2 exchange class
    """

    base_name: str = "solidly_v2"
    name: str = "solidly_v2"

    stable_fee: float = None
    volatile_fee: float = None
    static_pools: dict = field(default_factory=dict)

    def __post_init__(self):
        self.static_pools = pd.read_csv(SOLIDLY_V2_EVENT_MAPPINGS_PATH).to_dict(
            orient="list"
        )

    def add_pool(self, pool: SolidlyV2Pool):
        self.pools[pool.state[SolidlyV2Pool.unique_key()]] = pool

    @staticmethod
    @abstractmethod
    def get_abi() -> list:
        pass

    @staticmethod
    @abstractmethod
    def get_pool_abi() -> list:
        pass

    @staticmethod
    @abstractmethod
    def get_factory_abi() -> list:
        pass

    @classmethod
    def event_matches_format(cls, event: AttributeDict) -> bool:
        """
        Check if an event matches the format of a Uniswap v2 event.
        """
        event_args = event["args"]
        return (
            "reserve0" in event_args
            and event["address"] in cls.static_pools[f"{cls.name}_pools"]
        )

    @abstractmethod
    async def async_get_fee(
        self,
        address: str,
        contract: AsyncContract,
        factory_contract: AsyncContract = None,
    ) -> Tuple[str, float]:
        pass

    async def async_get_tkn0(
        self, address: str, contract: AsyncContract, event: Any
    ) -> str:
        return await contract.functions.token0().call()

    async def async_get_tkn1(
        self, address: str, contract: AsyncContract, event: Any
    ) -> str:
        return await contract.functions.token1().call()

    def get_events(self, contract: Contract) -> List[Type[Contract]]:
        return [contract.events.Sync] if self.exchange_initialized else []
