from dataclasses import dataclass, field
from typing import Any, List, Tuple, Type

from web3.contract import AsyncContract, Contract
from web3.datastructures import AttributeDict

from fastlane_bot.constants import UNSWAP_V3_POOL_MAPPINGS
from fastlane_bot.etl.models.exchanges.exchange import BaseExchange
from fastlane_bot.etl.models.exchanges.uniswap.v3 import abi
from fastlane_bot.etl.models.exchanges.uniswap.v3.pool import UniswapV3Pool


@dataclass
class UniswapV3Exchange(BaseExchange):
    """
    UniswapV3 exchange class
    """

    base_name: str = "uniswap_v3"
    name: str = "uniswap_v3"
    id: int = 4
    static_pools: dict = field(default_factory=dict)

    def __post_init__(self):
        self.static_pools = UNSWAP_V3_POOL_MAPPINGS

    def add_pool(self, pool: UniswapV3Pool):
        self.pools[pool.state[UniswapV3Pool.unique_key()]] = pool

    @staticmethod
    def get_abi():
        return abi.UNISWAP_V3_POOL_ABI

    @staticmethod
    def get_pool_abi() -> list:
        return abi.UNISWAP_V3_POOL_ABI

    @staticmethod
    def get_factory_abi():
        return abi.UNISWAP_V3_FACTORY_ABI

    @classmethod
    def event_matches_format(cls, event: AttributeDict) -> bool:
        """
        Check if an event matches the format of a Uniswap v3 event.
        """
        event_args = event["args"]
        return (
            "sqrtPriceX96" in event_args
            and event["address"] in cls.static_pools[f"{cls.name}_pools"]
        )

    def get_events(self, contract: Contract) -> List[Type[Contract]]:
        return [contract.events.Swap] if self.exchange_initialized else []

    async def async_get_fee(
        self,
        address: str,
        contract: AsyncContract,
        factory_contract: AsyncContract = None,
    ) -> Tuple[str, float]:
        fee = await contract.functions.fee().call()
        fee_float = float(fee) / 1e6
        return fee, fee_float

    @staticmethod
    async def async_get_tkn0(address: str, contract: AsyncContract, event: Any) -> str:
        return await contract.functions.token0().call()

    @staticmethod
    async def async_get_tkn1(address: str, contract: AsyncContract, event: Any) -> str:
        return await contract.functions.token1().call()
