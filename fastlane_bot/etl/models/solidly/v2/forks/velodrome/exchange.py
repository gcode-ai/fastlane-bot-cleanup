from dataclasses import dataclass
from typing import Tuple

from web3.contract import AsyncContract

from fastlane_bot.etl.models.exchanges.solidly.v2.exchange import SolidlyV2Exchange
from fastlane_bot.etl.models.exchanges.solidly.v2.forks.velodrome import abi


@dataclass
class VelodromeV2Exchange(SolidlyV2Exchange):
    """
    Class representing the Velodrome V2 exchange

    TODO: Validate this class / abi / factory_abi / async_get_fee() / etc with Kevin and Nick
    """

    name: str = "velodrome_v2"
    id: int = 12

    @staticmethod
    def get_abi() -> list:
        return abi.VELODROME_V2_POOL_ABI

    @staticmethod
    def get_pool_abi() -> list:
        return abi.VELODROME_V2_POOL_ABI

    @staticmethod
    def get_factory_abi() -> list:
        return abi.VELODROME_V2_FACTORY_ABI

    @staticmethod
    async def async_get_fee(
        address: str, contract: AsyncContract, factory_contract: AsyncContract = None
    ) -> Tuple[str, float]:
        fee = factory_contract.caller.getFee(address)
        fee_float = float(fee) / 10**5
        return str(fee_float), fee_float
