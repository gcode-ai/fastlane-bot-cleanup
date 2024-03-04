from dataclasses import dataclass
from typing import Tuple

from web3.contract import AsyncContract

from fastlane_bot.etl.models.exchanges.solidly.v2.exchange import SolidlyV2Exchange
from fastlane_bot.etl.models.exchanges.solidly.v2.forks.velocimeter import abi


@dataclass
class VelocimeterV2Exchange(SolidlyV2Exchange):
    """
    Class representing the Velocimeter V2 exchange
    """

    name: str = "velocimeter_v2"

    @staticmethod
    def get_abi() -> list:
        return abi.VELOCIMETER_V2_POOL_ABI

    @staticmethod
    def get_pool_abi() -> list:
        return abi.VELOCIMETER_V2_POOL_ABI

    @staticmethod
    def get_factory_abi() -> list:
        return abi.VELOCIMETER_V2_FACTORY_ABI

    @staticmethod
    async def async_get_fee(
        address: str, contract: AsyncContract, factory_contract: AsyncContract = None
    ) -> Tuple[str, float]:
        fee = factory_contract.caller.getFee(address)
        fee_float = float(fee) / 10**5
        return str(fee_float), fee_float
