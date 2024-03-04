from dataclasses import dataclass
from typing import Tuple

from web3.contract import AsyncContract

from fastlane_bot.etl.models.exchanges.solidly.v2.exchange import SolidlyV2Exchange
from fastlane_bot.etl.models.exchanges.solidly.v2.forks.aerodrome import abi


@dataclass
class AerodromeExchange(SolidlyV2Exchange):
    """
    Class representing the Aerodrome exchange
    """

    name: str = "aerodrome_v2"
    id: int = 12

    @staticmethod
    def get_abi() -> list:
        return abi.SOLIDLY_V2_POOL_ABI

    @staticmethod
    def get_pool_abi() -> list:
        return abi.SOLIDLY_V2_POOL_ABI

    @staticmethod
    def get_factory_abi() -> list:
        return abi.SOLIDLY_V2_FACTORY_ABI

    async def async_get_fee(
        self,
        address: str,
        contract: AsyncContract,
        factory_contract: AsyncContract = None,
    ) -> Tuple[str, float]:
        fee = await factory_contract.caller.getFee(
            address, await contract.caller.stable()
        )
        fee_float = float(fee) / 10**5
        self.fee = str(fee_float)
        return str(fee_float), fee_float
