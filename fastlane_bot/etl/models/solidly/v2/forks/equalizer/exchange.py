from dataclasses import dataclass
from typing import Tuple

from web3.contract import AsyncContract

from fastlane_bot.etl.models.exchanges.solidly.v2.exchange import SolidlyV2Exchange
from fastlane_bot.etl.models.exchanges.solidly.v2.forks.equalizer import abi


@dataclass
class EqualizerV2Exchange(SolidlyV2Exchange):
    """
    Class representing the EqualizerV2 exchange
    """

    name: str = "equalizer_v2"

    @staticmethod
    def get_abi() -> list:
        return abi.EQUALIZER_V2_POOL_ABI

    @staticmethod
    def get_pool_abi() -> list:
        return abi.EQUALIZER_V2_POOL_ABI

    @staticmethod
    def get_factory_abi() -> list:
        return abi.SCALE_V2_FACTORY_ABI

    async def async_get_fee(
        self,
        address: str,
        contract: AsyncContract,
        factory_contract: AsyncContract = None,
    ) -> Tuple[str, float]:
        fee = await factory_contract.caller.getRealFee(address)
        fee_float = float(fee) / 10**5
        self.fee = str(fee_float)
        return str(fee_float), fee_float
