# coding=utf-8
"""
Contains the exchange class for Bancor V3

(c) Copyright Bprotocol foundation 2024.
Licensed under MIT
"""
from dataclasses import dataclass
from typing import List, Type, Tuple, Any

from web3.contract import AsyncContract, Contract
from web3.datastructures import AttributeDict

from fastlane_bot import constants
from fastlane_bot.etl.models.bancor.v3 import abi
from fastlane_bot.etl.models.bancor.v3.pool import BancorV3Pool
from fastlane_bot.etl.models.exchanges.exchange import BaseExchange


@dataclass
class BancorV3Exchange(BaseExchange):
    """
    BancorV3 exchange class
    """

    base_name: str = "bancor_v3"
    name: str = "bancor_v3"
    id: int = 2
    fee: str = "0.000"

    def add_pool(self, pool: BancorV3Pool):
        self.pools[pool.state[BancorV3Pool.unique_key()]] = pool

    @staticmethod
    def get_abi() -> list:
        return abi.BANCOR_V3_POOL_COLLECTION_ABI

    @staticmethod
    def get_pool_abi() -> list:
        # Not used for Bancor V3
        return []

    @staticmethod
    def get_factory_abi() -> list:
        # Not used for Bancor V3
        return []

    @classmethod
    def event_matches_format(cls, event: AttributeDict) -> bool:
        """
        See base class.
        """
        event_args = event["args"]
        return "pool" in event_args

    def get_events(self, contract: Contract) -> List[Type[Contract]]:
        return [contract.events.TradingLiquidityUpdated]

    def async_get_fee(
        self,
        address: str,
        contract: AsyncContract,
        factory_contract: AsyncContract = None,
    ) -> Tuple[str, float]:
        """
        The Bancor V3 fee is not stored in the contract, so we use the default fee.
        """
        return self.fee, self.fee_float

    def async_get_tkn0(self, address: str, contract: Contract, event: Any) -> str:
        return constants.BNT_ADDRESS

    def async_get_tkn1(self, address: str, contract: Contract, event: Any) -> str:
        return (
            event["args"]["pool"]
            if event["args"]["pool"] != constants.BNT_ADDRESS
            else event["args"]["tkn_address"]
        )
