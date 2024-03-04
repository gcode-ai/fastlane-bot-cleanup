# coding=utf-8
"""
Contains the exchange class for Bancor V2.

(c) Copyright Bprotocol foundation 2024.
Licensed under MIT
"""
from dataclasses import dataclass
from typing import Any, List, Tuple, Type

from web3.contract import AsyncContract, Contract
from web3.datastructures import AttributeDict

from fastlane_bot.etl.models.bancor.v2 import abi
from fastlane_bot.etl.models.bancor.v2.pool import BancorV2Pool
from fastlane_bot.etl.models.exchanges.exchange import BaseExchange


@dataclass
class BancorV2Exchange(BaseExchange):
    """
    Bancor V2 exchange class
    """

    base_name = "bancor_v2"
    name: str = "bancor_v2"
    id: int = 1

    @staticmethod
    def get_pool_abi() -> list:
        pass

    @staticmethod
    def get_abi():
        return abi.CONVERTER_ABI

    @staticmethod
    def get_factory_abi():
        # Not used for Bancor V2
        return abi.CONVERTER_ABI

    @classmethod
    def event_matches_format(cls, event: AttributeDict) -> bool:
        """
        See base class.
        """
        event_args = event["args"]
        return "_rateN" in event_args

    def add_pool(self, pool: BancorV2Pool):
        self.pools[pool.state[BancorV2Pool.unique_key()]] = pool

    def get_events(self, contract: Contract) -> List[Type[Contract]]:
        return [contract.events.TokenRateUpdate]

    async def get_connector_tokens(self, contract: AsyncContract, i: int) -> str:
        return await contract.functions.connectorTokens(i).call()

    async def async_get_fee(
        self,
        address: str,
        contract: AsyncContract,
        factory_contract: AsyncContract = None,
    ) -> Tuple[str, float]:
        pool = self.get_pool(address)
        if pool:
            fee, fee_float = pool.state["fee"], pool.state["fee_float"]
        else:
            fee = await contract.functions.conversionFee().call()
            fee_float = float(fee) / 1e6
        return fee, fee_float

    async def async_get_tkn0(
        self, address: str, contract: AsyncContract, event: Any
    ) -> str:
        if event:
            return event["args"]["_token1"]
        return await contract.functions.reserveTokens().call()[0]

    async def async_get_tkn1(
        self, address: str, contract: AsyncContract, event: Any
    ) -> str:
        if event:
            return event["args"]["_token2"]
        return await contract.functions.reserveTokens().call()[1]

    async def get_anchor(self, contract: AsyncContract) -> str:
        return await contract.caller.anchor()
