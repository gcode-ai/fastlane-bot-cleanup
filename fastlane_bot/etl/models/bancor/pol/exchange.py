# coding=utf-8
"""
Contains the exchange class for Bancor Protocol-owned liquidity.

(c) Copyright Bprotocol foundation 2024.
Licensed under MIT
"""
from dataclasses import dataclass
from typing import List, Type, Tuple, Any, Dict, Callable

from web3.contract import AsyncContract, Contract
from web3.datastructures import AttributeDict

import abi
from fastlane_bot import constants
from fastlane_bot.etl.models.bancor.pol.pool import BancorPolPool
from fastlane_bot.etl.models.exchanges.exchange import BaseExchange
from fastlane_bot.etl.models.manager import Manager


@dataclass
class BancorPolExchange(BaseExchange):
    """
    Bancor protocol-owned liquidity exchange class
    """

    base_name: str = "bancor_pol"
    name: str = "bancor_pol"
    id: int = 8
    fee: str = "0.000"

    BANCOR_POL_ADDRESS = "0xD06146D292F9651C1D7cf54A3162791DFc2bEf46"

    @staticmethod
    def get_abi():
        return abi.BANCOR_POL_ABI

    @staticmethod
    def get_factory_abi():
        # Not used for Bancor POL
        return []

    @staticmethod
    def get_pool_abi() -> list:
        # Not used for Bancor POL
        return []

    @classmethod
    def event_matches_format(cls, event: AttributeDict) -> bool:
        """
        See base class.
        """
        event_args = event["args"]
        return "token" in event_args and "token0" not in event_args

    def add_pool(self, pool: BancorPolPool):
        self.pools[pool.state["tkn0_address"]] = pool

    def get_events(self, contract: Contract) -> List[Type[Contract]]:
        return [contract.events.TokenTraded, contract.events.TradingEnabled]

    async def async_get_fee(
        self,
        address: str,
        contract: AsyncContract,
        factory_contract: AsyncContract = None,
    ) -> Tuple[str, float]:
        return self.fee, float(self.fee)

    async def async_get_tkn0(self, address: str, contract: Contract, event: Any) -> str:
        return event["args"]["token"]

    async def async_get_tkn1(self, address: str, contract: Contract, event: Any) -> str:
        return (
            constants.ETH_ADDRESS
            if event["args"]["token"] not in constants.ETH_ADDRESS
            else constants.BNT_ADDRESS
        )

    def save_strategy(
        self,
        token: str,
        block_number: int,
        cfg: Manager,
        func: Callable,
    ) -> Dict[str, Any]:
        """
        Add the pool info from the strategy.

        Parameters
        ----------
        token : str
            The token address for POL
        block_number : int
            The block number.
        cfg : Config
            The config.
        func : Callable
            The function to call.

        Returns
        -------
        Dict[str, Any]
            The pool info.

        """
        cid = f"{self.name}_{token}"
        tkn0_address = cfg.w3.to_checksum_address(token)
        tkn1_address = (
            cfg.w3.to_checksum_address(constants.ETH_ADDRESS)
            if token not in constants.ETH_ADDRESS
            else constants.BNT_ADDRESS
        )

        return func(
            address=self.BANCOR_POL_ADDRESS,
            exchange_name=self.name,
            fee=self.fee,
            fee_float=self.fee_float,
            tkn0_address=tkn0_address,
            tkn1_address=tkn1_address,
            cid=cid,
            block_number=block_number,
        )
