# coding=utf-8
"""
Contains the pool class for Uniswap v2. This class is responsible for handling Uniswap v2 pools and updating the state of the pools.

(c) Copyright Bprotocol foundation 2023.
Licensed under MIT
"""
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from web3 import Web3
from web3.contract import AsyncContract, Contract

from fastlane_bot.etl.models.exchanges.pool import BasePool


@dataclass
class UniswapV2Pool(BasePool):
    """
    Class representing a Uniswap v2 pool.
    """

    @property
    def fee(self) -> str:
        """
        Get the fee.
        """
        return self._fee

    @fee.setter
    def fee(self, value: str):
        """
        Set the fee.

        Parameters
        ----------
        value : str
            The fee.
        """
        self._fee = value

    @property
    def fee_float(self) -> float:
        """
        Get the fee as a float.
        """
        return float(self.fee)

    @staticmethod
    def unique_key() -> str:
        """
        see base class.
        """
        return "address"

    def update_from_event(
        self,
        event_args: Dict[str, Any],
        data: Dict[str, Any],
        router_address: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        See base class.
        """
        event_args = event_args["args"]
        data["tkn0_balance"] = event_args["reserve0"]
        data["tkn1_balance"] = event_args["reserve1"]
        for key, value in data.items():
            self.state[key] = value

        data["cid"] = self.state["cid"]
        data["fee"] = self.state["fee"]
        data["fee_float"] = self.state["fee_float"]
        data["exchange_name"] = self.state["exchange_name"]
        return data

    def update_from_contract(
        self,
        contract: Contract,
        tenderly_fork_id: str = None,
        w3_tenderly: Web3 = None,
        w3: Web3 = None,
        tenderly_exchanges: List[str] = None,
        exchange_name: str = None,
        router_address: str = None,
    ) -> Dict[str, Any]:
        """
        See base class.
        """
        reserve_balance = contract.caller.getReserves()
        params = {
            "fee": self.fee,
            "fee_float": self.fee_float,
            "tkn0_balance": reserve_balance[0],
            "tkn1_balance": reserve_balance[1],
            "exchange_name": exchange_name,
            "router": router_address,
        }
        for key, value in params.items():
            self.state[key] = value
        return params

    async def async_update_from_contract(
        self,
        contract: AsyncContract,
        tenderly_fork_id: str = None,
        w3_tenderly: Web3 = None,
        w3: Web3 = None,
        tenderly_exchanges: List[str] = None,
        exchange_name: str = None,
        router_address: str = None,
    ) -> Optional[Dict[str, Any]]:
        """
        See base class.
        """
        reserve_balance = await contract.caller.getReserves()
        factory_address = await contract.caller.factory()
        params = {
            "fee": self.fee,
            "fee_float": self.fee_float,
            "tkn0_balance": reserve_balance[0],
            "tkn1_balance": reserve_balance[1],
            "exchange_name": exchange_name,
            "router": router_address,
            "factory": factory_address,
        }
        for key, value in params.items():
            self.state[key] = value
        return params
