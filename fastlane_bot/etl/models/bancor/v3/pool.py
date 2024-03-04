# coding=utf-8
"""
Contains the exchange class for BancorV3. This class is responsible for handling BancorV3 events and updating the state of the pools.

(c) Copyright Bprotocol foundation 2023.
Licensed under MIT
"""
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from web3 import Web3
from web3.contract import AsyncContract, Contract

from fastlane_bot import constants
from fastlane_bot.etl.models.exchanges.pool import BasePool


@dataclass
class BancorV3Pool(BasePool):
    """
    Class representing a Bancor v3 pool.
    """

    @staticmethod
    def unique_key() -> str:
        """
        see base class.
        """
        return "tkn1_address"

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
        if event_args["tkn_address"] == constants.BNT_ADDRESS:
            data["tkn0_balance"] = event_args["newLiquidity"]
        else:
            data["tkn1_balance"] = event_args["newLiquidity"]

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
        pool_balances = contract.caller.tradingLiquidity(self.state["tkn1_address"])

        params = {
            "fee": "0.000",
            "fee_float": 0.000,
            "tkn0_balance": pool_balances[0],
            "tkn1_balance": pool_balances[1],
            "exchange_name": self.state["exchange_name"],
            "address": self.state["address"],
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
        Bancor V3 uses multicall, so we can't use the async contract to get the pool balances.
        """
        return None
