# coding=utf-8
"""
Contains the pool class for Bancor v2.

(c) Copyright Bprotocol foundation 2024.
Licensed under MIT
"""
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from web3 import Web3
from web3.contract import AsyncContract, Contract

from fastlane_bot.etl.models.exchanges.pool import BasePool


@dataclass
class BancorV2Pool(BasePool):
    """
    Class representing a Bancor v2 pool.
    """

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
        **** IMPORTANT ****
        Bancor V2 pools emit 3 events per trade. Only one of them contains the new token balances we want.
        The one we want is the one where _token1 and _token2 match the token addresses of the pool.

        Args:
            router_address ():
        """
        data["tkn0_address"] = event_args["args"]["_token1"]
        data["tkn1_address"] = event_args["args"]["_token2"]
        data["tkn0_balance"] = event_args["args"]["_rateD"]
        data["tkn1_balance"] = event_args["args"]["_rateN"]

        for key, value in data.items():
            self.state[key] = value

        data["anchor"] = self.state["anchor"]
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
        reserve0, reserve1 = contract.caller.reserveBalances()
        tkn0_address, tkn1_address = contract.caller.reserveTokens()
        fee = contract.caller.conversionFee()

        params = {
            "fee": fee,
            "fee_float": fee / 1e6,
            "exchange_name": self.state["exchange_name"],
            "address": self.state["address"],
            "anchor": contract.caller.anchor(),
            "tkn0_balance": reserve0,
            "tkn1_balance": reserve1,
            "tkn0_address": tkn0_address,
            "tkn1_address": tkn1_address,
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
        reserve0, reserve1 = await contract.caller.reserveBalances()
        tkn0_address, tkn1_address = await contract.caller.reserveTokens()
        fee = await contract.caller.conversionFee()

        params = {
            "fee": fee,
            "fee_float": fee / 1e6,
            "exchange_name": self.state["exchange_name"],
            "address": self.state["address"],
            "anchor": await contract.caller.anchor(),
            "tkn0_balance": reserve0,
            "tkn1_balance": reserve1,
            "tkn0_address": tkn0_address,
            "tkn1_address": tkn1_address,
        }
        for key, value in params.items():
            self.state[key] = value
        return params
