# coding=utf-8
"""
Contains the pool class for Solidly v2.

(c) Copyright Bprotocol foundation 2024.
Licensed under MIT
"""
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from web3 import Web3
from web3.contract import AsyncContract, Contract

from fastlane_bot.etl.models.pool import BasePool


@dataclass
class SolidlyV2Pool(BasePool):
    """
    Class representing a Solidly v2 pool.
    """

    # If this is false, it's a Uni V2 style pool
    is_stable: bool = None

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
        data["router"] = router_address
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

        try:
            factory_address = contract.caller.factory()
        except Exception:
            # Velocimeter does not expose factory function - call voter to get an address that is the same for all
            # Velcoimeter pools
            factory_address = contract.caller.voter()

        self.is_stable = contract.caller.stable()
        return self._extracted(
            exchange_name, factory_address, reserve_balance, router_address
        )

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

        try:
            factory_address = await contract.caller.factory()
        except Exception:
            # Velocimeter does not expose factory function - call voter to get an address that is the same for all
            # Velocimeter pools
            factory_address = await contract.caller.voter()

        self.is_stable = await contract.caller.stable()
        return self._extracted(
            exchange_name, factory_address, reserve_balance, router_address
        )

    def _extracted(
        self, exchange_name, factory_address, reserve_balance, router_address
    ):
        params = {
            "tkn0_balance": reserve_balance[0],
            "tkn1_balance": reserve_balance[1],
            "exchange_name": exchange_name,
            "router": router_address,
            "factory": factory_address,
            "pool_type": self.pool_type,
        }
        for key, value in params.items():
            self.state[key] = value
        return params

    @property
    def pool_type(self):
        return "stable" if self.is_stable else "volatile"
