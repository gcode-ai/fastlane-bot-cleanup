# coding=utf-8
"""
Contains the pool class for Balancer v1.

(c) Copyright Bprotocol foundation 2024.
Licensed under MIT
"""
from dataclasses import dataclass
from typing import Dict, Any, List

from web3 import Web3
from web3.contract import Contract

from fastlane_bot.etl.models.exchanges.pool import BasePool


@dataclass
class BalancerV1Pool(BasePool):
    """
    Class representing a Balancer V1.
    """

    @staticmethod
    def unique_key() -> str:
        """
        see base class.
        """
        return "cid"

    def update_from_event(
        event_args, data: Dict[str, Any], router_address
    ) -> Dict[str, Any]:
        """
        See base class.

        Not using events to update Balancer pools

        Args:
            router_address ():
        """

        return data

    async def update_from_contract(
        self,
        contract: Contract,
        tenderly_fork_id: str = None,
        w3_tenderly: Web3 = None,
        w3: Web3 = None,
        tenderly_exchanges: List[str] = None,
    ) -> Dict[str, Any]:
        """
        See base class.
        """
        pool_balances = await contract.caller.getPoolTokens(self.state["anchor"])
        tokens = pool_balances[0]
        token_balances = pool_balances[1]
        params = {key: self.state[key] for key in self.state.keys()}

        for idx, tkn in enumerate(tokens):
            tkn_bal = "tkn" + str(idx) + "_balance"
            params[tkn_bal] = token_balances[idx]

        for key, value in params.items():
            self.state[key] = value

        return params
