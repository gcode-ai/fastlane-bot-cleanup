# coding=utf-8
"""
Contains the exchange class for Balancer V1.

(c) Copyright Bprotocol foundation 2024.
Licensed under MIT
"""
from dataclasses import dataclass
from typing import Dict, List, Type, Tuple, Any

from web3.contract import Contract

from fastlane_bot.etl.models.balancer.v1.pool import BalancerV1Pool
from fastlane_bot.etl.models.exchanges.exchange import BaseExchange
import abi


@dataclass
class BalancerV1Exchange(BaseExchange):
    """
    Balancer exchange class
    """

    base_name: str = "balancer_v1"
    name: str = "balancer_v1"
    id: int = 7

    @staticmethod
    def get_abi():
        return abi.VAULT_ABI

    @staticmethod
    def get_pool_abi():
        return abi.POOL_ABI

    @staticmethod
    def get_factory_abi():
        # Not used for Balancer
        return abi.VAULT_ABI

    @classmethod
    def event_matches_format(
        cls,
        event: Dict[str, Any],
        static_pools: Dict[str, Any],
        exchange_name: str = None,
    ) -> bool:
        """
        see base class.

        Not using events to update Balancer pools
        """

        return False

    def add_pool(self, pool: BalancerV1Pool):
        self.pools[pool.state["cid"]] = pool

    def get_events(self, contract: Contract) -> List[Type[Contract]]:
        return [contract.events.AuthorizerChanged]

    async def async_get_fee(
        self, pool_id: str, contract: Contract
    ) -> Tuple[str, float]:
        pool = self.get_pool(pool_id)
        if pool:
            fee, fee_float = pool.state["fee"], pool.state["fee_float"]
        else:
            fee = await contract.functions.getSwapFeePercentage().call()
            fee_float = float(fee) / 1e18
        return fee, fee_float

    async def get_tokens(self, address: str, contract: Contract, event: Any) -> []:
        pool_balances = await contract.caller.getPoolTokens(address).call()
        return pool_balances[0]

    async def get_token_balances(
        self, address: str, contract: Contract, event: Any
    ) -> []:
        pool_balances = await contract.caller.getPoolTokens(address)
        tokens = pool_balances[0]
        token_balances = pool_balances[1]
        return [{tkn: token_balances[idx]} for idx, tkn in enumerate(tokens)]

    async def async_get_tkn0(self, address: str, contract: Contract, event: Any) -> str:
        pool_balances = await contract.caller.getPoolTokens(address)
        tokens = pool_balances[0]
        token_balances = pool_balances[1]
        return token_balances[0]

    async def async_get_tkn1(self, address: str, contract: Contract, event: Any) -> str:
        pool_balances = await contract.caller.getPoolTokens(address)
        tokens = pool_balances[0]
        token_balances = pool_balances[1]
        return token_balances[1]

    async def get_tkn_n(
        self, address: str, contract: Contract, event: Any, index: int
    ) -> str:
        pool_balances = await contract.caller.getPoolTokens(address)
        tokens = pool_balances[0]
        token_balances = pool_balances[1]
        return token_balances[index]
