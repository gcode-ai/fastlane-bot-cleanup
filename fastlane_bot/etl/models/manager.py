import logging
import os
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from web3 import AsyncWeb3, Web3
from web3.datastructures import AttributeDict

from fastlane_bot.etl.models.blockchain import Blockchain
from fastlane_bot.etl.models.exchange import BaseExchange


@dataclass
class BaseManager(ABC):
    """
    Base manager class
    """
    logger: logging.Logger
    logging_path: str
    blockchain: Blockchain
    prefix_path: str
    pool_data: List[Dict[str, Any]]

    read_only: bool = False
    self_fund: bool = False

    exchanges: List[BaseExchange] = field(default_factory=list)
    flashloan_tokens: List[str] = field(default_factory=list)
    tokens: List[Dict[str, str]] = field(default_factory=dict)
    target_tokens: List[str] = field(default_factory=list)
    pools_to_add_from_contracts: List[str] = field(default_factory=list)
    uniswap_v2_event_mappings: Dict[str, List[str]] = field(default_factory=dict)
    uniswap_v3_event_mappings: Dict[str, List[str]] = field(default_factory=dict)
    solidly_v2_event_mappings: Dict[str, List[str]] = field(default_factory=dict)
    static_pools: Dict[str, List[str]] = field(default_factory=dict)

    _w3: Web3 = None
    _w3_async: AsyncWeb3 = None
    _w3_tenderly: Web3 = None

    @property
    def exchange_ids(self) -> List[int]:
        return [ex.id for ex in self.exchanges]

    @property
    def arb_contract(self):
        return self._arb_contract

    @arb_contract.setter
    def arb_contract(self, value: str):
        self._arb_contract = value

    @property
    def arb_reward_percentage(self):
        (
            reward_percent,
            max_profit,
        ) = self.arb_contract.caller.rewards()
        return str(int(reward_percent) / 1000000)

    @property
    def bancor_network_info_contract(self):
        return self._bancor_network_info_contract

    @bancor_network_info_contract.setter
    def bancor_network_info_contract(self, value: str):
        self._bancor_network_info_contract = value

    @property
    def w3(self):
        return self._w3

    @w3.setter
    def w3(self, value: Web3):
        self._w3 = value

    @property
    def w3_async(self):
        return self._w3_async

    @w3_async.setter
    def w3_async(self, value: AsyncWeb3):
        self._w3_async = value

    @property
    def w3_tenderly(self):
        return self._w3_tenderly

    @w3_tenderly.setter
    def w3_tenderly(self, value: Web3):
        self._w3_tenderly = value

    def exchange_name_from_event(self, event: AttributeDict) -> Optional[str]:
        """
        Get the exchange name from the event.

        Args:
            event (AttributeDict): The event.

        Returns:
            str: The exchange name.
        """
        return next(
            (ex.name for ex in self.exchanges if ex.event_matches_format(event)),
            None,
        )

    def get_exchange_by_name(self, exchange_name: str):
        for ex in self.exchanges:
            if ex.name == exchange_name:
                return ex

    def get_forks_of_exchange(self, exchange_name: str) -> List[BaseExchange]:
        return [
            exchange
            for exchange in self.blockchain.exchanges
            if exchange.fork_of == exchange_name
        ]

    @property
    def supported_exchanges(self):
        return [ex.name for ex in self.exchanges]

    @property
    def uniswap_v3_forks(self):
        ex = self.get_exchange_by_name("uniswap_v3")
        return self.get_forks_of_exchange(ex.name)

    @property
    def uniswap_v2_forks(self):
        ex = self.get_exchange_by_name("uniswap_v2")
        return self.get_forks_of_exchange(ex.name)

    @property
    def solidly_v2_forks(self):
        ex = self.get_exchange_by_name("solidly_v2")
        return self.get_forks_of_exchange(ex.name)

    @property
    def carbon_v1_forks(self):
        ex = self.get_exchange_by_name("carbon_v1")
        return self.get_forks_of_exchange(ex.name)

    @property
    def is_gas_token_in_flashloan_tokens(self):
        return self.blockchain.native_gas_token_address in self.flashloan_tokens

    @property
    def wallet(self):
        self.w3.eth.account.from_key(
            os.getenv('ETH_PRIVATE_KEY_BE_CAREFUL')
        )



class Manager(BaseManager):
    """
    Manager class
    """
    pass