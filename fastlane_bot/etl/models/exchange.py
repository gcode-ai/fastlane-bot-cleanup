# coding=utf-8
"""
Contains the base class for all exchanges.

(c) Copyright Bprotocol foundation 2024.
Licensed under MIT
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Any

from web3.contract import Contract, AsyncContract
from web3.datastructures import AttributeDict

from fastlane_bot.etl.models.blockchain import Blockchain
from fastlane_bot.etl.models.pool import BasePool


@dataclass
class BaseExchange(ABC):
    """
    Base class for exchanges.

    Attributes
    ----------
    name : str
        The name of the exchange
    blockchain : Blockchain
        The blockchain of the exchange
    id : int
        The id of the exchange. WARNING!!! This must match the Bancor FastLane ArbitrageContract id.
    fee : str
        The fee of the exchange
    router_address : str
        The router address of the exchange
    factory_address : str
        The factory address of the exchange
    """

    base_name: str
    name: str
    blockchain: Blockchain
    id: int
    fee: str
    start_block: int = 0
    is_active: bool = True
    supports_flashloans: bool = False

    pools: Dict[str, BasePool] = field(default_factory=dict)

    router_address: Optional[str] = None
    factory_address: Optional[str] = None
    exchange_initialized: bool = False

    @property
    def fee_float(self) -> float:
        return float(self.fee)

    @property
    def fork_of(self) -> Optional[str]:
        return self.base_name if self.base_name != self.name else None

    @property
    def is_fork(self) -> bool:
        return self.fork_of is not None

    @staticmethod
    @abstractmethod
    def get_abi() -> list:
        """
        Get the ABI of the exchange

        Returns
        -------
        ABI
            The ABI of the exchange Pool (if exists else empty list)

        """
        pass

    @staticmethod
    @abstractmethod
    def get_pool_abi() -> list:
        """
        Get the Pool ABI of the exchange (if exists)

        Returns
        -------
        ABI
            The ABI of the exchange Pool (if exists else empty list)

        """
        pass

    @staticmethod
    @abstractmethod
    def get_factory_abi() -> list:
        """
        Get the ABI of the exchange's Factory contract

        Returns
        -------
        ABI
            The ABI of the exchange Pool (if exists else empty list)

        """
        pass

    @abstractmethod
    def add_pool(self, pool: BasePool):
        """
        Add a pool to the exchange.

        Parameters
        ----------
        pool : Pool
            The pool object

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def get_events(self, contract: Contract) -> List[Type[Contract]]:
        """
        Get the events of the exchange

        Parameters
        ----------
        contract : Contract
            The contract object

        Returns
        -------
        List[Type[Contract]]
            The events of the exchange

        """
        pass

    @abstractmethod
    async def async_get_fee(
        self,
        address: str,
        contract: AsyncContract,
        factory_contract: AsyncContract = None,
    ) -> Tuple[str, float]:
        """
        Get the fee of the exchange

        Parameters
        ----------
        address : str
            The address of the exchange
        contract : Contract
            The contract object
        factory_contract : Contract
            The factory contract object

        Returns
        -------
        Tuple[str, float]
            The fee of the exchange

        """
        pass

    @staticmethod
    @abstractmethod
    async def async_get_tkn0(address: str, contract: AsyncContract, event: Any) -> str:
        """
        Get the tkn0 of the exchange

        Parameters
        ----------
        address : str
            The address of the exchange
        contract : Contract
            The contract object
        event : Any
            The event object

        Returns
        -------
        str
            The tkn0 of the exchange
        """
        pass

    @staticmethod
    @abstractmethod
    async def async_get_tkn1(address: str, contract: AsyncContract, event: Any) -> str:
        """
        Get the tkn1 of the exchange

        Parameters
        ----------
        address : str
            The address of the exchange
        contract : Contract
            The contract object
        event : Any
            The event object

        Returns
        -------
        str
            The tkn1 of the exchange

        """
        pass

    @classmethod
    @abstractmethod
    def event_matches_format(
        cls,
        event: AttributeDict,
    ) -> bool:
        """
        Check if an event matches the format for a given pool type.

        Args:
            event: The event arguments.

        Returns:
            bool: True if the event matches the format of a pool event, False otherwise.

        """
        pass

    def get_pools(self) -> List[BasePool]:
        """
        Get the pools of the exchange.

        Returns
        -------
        List[Pool]
            The pools of the exchange

        """
        return list(self.pools.values())

    def get_pool(self, key: str) -> BasePool:
        """

        Parameters
        ----------
        key: str - The unique key of the pool

        Returns
        -------
        Pool
            The pool object

        """
        return self.pools[key] if key in self.pools else None
