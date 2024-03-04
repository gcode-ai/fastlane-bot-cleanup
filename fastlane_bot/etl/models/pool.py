# coding=utf-8
"""
Contains the base class for all pools.

(c) Copyright Bprotocol foundation 2024.
Licensed under MIT
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional

from web3 import Web3
from web3.contract import AsyncContract, Contract


@dataclass
class BasePool(ABC):
    """
    Abstract base class representing a pool.

    Attributes
    ----------
    state : Dict[str, Any]
        The pool state.
    """

    state: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    @abstractmethod
    def unique_key() -> str:
        """
        Returns the unique key for the pool.
        """
        pass

    @abstractmethod
    def update_from_event(
        self,
        event_args: Dict[str, Any],
        data: Dict[str, Any],
        router_address: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Update the pool state from an event.

        Parameters
        ----------
        event_args : Dict[str, Any]
            The event arguments.
        data : Dict[str, Any]
            The pool staticdata.
        router_address : str, optional
            The router address, by default None

        Returns
        -------
        Dict[str, Any]
            The updated pool staticdata.

        Args:
            router_address ():
        """
        pass

    @abstractmethod
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
        Update the pool state from a contract.

        Parameters
        ----------
        contract : Contract
            The contract.
        tenderly_fork_id : str, optional
            The tenderly fork id, by default None
        w3_tenderly : Web3, optional
            The tenderly web3 instance, by default None
        w3 : Web3, optional
            The web3 instance, by default None
        tenderly_exchanges : List[str], optional
            The tenderly exchanges, by default None
        exchange_name : str, optional
            The exchange name, by default None
        router_address : str, optional
            The router address, by default None

        Returns
        -------
        Dict[str, Any]
            The updated pool staticdata.
        """
        pass

    @abstractmethod
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
        Update the pool state from a contract.
        Args:
            contract ():
            tenderly_fork_id ():
            w3_tenderly ():
            w3 ():
            tenderly_exchanges ():
            exchange_name ():
            router_address ():
        """
        pass

    @staticmethod
    def get_common_data(
        event: Dict[str, Any], pool_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get common (common to all Pool child classes) staticdata from an event and pool info.

        Args:
            event (Dict[str, Any]): The event staticdata.
            pool_info (Dict[str, Any]): The pool information.

        Returns:
            Dict[str, Any]: A dictionary containing common staticdata extracted from the event and pool info.
        """
        return {
            "last_updated_block": event["blockNumber"],
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "pair_name": pool_info["pair_name"],
            "descr": pool_info["descr"],
            "address": event["address"],
        }
