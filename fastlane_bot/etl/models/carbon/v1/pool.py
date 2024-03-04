# coding=utf-8
"""
Contains the pool class for Carbon v1.

(c) Copyright Bprotocol foundation 2024.
Licensed under MIT
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

from web3 import Web3
from web3.contract import AsyncContract, Contract

from fastlane_bot.etl.models.pool import BasePool


@dataclass
class CarbonV1Pool(BasePool):
    """
    Class representing a Carbon v1 pool.
    """

    @staticmethod
    def unique_key() -> str:
        """
        see base class.
        """
        return "cid"

    def update_from_event(
        self,
        event_args: Dict[str, Any],
        data: Dict[str, Any],
        router_address: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        See base class.
        """
        event_type = event_args["event"]
        assert event_type not in ["TradingFeePPMUpdated", "PairTradingFeePPMUpdated"], (
            "This event should not be " "handled by this class."
        )
        data = CarbonV1Pool.parse_event(data, event_args, event_type)
        data["router"] = (router_address,)
        for key, value in data.items():
            self.state[key] = value
        return data

    @staticmethod
    def parse_event(
        data: Dict[str, Any], event_args: Dict[str, Any], event_type: str
    ) -> Dict[str, Any]:
        """
        Parse the event args into a dict.

        Parameters
        ----------
        data : Dict[str, Any]
            The staticdata to update.
        event_args : Dict[str, Any]
            The event arguments.
        event_type : str
            The event type.

        Returns
        -------
        Dict[str, Any]
            The updated staticdata.
        """
        order0, order1 = CarbonV1Pool.parse_orders(event_args, event_type)
        data["cid"] = event_args["args"].get("id")
        if isinstance(order0, list) and isinstance(order1, list):
            data["y_0"] = order0[0]
            data["z_0"] = order0[1]
            data["A_0"] = order0[2]
            data["B_0"] = order0[3]
            data["y_1"] = order1[0]
            data["z_1"] = order1[1]
            data["A_1"] = order1[2]
            data["B_1"] = order1[3]
        else:
            data["y_0"] = order0["y"]
            data["z_0"] = order0["z"]
            data["A_0"] = order0["A"]
            data["B_0"] = order0["B"]
            data["y_1"] = order1["y"]
            data["z_1"] = order1["z"]
            data["A_1"] = order1["A"]
            data["B_1"] = order1["B"]

        return data

    @staticmethod
    def parse_orders(
        event_args: Dict[str, Any], event_type: str
    ) -> Tuple[List[int], List[int]]:
        """
        Parse the orders from the event args. If the event type is StrategyDeleted, then the orders are set to 0.

        Parameters
        ----------
        event_args : Dict[str, Any]
            The event arguments.
        event_type : str
            The event type.

        Returns
        -------
        Tuple[List[int], List[int]]
            The parsed orders.
        """
        if event_type not in ["StrategyDeleted"]:
            order0 = event_args["args"].get("order0")
            order1 = event_args["args"].get("order1")
        else:
            order0 = [0, 0, 0, 0]
            order1 = [0, 0, 0, 0]
        return order0, order1

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
        try:
            strategy = contract.caller.strategy(int(self.state["cid"]))
        except AttributeError:
            strategy = contract.functions.strategy(int(self.state["cid"])).call()

        fake_event = {
            "args": {
                "id": strategy[0],
                "order0": strategy[3][0],
                "order1": strategy[3][1],
            }
        }
        params = self.parse_event(self.state, fake_event, "None")
        params["exchange_name"] = exchange_name
        params["router"] = (router_address,)
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
        Carbon V1 uses multicall, so we don't use the async contract to get the pool balances.
        """
        return None
