from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from web3 import Web3
from web3.contract import Contract

from fastlane_bot.etl.models.carbon.v1.constants import CARBON_CONTROLLER_ADDRESS
from fastlane_bot.etl.models.carbon.v1.exchange import CarbonV1Exchange
from fastlane_bot.etl.models.carbon.v1.pool import CarbonV1Pool
from fastlane_bot.etl.models.manager import BaseManager


@dataclass
class CarbonV1AndForksManager(BaseManager):
    """
    Carbon v1 and forks manager class
    """

    carbon_initialized: bool = None

    _fee_pairs: Dict[Tuple[str, str], int] = field(default_factory=dict)

    def _init_event_contract(self):
        _abi = CarbonV1Exchange.get_abi()
        self._tenderly_event_contract = self.w3_tenderly.eth.contract(
            address=CARBON_CONTROLLER_ADDRESS,
            abi=_abi,
        )
        self._event_contract = self.w3.eth.contract(
            address=CARBON_CONTROLLER_ADDRESS,
            abi=_abi,
        )

    @staticmethod
    def pool_key_from_info(pool_info: Dict[str, Any]) -> str:
        return pool_info[CarbonV1Pool.unique_key()]

    @property
    def fee_pairs(self) -> Dict[Tuple[str, str], int]:
        return self._fee_pairs

    @fee_pairs.setter
    def fee_pairs(self, value: Dict[Tuple[str, str], int]):
        self._fee_pairs = value

    @property
    def tenderly_event_contract(self) -> Contract:
        return self._tenderly_event_contract

    @tenderly_event_contract.setter
    def tenderly_event_contract(self, value: Contract):
        self._tenderly_event_contract = value

    @property
    def event_contract(self) -> Contract:
        return self._event_contract

    @event_contract.setter
    def event_contract(self, value: Contract):
        self._event_contract = value

    def set_carbon_v1_fee_pairs(self):
        # Create or get CarbonController contract object
        carbon_controller = self.create_or_get_carbon_controller()

        # Get pairs by contract
        pairs = self.get_carbon_pairs(carbon_controller)

        # Get the fee for each pair
        fee_pairs = self.get_fee_pairs(pairs, carbon_controller)

        # Set the fee pairs
        self.exchanges["carbon_v1"].fee_pairs = fee_pairs

    def update_from_event(self, event: Dict[str, Any]) -> None:
        if event["event"] == "TradingFeePPMUpdated":
            self.handle_trading_fee_updated()
            return

        if event["event"] == "PairTradingFeePPMUpdated":
            self.handle_pair_trading_fee_updated(event)
            return

        if event["event"] == "PairCreated":
            self.handle_pair_created(event)
            return

        if event["event"] == "StrategyDeleted":
            self.handle_strategy_deleted(event)
            return

        return self._update_from_event(event)

    def handle_trading_fee_updated(self):
        # Create or get CarbonController contract object
        carbon_controller = self.create_or_get_carbon_controller()

        # Get pairs by state
        pairs = self.get_carbon_pairs(carbon_controller)

        # Update fee pairs
        self.fee_pairs = self.get_fee_pairs(pairs, carbon_controller)

        # Update pool info
        for pool in self.pool_data:
            if pool["exchange_name"] == "carbon_v1":
                pool["fee"] = self.fee_pairs[
                    (pool["tkn0_address"], pool["tkn1_address"])
                ]
                pool["fee_float"] = pool["fee"] / 1e6
                pool["descr"] = self.pool_descr_from_info(pool)

    def handle_pair_trading_fee_updated(
        self,
        event: Dict[str, Any] = None,
    ):
        tkn0_address = event["args"]["token0"]
        tkn1_address = event["args"]["token1"]
        fee = event["args"]["newFeePPM"]

        self.fee_pairs[(tkn0_address, tkn1_address)] = fee

        for idx, pool in enumerate(self.pool_data):
            if (
                pool["tkn0_address"] == tkn0_address
                and pool["tkn1_address"] == tkn1_address
                and pool["exchange_name"] == "carbon_v1"
            ):
                self._handle_pair_trading_fee_updated(fee, pool, idx)
            elif (
                pool["tkn0_address"] == tkn1_address
                and pool["tkn1_address"] == tkn0_address
                and pool["exchange_name"] == "carbon_v1"
            ):
                self._handle_pair_trading_fee_updated(fee, pool, idx)

    def handle_pair_created(self, event: Dict[str, Any]):
        fee_pairs = self.get_fee_pairs(
            [(event["args"]["token0"], event["args"]["token1"], 0, 5000)],
            self.create_or_get_carbon_controller(),
        )
        self.fee_pairs.update(fee_pairs)

    def handle_strategy_deleted(self, event: Dict[str, Any]) -> None:
        cid = event["args"]["id"]
        self.pool_data = [p for p in self.pool_data if p["cid"] != cid]
        self.exchanges["carbon_v1"].delete_strategy(event["args"]["id"])

    def get_key_and_value(
        self, event: Dict[str, Any], addr: str, ex_name: str
    ) -> Tuple[str, Any]:
        return "cid", event["args"]["id"]

    def create_or_get_carbon_controller(self):
        """
        Create or get the CarbonController contract object.

        Returns
        -------
        carbon_controller : Contract
            The CarbonController contract object.

        """
        if (
            CARBON_CONTROLLER_SELECTOR in self.pool_contracts["carbon_v1"]
            and not self.replay_from_block
        ):
            return self.pool_contracts["carbon_v1"][CARBON_CONTROLLER_ADDRESS]

        # Create a CarbonController contract object
        carbon_controller = self.cfg.w3.eth.contract(
            address=CARBON_CONTROLLER_ADDRESS,
            abi=self.exchanges["carbon_v1"].get_abi(),
        )

        # Store the contract object in pool_contracts
        self.pool_contracts["carbon_v1"][CARBON_CONTROLLER_ADDRESS] = carbon_controller
        return carbon_controller

    def get_carbon_pairs_by_state(self) -> List[Tuple[str, str]]:
        """
        Get the carbon pairs by state.

        Returns
        -------
        List[Tuple[str, str]]
            The carbon pairs.

        """
        return [
            (p["tkn0_address"], p["tkn1_address"])
            for p in self.pool_data
            if p["exchange_name"] == "carbon_v1"
        ]

    @staticmethod
    def get_carbon_pairs_by_contract(
        carbon_controller: Contract, replay_from_block: int or str = None
    ) -> List[Tuple[str, str]]:
        """
        Get the carbon pairs by contract.

        Parameters
        ----------
        carbon_controller : Contract
            The CarbonController contract object.
        replay_from_block : int or str, optional
            The block number to replay from, by default 'latest'

        Returns
        -------
        List[Tuple[str, str]]
            The carbon pairs.

        """
        return [
            (second, first)
            for first, second in carbon_controller.functions.pairs().call(
                block_identifier=replay_from_block or "latest"
            )
        ]

    def get_carbon_pairs(
        self, carbon_controller: Contract, target_tokens: List[str] = None
    ) -> List[Tuple[str, str, int, int]]:
        """
        Get the carbon pairs.

        Parameters
        ----------
        carbon_controller : Contract
            The CarbonController contract object.
        target_tokens : List[str], optional
            The target tokens, by default None

        Returns
        -------
        List[Tuple[str, str, int, int]]
            The carbon pairs.

        """
        pairs = (
            self.get_carbon_pairs_by_state()
            if self.carbon_initialized
            else self.get_carbon_pairs_by_contract(carbon_controller)
        )
        # Log whether the carbon pairs were retrieved from the state or the contract
        self.cfg.logger.debug(
            f"Retrieved {len(pairs)} carbon pairs from {'state' if self.carbon_initialized else 'contract'}"
        )
        if target_tokens is None or target_tokens == []:
            target_tokens = []
            for pair in pairs:
                if pair[0] not in target_tokens:
                    target_tokens.append(pair[0])
                if pair[1] not in target_tokens:
                    target_tokens.append(pair[1])
        return [
            (pair[0], pair[1], 0, 5000)
            for pair in pairs
            if pair[0] in target_tokens and pair[1] in target_tokens
        ]

    def get_fees_by_pair(
        self, all_pairs: List[Tuple[str, str]], carbon_controller: Contract
    ):
        """
        Get the fees by pair.

        Parameters
        ----------
        all_pairs : List[Tuple[str, str]]
            The pairs.
        carbon_controller : Contract
            The carbon controller contract object.

        Returns
        -------
        List[int]
            The fees by pair.

        """
        multicaller = MultiCaller(
            contract=carbon_controller,
            block_identifier=self.replay_from_block or "latest",
            multicall_address=MULTICALL_CONTRACT_ADDRESS,
            web3=self.web3,
        )

        with multicaller as mc:
            for pair in all_pairs:
                mc.add_call(
                    carbon_controller.functions.pairTradingFeePPM, pair[0], pair[1]
                )

        return multicaller.multicall()

    def get_fee_pairs(
        self, all_pairs: List[Tuple[str, str, int, int]], carbon_controller: Contract
    ) -> Dict[Tuple[str, str], int]:
        """
        Get the fees for each pair and store in a dictionary.

        Parameters
        ----------
        all_pairs : List[Tuple]
            A list of pairs.
        carbon_controller : Contract
            The CarbonController contract object.

        Returns
        -------
        Dict[Tuple[str, str], int]
            A dictionary of fees for each pair.
        """
        # Get the fees for each pair and store in a dictionary
        fees_by_pair = self.get_fees_by_pair(all_pairs, carbon_controller)
        fee_pairs = {
            (
                self.web3.to_checksum_address(pair[0]),
                self.web3.to_checksum_address(pair[1]),
            ): fee
            for pair, fee in zip(all_pairs, fees_by_pair)
        }
        # Add the reverse pair to the fee_pairs dictionary
        fee_pairs.update(
            {
                (
                    self.web3.to_checksum_address(pair[1]),
                    self.web3.to_checksum_address(pair[0]),
                ): fee
                for pair, fee in zip(all_pairs, fees_by_pair)
            }
        )
        return fee_pairs

    def _handle_pair_trading_fee_updated(
        self, fee: int, pool: Dict[str, Any], idx: int
    ):
        """
        Handle the pair trading fee updated event by updating the fee pairs and pool info for the given pair.

        Parameters
        ----------
        fee : int
            The fee.
        pool : Dict[str, Any]
            The pool.
        idx : int
            The index of the pool.

        """
        pool["fee"] = f"{fee}"
        pool["fee_float"] = fee / 1e6
        pool["descr"] = self.pool_descr_from_info(pool)
        self.pool_data[idx] = pool
