# coding=utf-8
"""
Contains the exchange class for BancorV3. This class is responsible for handling BancorV3 events and updating the state of the pools.

(c) Copyright Bprotocol foundation 2023.
Licensed under MIT
"""
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import web3
from web3 import Web3
from web3.contract import AsyncContract, Contract
import web3.exceptions

from _decimal import Decimal

from fastlane_bot import constants
from fastlane_bot.constants import BNT_ADDRESS, ETH_ADDRESS
from fastlane_bot.etl.models.bancor.pol import abi
from fastlane_bot.etl.models.exchanges.pool import BasePool


@dataclass
class BancorPolPool(BasePool):
    """
    Class representing a Bancor protocol-owned liquidity pool.
    """

    @staticmethod
    def unique_key() -> str:
        """
        see base class.
        """
        return "token"

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
        if event_type in "TradingEnabled":
            data["tkn0_address"] = event_args["args"]["token"]
            data["tkn1_address"] = (
                constants.ETH_ADDRESS
                if event_args["args"]["token"] not in ETH_ADDRESS
                else BNT_ADDRESS
            )

        if event_args["args"]["token"] == self.state["tkn0_address"] and event_type in [
            "TokenTraded"
        ]:
            # *** Balance now updated from multicall ***
            pass

        for key, value in data.items():
            self.state[key] = value

        data["cid"] = self.state["cid"]
        data["fee"] = 0
        data["fee_float"] = 0
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
        tkn0 = self.state["tkn0_address"]

        # Use ERC20 token contract to get balance of POL contract
        p0 = 0
        p1 = 0

        tkn_balance = self.get_erc20_tkn_balance(contract, tkn0, w3_tenderly, w3)

        if tenderly_fork_id and "bancor_pol" in tenderly_exchanges:
            contract = w3_tenderly.eth.contract(
                abi=abi.BANCOR_POL_ABI, address=contract.address
            )

        try:
            p0, p1 = contract.functions.tokenPrice(tkn0).call()
        except web3.exceptions.BadFunctionCallOutput:
            print(f"BadFunctionCallOutput: {tkn0}")

        token_price = Decimal(p1) / Decimal(p0)

        params = {
            "fee": "0.000",
            "fee_float": 0.000,
            "tkn0_balance": 0,
            "tkn1_balance": 0,
            "exchange_name": self.state["exchange_name"],
            "address": self.state["address"],
            "y_0": tkn_balance,
            "z_0": tkn_balance,
            "A_0": 0,
            "B_0": int(str(self.encode_token_price(token_price))),
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
        See base class. Not implemented for Bancor POL.
        """
        return None

    @staticmethod
    def get_erc20_tkn_balance(
        contract: Contract, tkn0: str, w3_tenderly: Web3 = None, w3: Web3 = None
    ) -> int:
        """
        Get the ERC20 token balance of the POL contract

        Parameters
        ----------
        contract: Contract
            The contract object
        tkn0: str
            The token address
        w3_tenderly: Web3
            The tenderly web3 object
        w3: Web3
            The web3 object

        Returns
        -------
        int
            The token balance

        """
        if w3_tenderly:
            contract = w3_tenderly.eth.contract(
                abi=abi.BANCOR_POL_ABI, address=contract.address
            )
        try:
            return contract.caller.amountAvailableForTrading(tkn0)
        except web3.exceptions.ContractLogicError:
            if w3_tenderly:
                erc20_contract = w3_tenderly.eth.contract(
                    abi=abi.pkg.ERC20_ABI, address=tkn0
                )
            else:
                erc20_contract = w3.eth.contract(abi=abi.pkg.ERC20_ABI, address=tkn0)
            return erc20_contract.functions.balanceOf(contract.address).call()

    @staticmethod
    def bitLength(value):
        return len(bin(value).lstrip("0b")) if value > 0 else 0

    def encodeFloat(self, value):
        exponent = self.bitLength(value // constants.pkg.ONE)
        mantissa = value >> exponent
        return mantissa | (exponent * constants.pkg.ONE)

    def encodeRate(self, value):
        data = int(value.sqrt() * constants.pkg.ONE)
        length = self.bitLength(data // constants.pkg.ONE)
        return (data >> length) << length

    def encode_token_price(self, price: Decimal) -> int:
        """
        Encode the token price.

        Args:
            price (Decimal): The price.

        Returns:
            int: The encoded price.

        """
        return self.encodeFloat(self.encodeRate((price)))

    def update_erc20_balance(
        self, token_contract: Contract, address: str
    ) -> Dict[str, Any]:
        """
        Update the ERC20 token balance.

        Args:
            token_contract: Contract
                The token contract.
            address: str
                The address.
        """

        balance = token_contract.caller.balanceOf(address)
        return {
            "y_0": balance,
            "z_0": balance,
        }
