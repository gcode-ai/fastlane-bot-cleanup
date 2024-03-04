from dataclasses import dataclass, field
from typing import List, Optional

from web3.contract import Contract

from fastlane_bot.constants import CHAIN_SPECIFIC_INFO


@dataclass
class Blockchain:
    """
    Blockchain class.
    """

    name: str
    exchanges: List["BaseExchange"] = field(
        default_factory=list
    )  # Use a default_factory for mutable default
    _flashloan_tokens: List[str] = field(default_factory=list)

    def __post_init__(self):
        df = CHAIN_SPECIFIC_INFO.copy()
        if self.name not in df['blockchain'].unique():
            raise ValueError(f"Chain specific info not found for {self.name}. Please add them to the constants "
                             f"CHAIN_SPECIFIC_INFO staticdata.")
        self._chain_df = df[df['blockchain'] == self.name]
        self._flashloan_tokens = self._chain_df['flashloan_token_addresses'].values[0].split(',')

    def add_exchange(self, exchange: "BaseExchange"):
        self.exchanges.append(exchange)

    def find_exchange(self, name: str) -> Optional["BaseExchange"]:
        return next((ex for ex in self.exchanges if ex.name == name), None)

    @property
    def chain_id(self) -> str:
        return self._chain_df['chain_id'].values[0]

    @property
    def is_supported(self) -> bool:
        return self._chain_df['is_supported'].values[0]

    @property
    def chain_id(self) -> int:
        return self._chain_df['chain_id'].values[0]

    @property
    def multicall_contract_address(self) -> str:
        return self._chain_df['multicall_contract_address'].values[0]

    @property
    def fastlane_contract_address(self) -> str:
        return self._chain_df['fastlane_contract_address'].values[0]

    @property
    def max_block_fetch(self) -> int:
        return self._chain_df['max_block_fetch'].values[0]

    @property
    def flashloan_token_symbols(self) -> List[str]:
        return self._chain_df['flashloan_token_symbols'].values[0].split(',')

    @property
    def flashloan_tokens(self) -> List[str]:
        return self._flashloan_tokens

    @flashloan_tokens.setter
    def flashloan_tokens(self, value: List[str]):
        self._flashloan_tokens = value

    @property
    def flashloan_fee(self) -> float:
        return self._chain_df['flashloan_tokens_fee'].values[0]

    @property
    def stablecoin_address(self) -> str:
        return self._chain_df['stablecoin_address'].values[0]

    @property
    def native_gas_token_address(self) -> str:
        return self._chain_df['native_gas_token_address'].values[0]

    @property
    def native_gas_token_symbol(self) -> str:
        return self._chain_df['native_gas_token_symbol'].values[0]

    @property
    def wrapped_gas_token_address(self) -> str:
        return self._chain_df['wrapped_gas_token_address'].values[0]

    @property
    def wrapped_gas_token_symbol(self) -> str:
        return self._chain_df['wrapped_gas_token_symbol'].values[0]

    @property
    def balancer_vault_address(self) -> str:
        return self._chain_df['balancer_vault_address'].values[0]

    @property
    def gas_oracle_address(self) -> str:
        return self._chain_df['gas_oracle_address'].values[0]

    @property
    def gas_oracle_contract(self) -> Contract:
        return self._gas_oracle_contract

    @gas_oracle_contract.setter
    def gas_oracle_contract(self, value: Contract):
        self._gas_oracle_contract = value

    @property
    def carbon_controller_address(self) -> float:
        return self._chain_df['carbon_controller_address'].values[0]

    @property
    def carbon_controller_voucher(self) -> float:
        return self._chain_df['carbon_controller_voucher'].values[0]

    @property
    def bancor_network_info_contract_address(self) -> float:
        return self._chain_df['bancor_network_info_contract_address'].values[0]

