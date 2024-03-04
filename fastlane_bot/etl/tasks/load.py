# coding=utf-8
"""
Contains the interface for querying data from the data fetcher module.

(c) Copyright Bprotocol foundation 2023.
Licensed under MIT
"""
from dataclasses import dataclass, field
from typing import List, Any, Dict, Optional

import fastlane_bot
from fastlane_bot.etl.models.exchange import BaseExchange
from fastlane_bot.etl.models.token import Token
from fastlane_bot.etl.tasks.base import BaseTask
from fastlane_bot.etl.tasks.transform import TransformTask
from fastlane_bot.helpers import Univ3Calculator
from fastlane_bot.tools.cpc import CPCContainer, ConstantProductCurve as CPC


@dataclass
class Pool(TransformTask):
    """
    This class represents a [transformed] pool. It is used to store pool data and tokens.
    """

    pass


@dataclass
class LoadTask(BaseTask):
    """
    Interface for querying data from the data fetcher module. These methods mirror the existing methods
    expected/used in the bot module. The implementation of these methods should allow for the bot module
    to be used with the new data fetcher module without any changes to the bot module.
    """

    @property
    def state(self) -> List[Dict[str, Any]]:
        """
        Get the state. This method returns the state of the pools.

        Returns
        -------
        List[Dict[str, Any]]
            The state of the pools

        """
        return self.mgr.pool_data

    def filter_target_tokens(self, target_tokens: List[str]):
        """
        Filter the pools to only include pools that are in the target pools list

        Parameters
        ----------
        target_tokens: List[str]
            The list of tokens to filter pools by. Pools must contain both tokens in the list to be included.
        """
        initial_state = self.state.copy()
        self.state = [
            pool
            for pool in self.state
            if pool["tkn0_address"] in target_tokens
            and pool["tkn1_address"] in target_tokens
        ]

        self.mgr.logger.info(
            f"[events.interface] Limiting pools by target_tokens. Removed {len(initial_state) - len(self.state)} non target-pools. {len(self.state)} pools remaining"
        )

        # Log the total number of pools remaining for each exchange
        self.mgr.logger.debug("Pools remaining per exchange:")
        for ex in self.mgr.exchanges:
            pools = self.filter_pools(ex.name)
            self.log_pool_numbers(pools, ex.name)

    def remove_unsupported_exchanges(self) -> None:
        initial_state = self.state.copy()
        self.state = [
            pool for pool in self.state if pool["exchange_name"] in self.mgr.exchanges
        ]
        self.mgr.logger.debug(
            f"Removed {len(initial_state) - len(self.state)} unsupported exchanges. {len(self.state)} pools remaining"
        )

        # Log the total number of pools remaining for each exchange
        self.mgr.logger.debug("Pools remaining per exchange:")
        for ex in self.mgr.exchanges:
            pools = self.filter_pools(ex.name)
            self.log_pool_numbers(pools, ex.name)

    def has_balance(self, pool: Dict[str, Any], keys: List[str]) -> bool:
        """
        Check if a pool has a balance for a given key

        Parameters
        ----------
        pool: Dict[str, Any]
            The pool to check
        keys: List[str]
            The keys to check for a balance

        Returns
        -------
        bool
            True if the pool has a balance for the given key, False otherwise

        """

        return any(key in pool and pool[key] > 0 for key in keys)

    def get_tokens_from_exchange(self, exchange_name: str) -> List[str]:
        """
        This token gets all tokens that exist in pools on the specified exchange.
        Parameters
        ----------
        exchange_name: str
            The exchange from which to get tokens.

        Returns
        -------
        list[str]
            Returns a list of token keys.
        """
        pools = self.filter_pools(exchange_name=exchange_name)
        tokens = []
        for pool in pools:
            for idx in range(8):
                try:
                    tkn = pool[f"tkn{idx}_address"]
                    if type(tkn) == str:
                        tokens.append(tkn)
                except KeyError:
                    # Out of bounds
                    break
        return list(set(tokens))

    def filter_pools(
        self, exchange_name: str, keys: List[str] = ""
    ) -> List[Dict[str, Any]]:
        """
        Filter pools by exchange name and key

        Parameters
        ----------
        exchange_name: str
            The exchange name to filter by
        keys: str
            The key to filter by

        Returns
        -------
        List[Dict[str, Any]]
            The filtered pools
        """
        if keys:
            return [
                pool
                for pool in self.state
                if pool["exchange_name"] == exchange_name
                and self.has_balance(pool, keys)
                and pool["tkn0_decimals"] is not None
                and pool["tkn1_decimals"] is not None
            ]
        else:
            return [
                pool for pool in self.state if pool["exchange_name"] == exchange_name
            ]

    def log_pool_numbers(self, pools: List[Dict[str, Any]], exchange_name: str) -> None:
        """
        Log the number of pools for a given exchange name

        Parameters
        ----------
        pools: List[Dict[str, Any]]
            The pools to log
        exchange_name: str
            The exchange name to log

        """
        self.mgr.logger.debug(f"[events.interface] {exchange_name}: {len(pools)}")

    def remove_zero_liquidity_pools(self) -> None:
        """
        # TODO: Move to exchange-specific Manager classes
        Remove pools with zero liquidity.
        """
        initial_state = self.state.copy()

        exchanges = []
        keys = []

        for ex in self.mgr.exchanges:
            if ex in self.mgr.uniswap_v2_forks + self.mgr.solidly_v2_forks + [
                "bancor_v2",
                "bancor_v3",
            ]:
                exchanges.append(ex)
                keys.append(["tkn0_balance"])
            elif ex in self.mgr.uniswap_v3_forks:
                exchanges.append(ex)
                keys.append(["liquidity"])
            elif ex in self.mgr.carbon_v1_forks:
                exchanges.append(ex)
                keys.append(["y_0", "y_1"])
            elif ex in "bancor_pol":
                exchanges.append(ex)
                keys.append(["y_0"])
            elif ex in "balancer":
                exchanges.append(ex)
                keys.append(["tkn0_balance"])

        self.state = [
            pool
            for exchange, key in zip(exchanges, keys)
            for pool in self.filter_pools(exchange, key)
        ]

        for exchange in exchanges:
            self.log_pool_numbers(
                [pool for pool in self.state if pool["exchange_name"] == exchange.name],
                exchange.name,
            )

        zero_liquidity_pools = [
            pool for pool in initial_state if pool not in self.state
        ]

        for exchange in exchanges:
            self.log_pool_numbers(
                [
                    pool
                    for pool in zero_liquidity_pools
                    if pool["exchange_name"] == exchange
                ],
                f"{exchange}_zero_liquidity_pools",
            )

    def remove_unmapped_uniswap_v2_pools(self) -> None:
        """
        Remove unmapped uniswap_v2 pools
        """
        initial_state = self.state.copy()
        self.state = [
            pool
            for pool in self.state
            if pool["exchange_name"] != "uniswap_v2"
            or (
                pool["exchange_name"] in self.mgr.uniswap_v2_forks
                and pool["address"] in self.mgr.uniswap_v2_event_mappings
            )
        ]
        self.mgr.logger.debug(
            f"Removed {len(initial_state) - len(self.state)} unmapped uniswap_v2/sushi pools. {len(self.state)} uniswap_v2/sushi pools remaining"
        )
        self.log_umapped_pools_by_exchange(initial_state)

    def remove_unmapped_uniswap_v3_pools(self) -> None:
        """
        Remove unmapped uniswap_v2 pools
        """
        initial_state = self.state.copy()
        self.state = [
            pool
            for pool in self.state
            if pool["exchange_name"] != "uniswap_v3"
            or (
                pool["exchange_name"] in self.mgr.uniswap_v3_forks
                and pool["address"] in self.uniswap_v3_event_mappings
            )
        ]
        self.mgr.logger.debug(
            f"Removed {len(initial_state) - len(self.state)} unmapped uniswap_v2/sushi pools. {len(self.state)} uniswap_v2/sushi pools remaining"
        )
        self.log_umapped_pools_by_exchange(initial_state)

    def log_umapped_pools_by_exchange(self, initial_state):
        # Log the total number of pools filtered out for each exchange
        self.mgr.logger.debug("Unmapped uniswap_v2/sushi pools:")
        unmapped_pools = [pool for pool in initial_state if pool not in self.state]
        assert len(unmapped_pools) == len(initial_state) - len(self.state)
        uniswap_v2_unmapped = [
            pool for pool in unmapped_pools if pool["exchange_name"] == "uniswap_v2"
        ]
        self.log_pool_numbers(uniswap_v2_unmapped, "uniswap_v2")
        sushiswap_v2_unmapped = [
            pool for pool in unmapped_pools if pool["exchange_name"] == "sushiswap_v2"
        ]
        self.log_pool_numbers(sushiswap_v2_unmapped, "sushiswap_v2")

    def remove_faulty_token_pools(self) -> None:
        """
        Remove pools with faulty tokens
        """
        self.mgr.logger.debug(
            f"Total number of pools. {len(self.state)} before removing faulty token pools"
        )

        safe_pools = []
        for pool in self.state:
            self.mgr.logger.info(pool)
            try:
                self.get_token(pool["tkn0_address"])
                self.get_token(pool["tkn1_address"])
                safe_pools.append(pool)
            except Exception as e:
                self.mgr.logger.warning(f"[events.interface] Exception: {e}")
                self.mgr.logger.warning(
                    f"Removing pool for exchange={pool['pair_name']}, pair_name={pool['pair_name']} token={pool['tkn0_key']} from state for faulty token"
                )

        self.state = safe_pools

    def update_state(self, state: List[Dict[str, Any]]) -> None:
        """
        Update the state.

        Args:
            state (List[Dict[str, Any]]): The new state.

        """
        self.state = state.copy()
        if self.state == state:
            self.mgr.logger.warning("WARNING: State not updated")

    def drop_all_tables(self) -> None:
        """
        Drop all tables. Deprecated.
        """
        raise DeprecationWarning("Method not implemented")

    def get_pool_data_with_tokens(self) -> List[Pool]:
        """
        Get pool data with tokens as a List
        """
        if self.pool_data_list is None:
            self.refresh_pool_data()
        return self.pool_data_list

    def get_pool_data_lookup(self) -> Dict[str, Pool]:
        """
        Get pool data with tokens as a Dict to find specific pools
        """
        if self.pool_data is None:
            self.refresh_pool_data()
        return self.pool_data

    def refresh_pool_data(self):
        """
        Refreshes pool data to ensure it is up-to-date
        """
        self.pool_data_list = [
            self.create_pool_and_tokens(idx, record)
            for idx, record in enumerate(self.state)
        ]
        self.pool_data = {str(pool.cid): pool for pool in self.pool_data_list}

    def create_pool_and_tokens(self, idx: int, record: Dict[str, Any]) -> Pool:
        """
        Create a pool and tokens object from a record
        """
        result = Pool(
            id=idx,
            **{
                key: record.get(key)
                for key in [
                    "cid",
                    "last_updated",
                    "last_updated_block",
                    "descr",
                    "pair_name",
                    "exchange_name",
                    "fee",
                    "fee_float",
                    "tkn0_balance",
                    "tkn1_balance",
                    "z_0",
                    "y_0",
                    "A_0",
                    "B_0",
                    "z_1",
                    "y_1",
                    "A_1",
                    "B_1",
                    "sqrt_price_q96",
                    "tick",
                    "tick_spacing",
                    "liquidity",
                    "address",
                    "anchor",
                    "tkn0",
                    "tkn1",
                    "tkn0_address",
                    "tkn0_decimals",
                    "tkn1_address",
                    "tkn1_decimals",
                    "tkn0_weight",
                    "tkn1_weight",
                    "tkn2",
                    "tkn2_balance",
                    "tkn2_address",
                    "tkn2_decimals",
                    "tkn2_weight",
                    "tkn3",
                    "tkn3_balance",
                    "tkn3_address",
                    "tkn3_decimals",
                    "tkn3_weight",
                    "tkn4",
                    "tkn4_balance",
                    "tkn4_address",
                    "tkn4_decimals",
                    "tkn4_weight",
                    "tkn5",
                    "tkn5_balance",
                    "tkn5_address",
                    "tkn5_decimals",
                    "tkn5_weight",
                    "tkn6",
                    "tkn6_balance",
                    "tkn6_address",
                    "tkn6_decimals",
                    "tkn6_weight",
                    "tkn7",
                    "tkn7_balance",
                    "tkn7_address",
                    "tkn7_decimals",
                    "tkn7_weight",
                    "pool_type",
                ]
            },
        )
        result.tkn0 = result.pair_name.split("/")[0].split("-")[0]
        result.tkn1 = result.pair_name.split("/")[1].split("-")[0]
        result.tkn0_address = result.pair_name.split("/")[0]
        result.tkn1_address = result.pair_name.split("/")[1]
        return result

    # def get_tokens(self) -> List[Token]:
    #     """
    #     Get tokens. This method returns a list of tokens that are in the state.
    #     """
    #     token_set = set()
    #     for record in self.state:
    #         for idx in range(len(record["descr"].split("/"))):
    #             try:
    #                 token_set.add(self.create_token(record, f"tkn{str(idx)}_"))
    #             except AttributeError:
    #                 pass
    #     if self.mgr.is_gas_token_in_flashloan_tokens:
    #         token_set.add(
    #             Token(
    #                 symbol=self.blockchain.NATIVE_GAS_TOKEN_SYMBOL,
    #                 address=self.blockchain.NATIVE_GAS_TOKEN_ADDRESS,
    #                 decimals=18,  # TODO: This should not be hardcoded
    #             )
    #         )
    #         token_set.add(
    #             Token(
    #                 symbol=self.blockchain.WRAPPED_GAS_TOKEN_SYMBOL,
    #                 address=self.blockchain.WRAPPED_GAS_TOKEN_ADDRESS,
    #                 decimals=18,  # TODO: This should not be hardcoded
    #             )
    #         )
    #     return list(token_set)
    #
    # def populate_tokens(self):
    #     """
    #     Populate the token Dict with tokens using the available pool data.
    #     """
    #     self.token_list = {}
    #     for record in self.state:
    #         for idx in range(len(record["descr"].split("/"))):
    #             try:
    #                 token = self.create_token(record, f"tkn{str(idx)}_")
    #                 self.token_list[token.address] = token
    #             except AttributeError:
    #                 pass
    #
    #     if self.mgr.is_gas_token_in_flashloan_tokens:
    #         native_gas_tkn = Token(
    #             symbol=self.blockchain.NATIVE_GAS_TOKEN_SYMBOL,
    #             address=self.blockchain.NATIVE_GAS_TOKEN_ADDRESS,
    #             decimals=18,  # TODO: This should not be hardcoded
    #         )
    #         wrapped_gas_tkn = Token(
    #             symbol=self.blockchain.WRAPPED_GAS_TOKEN_SYMBOL,
    #             address=self.blockchain.WRAPPED_GAS_TOKEN_ADDRESS,
    #             decimals=18,  # TODO: This should not be hardcoded
    #         )
    #         self.token_list[native_gas_tkn.address] = native_gas_tkn
    #         self.token_list[wrapped_gas_tkn.address] = wrapped_gas_tkn

    def _add_gas_tokens(self, token_collection):
        """Add gas tokens to the collection."""
        # Handling based on the type of the collection (set or dict)
        if isinstance(token_collection, set):
            token_collection.update(
                [
                    Token(
                        symbol=self.blockchain.NATIVE_GAS_TOKEN_SYMBOL,
                        address=self.blockchain.NATIVE_GAS_TOKEN_ADDRESS,
                        decimals=18,
                    ),  # TODO: This should not be hardcoded
                    Token(
                        symbol=self.blockchain.WRAPPED_GAS_TOKEN_SYMBOL,
                        address=self.blockchain.WRAPPED_GAS_TOKEN_ADDRESS,
                        decimals=18,
                    ),  # TODO: This should not be hardcoded
                ]
            )
        elif isinstance(token_collection, dict):
            native_gas_tkn = Token(
                symbol=self.blockchain.NATIVE_GAS_TOKEN_SYMBOL,
                address=self.blockchain.NATIVE_GAS_TOKEN_ADDRESS,
                decimals=18,
            )  # TODO: This should not be hardcoded
            wrapped_gas_tkn = Token(
                symbol=self.blockchain.WRAPPED_GAS_TOKEN_SYMBOL,
                address=self.blockchain.WRAPPED_GAS_TOKEN_ADDRESS,
                decimals=18,
            )  # TODO: This should not be hardcoded
            token_collection[native_gas_tkn.address] = native_gas_tkn
            token_collection[wrapped_gas_tkn.address] = wrapped_gas_tkn

    def _create_tokens_from_state(self, token_collection):
        """Create tokens from state and add them to the collection."""
        for record in self.state:
            for idx, token_descr in enumerate(record["descr"].split("/")):
                try:
                    token = self.create_token(record, f"tkn{str(idx)}_")
                    if isinstance(token_collection, set):
                        token_collection.add(token)
                    elif isinstance(token_collection, dict):
                        token_collection[token.address] = token
                except AttributeError:
                    pass

    def get_tokens(self) -> List[Token]:
        """Get tokens. This method returns a list of tokens that are in the state."""
        token_set = set()
        self._create_tokens_from_state(token_set)
        if self.mgr.is_gas_token_in_flashloan_tokens:
            self._add_gas_tokens(token_set)
        return list(token_set)

    def populate_tokens(self):
        """Populate the token Dict with tokens using the available pool data."""
        self.token_list = {}
        self._create_tokens_from_state(self.token_list)
        if self.mgr.is_gas_token_in_flashloan_tokens:
            self._add_gas_tokens(self.token_list)

    def create_token(self, record: Dict[str, Any], prefix: str) -> Token:
        """
        Create a token from a record

        Parameters
        ----------
        record: Dict[str, Any]
            The record
        prefix: str
            The prefix of the token

        Returns
        -------
        Token
            The token

        """
        return Token(
            symbol=record.get(f"{prefix}symbol"),
            decimals=record.get(f"{prefix}decimals"),
            address=record.get(f"{prefix}address"),
        )

    def get_bnt_price_from_tokens(self, price: float, tkn: Token) -> float:
        """
        Get the BNT price from tokens

        Parameters
        ----------
        price: float
            The price
        tkn: Token
            The token

        Returns
        -------
        float
            The BNT price

        """
        raise DeprecationWarning("Method not implemented")

    def get_token(self, tkn_address: str) -> Optional[Token]:
        """
        Get a token from the state

        Parameters
        ----------
        tkn_address: str
            The token address

        Returns
        -------
        Optional[Token]
            The token

        """
        if self.token_list is None:
            self.populate_tokens()
        try:
            return self.token_list.get(tkn_address)
        except KeyError:
            try:
                self.populate_tokens()
                return self.token_list.get(tkn_address)
            except KeyError as e:
                self.mgr.logger.info(
                    f"[interface.py get_token] Could not find token: {tkn_address} in token_list"
                )
                tokens = self.get_tokens()
                if tkn_address.startswith("0x"):
                    return next(
                        (tkn for tkn in tokens if tkn.address == tkn_address), None
                    )
                else:
                    raise ValueError(f"[get_token] Invalid token: {tkn_address}")

    def get_pool(self, **kwargs) -> Optional[Pool]:
        """
        Get a pool from the state

        Parameters
        ----------
        kwargs: Dict[str, Any]
            The pool parameters

        Returns
        -------
        Pool
            The pool

        """
        pool_data_with_tokens = self.get_pool_data_lookup()
        if "cid" in kwargs:
            cid = str(kwargs["cid"])
            try:
                return pool_data_with_tokens[cid]
            except KeyError:
                # pool not in data
                self.mgr.logger.error(
                    f"[interface.py get_pool] pool with cid: {cid} not in data"
                )
                return None
        else:
            try:
                return next(
                    (
                        pool
                        for pool in pool_data_with_tokens
                        if all(getattr(pool, key) == kwargs[key] for key in kwargs)
                    ),
                    None,
                )
            except AttributeError:
                return None

    def get_pools(self) -> List[Pool]:
        """
        Get all pools from the state

        Returns
        -------
        List[Pool]
            The list of pools

        """
        return self.get_pool_data_with_tokens()

    def update_recently_traded_pools(self, cids: List[str]):
        """
        Update recently traded pools. Deprecated.

        Parameters
        ----------
        cids: List[str]
            The list of cids

        """
        raise DeprecationWarning("Method not implemented")

    def run(self, args: Any):
        self.refresh_pool_data()
        pools_and_tokens = self.get_pool_data_with_tokens()
        tokens = self.get_tokens()
        ADDRDEC = {t.address: (t.address, int(t.decimals)) for t in tokens}
        curves = []
        for p in pools_and_tokens:
            try:
                p.ADDRDEC = ADDRDEC
                curves += p.run()
            except NotImplementedError as e:
                # Currently not supporting Solidly V2 Stable pools. This will be removed when support is added,
                # but for now the error message is suppressed.
                if "Stable Solidly V2" in str(e):
                    continue
                else:
                    self.mgr.logger.error(
                        f"[bot.get_curves] Pool type not yet supported, error: {e}\n"
                    )
            except ZeroDivisionError as e:
                self.mgr.logger.error(
                    f"[bot.get_curves] MUST FIX INVALID CURVE {p} [{e}]\n"
                )
            except CPC.CPCValidationError as e:
                self.mgr.logger.error(
                    f"[bot.get_curves] MUST FIX INVALID CURVE {p} [{e}]\n"
                )
            except TypeError as e:
                if fastlane_bot.__version__ not in ["3.0.31", "3.0.32"]:
                    self.mgr.logger.error(
                        f"[bot.get_curves] MUST FIX DECIMAL ERROR CURVE {p} [{e}]\n"
                    )
            except p.DoubleInvalidCurveError as e:
                self.mgr.logger.error(
                    f"[bot.get_curves] MUST FIX DOUBLE INVALID CURVE {p} [{e}]\n"
                )
            except Univ3Calculator.DecimalsMissingError as e:
                self.mgr.logger.error(
                    f"[bot.get_curves] MUST FIX DECIMALS MISSING [{e}]\n"
                )
            except Exception as e:
                self.mgr.logger.error(
                    f"[bot.get_curves] error converting pool to curve {p}\n[ERR={e}]\n\n"
                )

        return CPCContainer(curves)


@dataclass
class QueryInterface(LoadTask):
    """
    This class represents the interface for querying data from the data fetcher module. These methods mirror the
    existing methods
    """

    pass