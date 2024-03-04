import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, field
from decimal import Decimal
from glob import glob
from typing import Any, Callable, Dict, Hashable, List, Optional, Tuple, Type

import nest_asyncio
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from web3 import AsyncWeb3, Web3
from web3.contract import AsyncContract
from web3.datastructures import AttributeDict

from fastlane_bot.abi import ERC20_ABI
from fastlane_bot.constants import (
    CommonEthereumTokens,
    MULTICALLABLE_EXCHANGES,
    ONE,
    TOKENS_PATH,
)
from fastlane_bot.etl.models.carbon.v1.pool import CarbonV1Pool
from fastlane_bot.etl.models.manager import BaseManager
from fastlane_bot.etl.models.pool import BasePool
from fastlane_bot.etl.multicaller import MultiCaller
from fastlane_bot.etl.tasks.base import BaseTask
from fastlane_bot.etl.utils import complex_handler, safe_int, save_events_to_json
from fastlane_bot.exceptions import AsyncUpdateRetryException

nest_asyncio.apply()


@dataclass
class ExtractTask(BaseTask):
    """
    Extract task class
    """

    last_block_queried: int = 0
    last_block: int = 0
    loop_idx: int = 0
    start_timeout: float = 0
    total_iteration_time: float = 0
    provider_uri: str = None
    initial_state: List[Dict[str, Any]] = field(default_factory=list)

    def run(self, args: Any):
        # Save initial state of pool data to assert whether it has changed
        initial_state = self.mgr.pool_data.copy()

        # ensure 'last_updated_block' is in pool_data for all pools
        for idx, pool in enumerate(self.mgr.pool_data):
            if "last_updated_block" not in pool:
                pool["last_updated_block"] = self.last_block_queried
                self.mgr.pool_data[idx] = pool
            if not pool["last_updated_block"]:
                pool["last_updated_block"] = self.last_block_queried
                self.mgr.pool_data[idx] = pool

        # Get current block number, then adjust to the block number reorg_delay blocks ago to avoid reorgs
        start_block, replay_from_block = self.get_start_block(
            args.max_block_fetch,
            self.last_block,
            self.mgr,
            args.reorg_delay,
            args.replay_from_block,
        )

        # Get all events from the last block to the current block
        current_block = self.get_current_block(
            mgr=self.mgr,
            last_block=self.last_block,
            reorg_delay=args.reorg_delay,
            replay_from_block=replay_from_block,
            tenderly_fork_id=args.tenderly_fork_id,
        )

        # Log the current start, end and last block
        self.mgr.logger.info(
            f"Fetching events from {start_block} to {current_block}... {self.last_block}"
        )

        # Get the events
        latest_events = (
            self.get_cached_events(self.mgr)
            if args.use_cached_events
            else self.get_latest_events(
                current_block,
                self.mgr,
                args.n_jobs,
                start_block,
                args.cache_latest_only,
                args.logging_path,
            )
        )
        iteration_start_time = time.time()

        # Update the pools from the latest events
        self.update_pools_from_events(args.n_jobs, self.mgr, latest_events)

        # Update new pool events from contracts
        if len(self.mgr.pools_to_add_from_contracts) > 0:
            self.mgr.logger.info(
                f"Adding {len(self.mgr.pools_to_add_from_contracts)} new pools from contracts, "
                f"{len(self.mgr.pool_data)} total pools currently exist. Current block: {current_block}."
            )
            self.run_async_update_with_retries(
                self.mgr,
                current_block=current_block,
            )
            self.mgr.pools_to_add_from_contracts = []

        # Increment the loop index
        self.loop_idx += 1

        # Handle the initial iteration (backdate pools, update pools from contracts, etc.)
        self.async_handle_initial_iteration(
            backdate_pools=args.backdate_pools,
            current_block=current_block,
            last_block=self.last_block,
            mgr=self.mgr,
            start_block=start_block,
        )

        # Run multicall every iteration
        self.multicall_every_iteration(current_block=current_block, mgr=self.mgr)

        # Update the last block number
        last_block = current_block

        if not self.mgr.read_only:
            # Write the pool data to disk
            self.write_pool_data_to_disk(
                cache_latest_only=args.cache_latest_only,
                logging_path=args.logging_path,
                mgr=self.mgr,
                current_block=current_block,
            )

        # Handle/remove duplicates in the pool data
        self.handle_duplicates(self.mgr)

        return iteration_start_time, last_block, initial_state, current_block

    async def get_missing_tkn(self, contract: AsyncContract, tkn: str) -> pd.DataFrame:
        """
        Get the missing token.

        Args:
            contract (AsyncContract): The contract.
            tkn (str): The token address.

        Returns:
            pd.DataFrame: The token info.

        """
        try:
            symbol = await contract.functions.symbol().call()
        except Exception:
            symbol = None
        try:
            decimals = await contract.functions.decimals().call()
        except Exception:
            decimals = None
        try:
            df = pd.DataFrame(
                [
                    {
                        "address": tkn,
                        "symbol": symbol,
                        "decimals": decimals,
                    }
                ]
            )
        except Exception as e:
            self.mgr.logger.error(f"Failed to get token info for {tkn} {e}")
            df = pd.DataFrame(
                [
                    {
                        "address": tkn,
                        "symbol": None,
                        "decimals": decimals,
                    }
                ]
            )
        return df

    async def main_get_missing_tkn(self, lst: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Get the missing token.

        Args:
            lst (List[Dict[str, Any]]): The list of tokens.

        Returns:
            pd.DataFrame: The token info.

        """
        vals = await asyncio.wait_for(
            asyncio.gather(*[self.get_missing_tkn(**args) for args in lst]),
            timeout=20 * 60,
        )
        return pd.concat(vals)

    async def get_token_and_fee(
        self,
        exchange_name: str,
        ex: Any,
        address: str,
        contract: AsyncContract,
        event: Any,
    ) -> Tuple[
        str, str, Optional[str], Optional[str], Optional[str], Optional[str], int
    ]:
        """
        Get the token and fee.

        Args:
            exchange_name (str): The exchange name.
            ex (Any): The exchange.
            address (str): The address.
            contract (AsyncContract): The contract.
            event (Any): The event.

        Returns:
            Tuple[str, str, Optional[str], Optional[str], Optional[str], Optional[str], int]: The token and fee.

        """
        anchor = None
        tkns = CommonEthereumTokens()
        try:
            tkn0 = await ex.get_tkn0(address, contract, event=event)
            tkn1 = await ex.get_tkn1(address, contract, event=event)
            fee = await ex.get_fee(address, contract)
            if exchange_name == "bancor_v2":
                anchor = await ex.get_anchor(contract)
                for i in [0, 1]:
                    connector_token = await ex.get_connector_tokens(contract, i)
                    if connector_token != tkns.BNT:
                        break

                if tkn0 == tkns.BNT:
                    tkn1 = connector_token

                elif tkn1 == tkns.BNT:
                    tkn0 = connector_token

            cid = str(event["args"]["id"]) if exchange_name == "carbon_v1" else None

            return exchange_name, address, tkn0, tkn1, fee, cid, anchor
        except Exception as e:
            self.mgr.logger.info(
                f"Failed to get tokens and fee for {address} {exchange_name} {e}"
            )
            return exchange_name, address, None, None, None, None, anchor

    async def main_get_tokens_and_fee(self, lst: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Get the tokens and fee.

        Args:
            lst (List[Dict[str, Any]]): The list of tokens.

        Returns:
            pd.DataFrame: The tokens and fee.

        """
        vals = await asyncio.wait_for(
            asyncio.gather(*[self.get_token_and_fee(**args) for args in lst]),
            timeout=20 * 60,
        )
        return pd.DataFrame(
            vals,
            columns=[
                "exchange_name",
                "address",
                "tkn0_address",
                "tkn1_address",
                "fee",
                "cid",
                "anchor",
            ],
        )

    @staticmethod
    def get_pool_info(
        pool: pd.Series,
        mgr: Any,
        current_block: int,
        tkn0: Dict[str, Any],
        tkn1: Dict[str, Any],
        pool_data_keys: frozenset,
    ) -> Dict[str, Any]:
        """
        Get the pool info.
        Args:
            pool (pd.Series): The pool.
            mgr (Any): The manager.
            current_block (int): The current block.
            tkn0 (Dict[str, Any]): The token 0.
            tkn1 (Dict[str, Any]): The token 1.
            pool_data_keys (frozenset): The pool staticdata keys.

        Returns:
            Dict[str, Any]: The pool info.

        """
        fee_raw = eval(str(pool["fee"]))
        pool_info = {
            "exchange_name": pool["exchange_name"],
            "address": pool["address"],
            "tkn0_address": pool["tkn0_address"],
            "tkn1_address": pool["tkn1_address"],
            "fee": fee_raw[0],
            "fee_float": fee_raw[1],
            "blockchain": mgr.blockchain,
            "anchor": pool["anchor"],
            "exchange_id": mgr.EXCHANGE_IDS[pool["exchange_name"]],
            "last_updated_block": current_block,
            "tkn0_symbol": tkn0["symbol"],
            "tkn0_decimals": tkn0["decimals"],
            "tkn1_symbol": tkn1["symbol"],
            "tkn1_decimals": tkn1["decimals"],
            "pair_name": tkn0["address"] + "/" + tkn1["address"],
        }
        if len(pool_info["pair_name"].split("/")) != 2:
            raise Exception(f"pair_name is not valid for {pool_info}")

        pool_info["descr"] = mgr.pool_descr_from_info(pool_info)
        pool_info["cid"] = (
            Web3.keccak(text=f"{pool_info['descr']}").hex()
            if pool_info["exchange_name"] != "carbon_v1"
            else str(pool["cid"])
        )

        pool_info["last_updated"] = time.time()

        for key in pool_data_keys:
            if key not in pool_info.keys():
                pool_info[key] = np.nan

        return pool_info

    def get_new_pool_data(
        self,
        current_block: int,
        keys: List[str],
        mgr: Any,
        tokens_and_fee_df: pd.DataFrame,
        tokens_df: pd.DataFrame,
    ) -> List[Dict]:
        """
        Get the new pool staticdata.

        Args:
            current_block (int): The current block.
            keys (List[str]): The keys.
            mgr (Any): The manager.
            tokens_and_fee_df (pd.DataFrame): The tokens and fee.
            tokens_df (pd.DataFrame): The tokens.

        Returns:
            List[Dict]: The new pool staticdata.

        """
        # Convert tokens_df to a dictionary keyed by address for faster access
        tokens_dict = tokens_df.set_index("address").to_dict(orient="index")

        # Convert pool_data_keys to a frozenset for faster containment checks
        all_keys = set()
        for pool in mgr.pool_data:
            all_keys.update(pool.keys())
        if "last_updated_block" not in all_keys:
            all_keys.update("last_updated_block")
        pool_data_keys: frozenset = frozenset(all_keys)
        new_pool_data: List[Dict] = []
        for idx, pool in tokens_and_fee_df.iterrows():
            tkn0 = tokens_dict.get(pool["tkn0_address"])
            tkn1 = tokens_dict.get(pool["tkn1_address"])
            if not tkn0 or not tkn1:
                mgr.logger.info(
                    f"tkn0 or tkn1 not found: {pool['tkn0_address']}, {pool['tkn1_address']}, {pool['address']} "
                )
                continue
            tkn0["address"] = pool["tkn0_address"]
            tkn1["address"] = pool["tkn1_address"]
            pool_info = self.get_pool_info(
                pool, mgr, current_block, tkn0, tkn1, pool_data_keys
            )
            new_pool_data.append(pool_info)
        return new_pool_data

    def get_token_contracts(
        self, tokens_and_fee_df: pd.DataFrame
    ) -> Tuple[
        List[Dict[str, Type[AsyncContract] or AsyncContract or Any] or None or Any],
        pd.DataFrame,
    ]:
        # for each token in the pools, check whether we have the token info in the tokens.csv static staticdata,
        # and ifr not, add it
        tokens = (
            tokens_and_fee_df["tkn0_address"].tolist()
            + tokens_and_fee_df["tkn1_address"].tolist()
        )
        tokens = list(set(tokens))
        tokens_df = pd.read_csv(
            TOKENS_PATH.replace("{{blockchain}}", self.mgr.blockchain.name)
        )
        missing_tokens = [
            tkn for tkn in tokens if tkn not in tokens_df["address"].tolist()
        ]
        contracts = []
        failed_contracts = []
        contracts.extend(
            {
                "contract": self.w3_async.eth.contract(address=tkn, abi=ERC20_ABI),
                "tkn": tkn,
            }
            for tkn in missing_tokens
            if tkn is not None and str(tkn) != "nan"
        )
        self.mgr.logger.debug(
            f"[async_event_update_utils.get_token_contracts] successful token contracts: {len(contracts) - len(failed_contracts)} of {len(contracts)} "
        )
        return contracts, tokens_df

    def process_contract_chunks(
        self,
        base_filename: str,
        chunks: List[Any],
        dirname: str,
        filename: str,
        subset: List[str],
        func: Callable,
        df_combined: pd.DataFrame = None,
        read_only: bool = False,
    ) -> pd.DataFrame:
        """
        Process the contract chunks.

        Args:
            base_filename (str): The base filename.
            chunks (List[Any]): The chunks.
            dirname (str): The directory name.
            filename (str): The filename.
            subset (List[str]): The subset.
            func (Callable): The function.
            df_combined (pd.DataFrame): The combined dataframe.
            read_only (bool): Whether to read only.

        Returns:
            pd.DataFrame: The combined dataframe.

        """
        lst = []
        # write chunks to csv
        for idx, chunk in enumerate(chunks):
            loop = asyncio.get_event_loop()
            df = loop.run_until_complete(func(chunk))
            if not read_only:
                df.to_csv(f"{dirname}/{base_filename}{idx}.csv", index=False)
            else:
                lst.append(df)

        filepaths = glob(f"{dirname}/*.csv")

        if not read_only:
            # concatenate and deduplicate

            if filepaths:
                df_orig = df_combined.copy() if df_combined is not None else None
                df_combined = pd.concat(
                    [pd.read_csv(filepath) for filepath in filepaths]
                )
                df_combined = (
                    pd.concat([df_orig, df_combined])
                    if df_orig is not None
                    else df_combined
                )
                df_combined = df_combined.drop_duplicates(subset=subset)
                df_combined.to_csv(filename, index=False)
                # clear temp dir
                for filepath in filepaths:
                    try:
                        os.remove(filepath)
                    except Exception as e:
                        self.mgr.logger.error(
                            f"Failed to remove {filepath} {e}??? This is spooky..."
                        )
        else:
            if lst:
                dfs = pd.concat(lst)
                dfs = dfs.drop_duplicates(subset=subset)
                if df_combined is not None:
                    df_combined = pd.concat([df_combined, dfs])
                else:
                    df_combined = dfs

        return df_combined

    def get_pool_contracts(self, mgr: BaseManager) -> List[Dict[str, Any]]:
        """
        Get the pool contracts.

        Args:
            mgr (BaseManager): The manager.

        Returns:
            List[Dict[str, Any]]: The pool contracts.

        """
        contracts = []
        for add, en, event, key, value in self.mgr.pools_to_add_from_contracts:
            exchange_name = mgr.exchange_name_from_event(event)
            ex = mgr.exchanges[exchange_name]
            abi = ex.get_abi()
            address = event["address"]
            contracts.append(
                {
                    "exchange_name": exchange_name,
                    "ex": ex,
                    "address": address,
                    "contract": mgr.w3_async.eth.contract(address=address, abi=abi),
                    "event": event,
                }
            )
        return contracts

    @staticmethod
    def get_contract_chunks(
        contracts: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        return [contracts[i : i + 1000] for i in range(0, len(contracts), 1000)]

    def async_update_pools_from_contracts(self, mgr: Any, current_block: int):
        """
        Async update pools from contracts.

        Args:
            mgr (Any): The manager.
            current_block (int): The current block.

        """
        dirname = "temp"
        keys = [
            "liquidity",
            "tkn0_balance",
            "tkn1_balance",
            "y_0",
            "y_1",
            "liquidity",
        ]
        if not mgr.read_only and not os.path.exists(dirname):
            os.mkdir(dirname)

        start_time = time.time()
        orig_num_pools_in_data = len(mgr.pool_data)
        mgr.logger.info("Async process now updating pools from contracts...")

        all_events = [
            event
            for address, exchange_name, event, key, value in mgr.pools_to_add_from_contracts
        ]

        # split contracts into chunks of 1000
        contracts = self.get_pool_contracts(mgr)
        chunks = self.get_contract_chunks(contracts)
        tokens_and_fee_df = self.process_contract_chunks(
            base_filename="tokens_and_fee_df_",
            chunks=chunks,
            dirname=dirname,
            filename="tokens_and_fee_df.csv",
            subset=["exchange_name", "address", "cid", "tkn0_address", "tkn1_address"],
            func=self.main_get_tokens_and_fee,
            read_only=mgr.read_only,
        )

        contracts, tokens_df = self.get_token_contracts(tokens_and_fee_df)
        tokens_df = self.process_contract_chunks(
            base_filename="missing_tokens_df_",
            chunks=self.get_contract_chunks(contracts),
            dirname=dirname,
            filename="missing_tokens_df.csv",
            subset=["address"],
            func=self.main_get_missing_tkn,
            df_combined=pd.read_csv(
                TOKENS_PATH.replace("{{blockchain}}", mgr.blockchain)
            ),
            read_only=mgr.read_only,
        )
        tokens_df["symbol"] = (
            tokens_df["symbol"]
            .str.replace(" ", "_")
            .str.replace("/", "_")
            .str.replace("-", "_")
        )
        if not mgr.read_only:
            tokens_df.to_csv(
                TOKENS_PATH.replace("{{blockchain}}", mgr.blockchain), index=False
            )
        tokens_df["address"] = tokens_df["address"].apply(
            lambda x: Web3.to_checksum_address(x)
        )
        tokens_df = tokens_df.drop_duplicates(subset=["address"])

        new_pool_data = self.get_new_pool_data(
            current_block, keys, mgr, tokens_and_fee_df, tokens_df
        )

        new_pool_data_df = pd.DataFrame(new_pool_data).sort_values(
            "last_updated_block", ascending=False
        )

        new_pool_data_df = new_pool_data_df.dropna(
            subset=[
                "pair_name",
                "exchange_name",
                "fee",
                "tkn0_symbol",
                "tkn1_symbol",
                "tkn0_decimals",
                "tkn1_decimals",
            ]
        )

        new_pool_data_df["descr"] = (
            new_pool_data_df["exchange_name"]
            + " "
            + new_pool_data_df["pair_name"]
            + " "
            + new_pool_data_df["fee"].astype(str)
        )

        # Initialize web3
        new_pool_data_df["cid"] = [
            Web3.keccak(text=f"{row['descr']}").hex()
            if row["exchange_name"] not in mgr.carbon_v1_forks
            else int(row["cid"])
            for index, row in new_pool_data_df.iterrows()
        ]

        # print duplicate cid rows
        duplicate_cid_rows = new_pool_data_df[
            new_pool_data_df.duplicated(subset=["cid"])
        ]

        new_pool_data_df = (
            new_pool_data_df.sort_values("last_updated_block", ascending=False)
            .drop_duplicates(subset=["cid"])
            .set_index("cid")
        )

        duplicate_new_pool_ct = len(duplicate_cid_rows)

        all_pools_df = (
            pd.DataFrame(mgr.pool_data)
            .sort_values("last_updated_block", ascending=False)
            .drop_duplicates(subset=["cid"])
            .set_index("cid")
        )

        new_pool_data_df = new_pool_data_df[all_pools_df.columns]

        # add new_pool_data to pool_data, ensuring no duplicates
        all_pools_df.update(new_pool_data_df, overwrite=True)

        new_pool_data_df = new_pool_data_df[
            ~new_pool_data_df.index.isin(all_pools_df.index)
        ]
        all_pools_df = pd.concat([all_pools_df, new_pool_data_df])
        all_pools_df[["tkn0_decimals", "tkn1_decimals"]] = (
            all_pools_df[["tkn0_decimals", "tkn1_decimals"]].fillna(0).astype(int)
        )
        all_pools = (
            all_pools_df.sort_values("last_updated_block", ascending=False)
            .reset_index()
            .to_dict(orient="records")
        )

        mgr.pool_data = all_pools
        new_num_pools_in_data = len(mgr.pool_data)
        new_pools_added = new_num_pools_in_data - orig_num_pools_in_data

        mgr.logger.debug(
            f"[async_event_update_utils.async_update_pools_from_contracts] new_pools_added: {new_pools_added}"
        )
        mgr.logger.debug(
            f"[async_event_update_utils.async_update_pools_from_contracts] orig_num_pools_in_data: {orig_num_pools_in_data}"
        )
        mgr.logger.debug(
            f"[async_event_update_utils.async_update_pools_from_contracts] duplicate_new_pool_ct: {duplicate_new_pool_ct}"
        )
        mgr.logger.debug(
            f"[async_event_update_utils.async_update_pools_from_contracts] pools_to_add_from_contracts: {len(mgr.pools_to_add_from_contracts)}"
        )
        mgr.logger.debug(
            f"[async_event_update_utils.async_update_pools_from_contracts] final pool_data ct: {len(mgr.pool_data)}"
        )
        mgr.logger.debug(
            f"[async_event_update_utils.async_update_pools_from_contracts] compare {new_pools_added + duplicate_new_pool_ct},{len(mgr.pools_to_add_from_contracts)}"
        )

        # update the pool_data from events
        self.update_pools_from_events(-1, mgr, all_events)

        mgr.logger.info(
            f"Async Updating pools from contracts took {(time.time() - start_time):0.4f} seconds"
        )

    @staticmethod
    def update_pools_from_events(n_jobs: int, mgr: Any, latest_events: List[Any]):
        """
        Updates the pools with the given events.

        Parameters
        ----------
        n_jobs : int
            The number of jobs to run in parallel.
        mgr : Any
            The manager object.
        latest_events : List[Any]
            The latest events.

        """
        Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(mgr.update_from_event)(event=event) for event in latest_events
        )

    async def async_main_backdate_from_contracts(
        self, lst: List[Dict[str, Any]], w3_async: AsyncWeb3
    ) -> Tuple[Any]:
        return await asyncio.wait_for(
            asyncio.gather(
                *[
                    self.async_handle_main_backdate_from_contracts(
                        **args, w3_async=w3_async
                    )
                    for args in lst
                ]
            ),
            timeout=20 * 60,
        )

    @staticmethod
    async def async_handle_main_backdate_from_contracts(
        idx: int,
        pool: Any,
        w3_tenderly: Any,
        w3_async: AsyncWeb3,
        tenderly_fork_id: str,
        pool_info: Dict,
        contract: Any,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Handle the main backdate from contracts.

        Args:
            idx (int): The index.
            pool (Any): The pool.
            w3_tenderly (Any): The tenderly web3.
            w3_async (AsyncWeb3): The async web3.
            tenderly_fork_id (str): The tenderly fork id.
            pool_info (Dict): The pool info.
            contract (Any): The contract.

        Returns:
            Tuple[int, Dict[str, Any]]: The index and pool info.

        """
        params = await pool.async_update_from_contract(
            contract,
            tenderly_fork_id=tenderly_fork_id,
            w3_tenderly=w3_tenderly,
            w3=w3_async,
        )
        for key, value in params.items():
            pool_info[key] = value
        return idx, pool_info

    def async_handle_initial_iteration(
        self,
        backdate_pools: bool,
        last_block: int,
        mgr: Any,
        start_block: int,
        current_block: int,
    ):
        if last_block == 0:
            non_multicall_rows_to_update = mgr.get_rows_to_update(start_block)

            if backdate_pools:
                # Remove duplicates
                non_multicall_rows_to_update = list(set(non_multicall_rows_to_update))

                # Parse the rows to update
                other_pool_rows = self.parse_non_multicall_rows_to_update(
                    mgr, non_multicall_rows_to_update
                )

                mgr.logger.info(
                    f"Backdating {len(other_pool_rows)} pools from {start_block} to {current_block}"
                )
                start_time = time.time()
                self.async_backdate_from_contracts(
                    mgr=mgr,
                    rows=other_pool_rows,
                )
                mgr.logger.info(
                    f"Backdating {len(other_pool_rows)} pools took {(time.time() - start_time):0.4f} seconds"
                )

    def async_backdate_from_contracts(self, mgr: Any, rows: List[int]):
        """
        Async backdate from contracts.

        Args:
            mgr (Any): The manager.
            rows (List[int]): The rows.
        """
        abis = self.get_abis_and_exchanges(mgr)
        contracts = self.get_backdate_contracts(abis, mgr, rows)
        chunks = self.get_contract_chunks(contracts)
        for chunk in chunks:
            loop = asyncio.get_event_loop()
            vals = loop.run_until_complete(
                self.async_main_backdate_from_contracts(chunk, w3_async=mgr.w3_async)
            )
            idxes = [val[0] for val in vals]
            updated_pool_info = [val[1] for val in vals]
            for i, idx in enumerate(idxes):
                updated_pool_data = updated_pool_info[i]
                mgr.pool_data[idx] = updated_pool_data

    @staticmethod
    def get_abis_and_exchanges(mgr: Any) -> Dict[str, Any]:
        """
        Get the abis and exchanges.
        Args:
            mgr (Any): The manager.

        Returns:
            Dict[str, Any]: The abis and exchanges.

        """
        return {
            exchange_name: exchange.get_abi()
            for exchange_name, exchange in mgr.exchanges.items()
        }

    @staticmethod
    def get_backdate_contracts(
        abis: Dict, mgr: Any, rows: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Get the backdate contracts.

        Args:
            abis (Dict): The abis.
            mgr (Any): The manager.
            rows (List[int]): The rows.

        Returns:
            List[Dict[str, Any]]: The backdate contracts.

        """
        contracts = []
        for idx in rows:
            pool_info = mgr.pool_data[idx]
            contracts.append(
                {
                    "idx": idx,
                    "pool": mgr.get_or_init_pool(pool_info),
                    "w3_tenderly": mgr.w3_tenderly,
                    "tenderly_fork_id": mgr.tenderly_fork_id,
                    "pool_info": pool_info,
                    "contract": mgr.w3_async.eth.contract(
                        address=mgr.pool_data[idx]["address"],
                        abi=abis[mgr.pool_data[idx]["exchange_name"]],
                    ),
                }
            )
        return contracts

    @staticmethod
    def parse_non_multicall_rows_to_update(
        mgr: Any,
        rows_to_update: List[Hashable],
    ) -> List[Hashable]:
        """
        Parses the rows to update for Bancor v3 pools.

        Args:
            mgr (Any): The manager.
            rows_to_update (List[Hashable]): The rows to update.

        Returns:
            List[Hashable]: The parsed rows to update.
        """

        return [
            idx
            for idx in rows_to_update
            if mgr.pool_data[idx]["exchange_name"] not in MULTICALLABLE_EXCHANGES
        ]

    @staticmethod
    def bit_length(value: int) -> int:
        """
        Get the bit length of a value.

        Args:
            value (int): The value.

        Returns:
            int: The bit length.

        """

        return len(bin(value).lstrip("0b")) if value > 0 else 0

    def encode_float(self, value: int) -> int:
        """
        Encode a float value.

        Args:
            value (int): The value to encode.

        Returns:
            int: The encoded value.

        """
        exponent = self.bit_length(value // ONE)
        mantissa = value >> exponent
        return mantissa | (exponent * ONE)

    def encode_rate(self, value: int) -> int:
        """
        Encode a rate value.

        Args:
            value (int): The value to encode.

        Returns:
            int: The encoded value.

        """
        data = int(value.sqrt() * ONE)
        length = self.bit_length(data // ONE)
        return (data >> length) << length

    def encode_token_price(self, price: Decimal) -> int:
        """
        Encode a token price.

        Args:
            price (Decimal): The price.

        Returns:
            int: The encoded price.

        """
        return self.encode_float(self.encode_rate((price)))

    @staticmethod
    def get_pools_for_exchange(exchange: str, mgr: Any) -> [Any]:
        """
        Handles the initial iteration of the bot.

        Parameters
        ----------
        mgr : Any
            The manager object.
        exchange : str
            The exchange for which to get pools

        Returns
        -------
        List[Any]
            A list of pools for the specified exchange.
        """
        return [
            idx
            for idx, pool in enumerate(mgr.pool_data)
            if pool["exchange_name"] == exchange
        ]

    def multicall_helper(
        self,
        exchange: str,
        rows_to_update: List,
        multicall_contract: Any,
        mgr: Any,
        current_block: int,
    ):
        """
        Helper function for multicall.

        Args:
            exchange (str): The exchange.
            rows_to_update (List): The rows to update.
            multicall_contract (Any): The multicall contract.
            mgr (Any): The manager.
            current_block (int): The current block.

        """
        multicaller = MultiCaller(
            contract=multicall_contract,
            block_identifier=current_block,
            web3=mgr.web3,
            multicall_address=mgr.MULTICALL_CONTRACT_ADDRESS,
        )
        with multicaller as mc:
            for row in rows_to_update:
                pool_info = mgr.pool_data[row]
                pool_info["last_updated_block"] = current_block
                # Function to be defined elsewhere based on what each exchange type needs
                self.multicall_fn(exchange, mc, mgr, multicall_contract, pool_info)
            result_list = mc.multicall()
        self.process_results_for_multicall(exchange, rows_to_update, result_list, mgr)

    def process_results_for_multicall(
        self, exchange: str, rows_to_update: List, result_list: List, mgr: Any
    ) -> None:
        """
        Process the results for multicall.

        Args:
            exchange (str): The exchange.
            rows_to_update (List): The rows to update.
            result_list (List): The result list.
            mgr (Any): The manager.

        """
        for row, result in zip(rows_to_update, result_list):
            pool_info = mgr.pool_data[row]
            params = self.extract_params_for_multicall(exchange, result, pool_info, mgr)
            pool = mgr.get_or_init_pool(pool_info)
            pool, pool_info = self.update_pool_for_multicall(params, pool_info, pool)
            mgr.pool_data[row] = pool_info
            self.update_mgr_exchanges_for_multicall(mgr, exchange, pool, pool_info)

    def _extract_pol_params_for_multicall(
        self, result: Any, pool_info: Dict, mgr: Any
    ) -> Dict[str, Any]:
        """
        Extract the Bancor POL params for multicall.

        Args:
            result (Any): The result.
            pool_info (Dict): The pool info.
            mgr (Any): The manager.

        Returns:
            Dict[str, Any]: The extracted parameters.

        """
        tkn0_address = pool_info["tkn0_address"]
        p0, p1, tkn_balance = result
        token_price = Decimal(p1) / Decimal(p0)
        token_price = int(str(self.encode_token_price(token_price)))

        result = {
            "fee": "0.000",
            "fee_float": 0.000,
            "tkn0_balance": 0,
            "tkn1_balance": 0,
            "exchange_name": pool_info["exchange_name"],
            "address": pool_info["address"],
            "y_0": tkn_balance,
            "z_0": tkn_balance,
            "A_0": 0,
            "B_0": token_price,
            "y_1": 0,
            "z_1": 0,
            "A_1": 0,
            "B_1": 0,
        }
        return result

    @staticmethod
    def update_pool_for_multicall(
        params: Dict[str, Any], pool_info: Dict, pool: Any
    ) -> Tuple[BasePool, Dict]:
        """
        Update the pool for multicall.

        Args:
            params (Dict[str, Any]): The parameters.
            pool_info (Dict): The pool info.
            pool (Any): The pool.

        Returns:
            Tuple[BasePool, Dict]: The pool and pool info.

        """
        for key, value in params.items():
            pool_info[key] = value
            pool.state[key] = value
        return pool, pool_info

    @staticmethod
    def update_mgr_exchanges_for_multicall(
        mgr: Any, exchange: str, pool: Any, pool_info: Dict[str, Any]
    ):
        """
        Update the manager exchanges for multicall.

        Args:
            mgr (Any): The manager.
            exchange (str): The exchange.
            pool (Any): The pool.
            pool_info (Dict[str, Any]): The pool info.

        """
        unique_key = pool.unique_key()
        if unique_key == "token":
            # Handles the bancor POL case
            unique_key = "tkn0_address"

        unique_key_value = pool_info[unique_key]
        exchange_pool_idx = [
            idx
            for idx in range(len(mgr.exchanges[exchange].pools))
            if mgr.exchanges[exchange].pools[unique_key_value].state[unique_key]
            == pool_info[unique_key]
        ][0]
        mgr.exchanges[exchange].pools[exchange_pool_idx] = pool

    def multicall_every_iteration(self, current_block: int, mgr: Any):
        """
        For each exchange that supports Multicall, use multicall to update the state of the pools on every search iteration.

        Args:
            current_block (int): The current block.
            mgr (Any): The manager.
        """
        multicallable_exchanges = [
            exchange
            for exchange in mgr.MULTICALLABLE_EXCHANGES
            if exchange in mgr.exchanges
        ]
        multicallable_pool_rows = [
            list(set(self.get_pools_for_exchange(mgr=mgr, exchange=ex_name)))
            for ex_name in multicallable_exchanges
            if ex_name in mgr.exchanges
        ]

        for idx, exchange in enumerate(multicallable_exchanges):
            multicall_contract = self.get_multicall_contract_for_exchange(mgr, exchange)
            rows_to_update = multicallable_pool_rows[idx]
            self.multicall_helper(
                exchange, rows_to_update, multicall_contract, mgr, current_block
            )

    @staticmethod
    def get_multicall_contract_for_exchange(mgr: Any, exchange: str) -> str:
        """
        TODO: Move this into Exchange classes
        Get the multicall contract for the exchange.

        Args:
            mgr (Any): The manager.
            exchange (str): The exchange.

        Returns:
            str: The multicall contract.

        """
        if exchange == "bancor_v3":
            return mgr.pool_contracts[exchange][mgr.BANCOR_V3_NETWORK_INFO_ADDRESS]
        elif exchange == "bancor_pol":
            return mgr.pool_contracts[exchange][mgr.BANCOR_POL_ADDRESS]
        elif exchange == "carbon_v1":
            return mgr.pool_contracts[exchange][mgr.CARBON_CONTROLLER_ADDRESS]
        elif exchange == "balancer":
            return mgr.pool_contracts[exchange][mgr.BALANCER_VAULT_ADDRESS]
        else:
            raise ValueError(f"Exchange {exchange} not supported.")

    def extract_params_for_multicall(
        self, exchange: str, result: Any, pool_info: Dict, mgr: Any
    ) -> Dict[str, Any]:
        """
        TODO: Move this into Exchange classes
        Extract the parameters for multicall.

        Args:
            exchange (str): The exchange.
            result (Any): The result.
            pool_info (Dict): The pool info.
            mgr (Any): The manager.

        Returns:
            Dict[str, Any]: The extracted parameters.

        """
        params = {}
        if exchange == "carbon_v1":
            strategy = result
            fake_event = {
                "args": {
                    "id": strategy[0],
                    "order0": strategy[3][0],
                    "order1": strategy[3][1],
                }
            }
            params = CarbonV1Pool.parse_event(pool_info["state"], fake_event, "None")
            params["exchange_name"] = exchange
        elif exchange == "bancor_pol":
            params = self._extract_pol_params_for_multicall(result, pool_info, mgr)
        elif exchange == "bancor_v3":
            pool_balances = result
            params = {
                "fee": "0.000",
                "fee_float": 0.000,
                "tkn0_balance": pool_balances[0],
                "tkn1_balance": pool_balances[1],
                "exchange_name": exchange,
                "address": pool_info["address"],
            }
        elif exchange == "balancer":
            pool_balances = result

            params = {
                "exchange_name": exchange,
                "address": pool_info["address"],
            }

            for idx, bal in enumerate(pool_balances):
                params[f"tkn{str(idx)}_balance"] = int(bal)

        else:
            raise ValueError(f"Exchange {exchange} not supported.")

        return params

    @staticmethod
    def multicall_fn(
        exchange: str,
        mc: Any,
        mgr: Any,
        multicall_contract: Any,
        pool_info: Dict[str, Any],
    ) -> None:
        """
        TODO: Move this into Exchange classes
        Function to be defined elsewhere based on what each exchange type needs.

        Args:
            exchange (str): The exchange.
            mc (Any): The multicaller.
            mgr (Any): The manager.
            multicall_contract (Any): The multicall contract.
            pool_info (Dict[str, Any]): The pool info.

        """
        if exchange == "bancor_v3":
            mc.add_call(
                multicall_contract.functions.tradingLiquidity, pool_info["tkn1_address"]
            )
        elif exchange == "bancor_pol":
            mc.add_call(
                multicall_contract.functions.tokenPrice, pool_info["tkn0_address"]
            )
            if mgr.ARB_CONTRACT_VERSION >= 10:
                mc.add_call(
                    multicall_contract.functions.amountAvailableForTrading,
                    pool_info["tkn0_address"],
                )
        elif exchange == "carbon_v1":
            mc.add_call(multicall_contract.functions.strategy, pool_info["cid"])
        elif exchange == "balancer":
            mc.add_call(multicall_contract.functions.getPoolTokens, pool_info["anchor"])
        else:
            raise ValueError(f"Exchange {exchange} not supported.")

    @staticmethod
    def get_event_filters(
        n_jobs: int, mgr: Any, start_block: int, current_block: int
    ) -> Any:
        """
        Creates event filters for the specified block range.
        """
        bancor_pol_events = ["TradingEnabled", "TokenTraded"]

        # Get for exchanges except POL contract
        by_block_events = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(event.create_filter)(fromBlock=start_block, toBlock=current_block)
            for event in mgr.events
            if event.__name__ not in bancor_pol_events
        )

        # Get all events since the beginning of time for Bancor POL contract
        max_num_events = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(event.create_filter)(fromBlock=0, toBlock="latest")
            for event in mgr.events
            if event.__name__ in bancor_pol_events
        )
        return by_block_events + max_num_events

    @staticmethod
    def get_all_events(n_jobs: int, event_filters: Any) -> List[Any]:
        """
        Fetches all events using the given event filters.
        """

        def throttled_get_all_entries(event_filter):
            try:
                return event_filter.get_all_entries()
            except Exception as e:
                if "Too Many Requests for url" in str(e):
                    time.sleep(random.random())
                    return event_filter.get_all_entries()
                else:
                    raise e

        return Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(throttled_get_all_entries)(event_filter)
            for event_filter in event_filters
        )

    def get_latest_events(
        self,
        current_block: int,
        mgr: Any,
        n_jobs: int,
        start_block: int,
        cache_latest_only: bool,
        logging_path: str,
    ) -> List[Any]:
        """
        Gets the latest events.
        """
        tenderly_events = []

        if mgr.tenderly_fork_id and mgr.tenderly_event_exchanges:
            tenderly_events = self.get_tenderly_events(
                mgr=mgr,
                start_block=start_block,
                current_block=current_block,
                tenderly_fork_id=mgr.tenderly_fork_id,
            )
            mgr.logger.info(
                f"[events.utils.get_latest_events] tenderly_events: {len(tenderly_events)}"
            )

        # Get all event filters, events, and flatten them
        events = [
            complex_handler(event)
            for event in [
                complex_handler(event)
                for event in self.get_all_events(
                    n_jobs,
                    self.get_event_filters(n_jobs, mgr, start_block, current_block),
                )
            ]
        ]

        # Filter out the latest events per pool, save them to disk, and update the pools
        latest_events = self.filter_latest_events(mgr, events)
        if mgr.tenderly_fork_id:
            if tenderly_events:
                latest_tenderly_events = self.filter_latest_events(mgr, tenderly_events)
                latest_events += latest_tenderly_events

            # remove the events from any mgr.tenderly_event_exchanges exchanges
            for exchange in mgr.tenderly_event_exchanges:
                if pool_type := mgr.pool_type_from_exchange_name(exchange):
                    latest_events = [
                        event
                        for event in latest_events
                        if not pool_type.event_matches_format(event)
                    ]

        carbon_pol_events = [
            event for event in latest_events if "token" in event["args"]
        ]
        mgr.logger.info(
            f"[events.utils.get_latest_events] Found {len(latest_events)} new events, {len(carbon_pol_events)} carbon_pol_events"
        )

        # Save the latest events to disk
        save_events_to_json(
            cache_latest_only,
            logging_path,
            mgr,
            latest_events,
            start_block,
            current_block,
        )
        return latest_events

    def filter_latest_events(
        self, mgr: BaseManager, events: List[List[AttributeDict]]
    ) -> List[AttributeDict]:
        """
        This function filters out the latest events for each pool. Given a nested list of events, it iterates through all events
        and keeps track of the latest event (i.e., with the highest block number) for each pool. The key used to identify each pool
        is derived from the event data using manager's methods.

        Args:
            mgr (Base): A Base object that provides methods to handle events and their related pools.
            events (List[List[AttributeDict]]): A nested list of events, where each event is an AttributeDict that includes
            the event data and associated metadata.

        Returns:
            List[AttributeDict]: A list of events, each representing the latest event for its corresponding pool.
        """
        latest_entry_per_pool = {}
        all_events = [event for event_list in events for event in event_list]

        # Handles the case where multiple pools are created in the same block
        all_events.reverse()

        bancor_v2_anchor_addresses = {
            pool["anchor"]
            for pool in mgr.pool_data
            if pool["exchange_name"] == "bancor_v2"
        }

        for event in all_events:
            pool_type = mgr.pool_type_from_exchange_name(
                mgr.exchange_name_from_event(event)
            )
            if pool_type:
                key = pool_type.unique_key()
            else:
                continue
            if key == "cid":
                key = "id"
            elif key == "tkn1_address":
                if event["args"]["pool"] != mgr.BNT_ADDRESS:
                    key = "pool"
                else:
                    key = "tkn_address"

            unique_key = event[key] if key in event else event["args"][key]

            # Skip events for Bancor v2 anchors
            if (
                key == "address"
                and "_token1" in event["args"]
                and (
                    event["args"]["_token1"] in bancor_v2_anchor_addresses
                    or event["args"]["_token2"] in bancor_v2_anchor_addresses
                )
            ):
                continue

            if unique_key in latest_entry_per_pool:
                if (
                    event["blockNumber"]
                    > latest_entry_per_pool[unique_key]["blockNumber"]
                ):
                    latest_entry_per_pool[unique_key] = event
                elif (
                    event["blockNumber"]
                    == latest_entry_per_pool[unique_key]["blockNumber"]
                ):
                    if (
                        event["transactionIndex"]
                        == latest_entry_per_pool[unique_key]["transactionIndex"]
                    ):
                        if (
                            event["logIndex"]
                            > latest_entry_per_pool[unique_key]["logIndex"]
                        ):
                            latest_entry_per_pool[unique_key] = event
                    elif (
                        event["transactionIndex"]
                        > latest_entry_per_pool[unique_key]["transactionIndex"]
                    ):
                        latest_entry_per_pool[unique_key] = event
                    else:
                        continue
                else:
                    continue
            else:
                latest_entry_per_pool[unique_key] = event

        return list(latest_entry_per_pool.values())

    @staticmethod
    def get_start_block(
        alchemy_max_block_fetch: int,
        last_block: int,
        mgr: Any,
        reorg_delay: int,
        replay_from_block: int,
    ) -> Tuple[int, int or None]:
        """
        Gets the starting block number.
        """
        if last_block == 0:
            if replay_from_block:
                return (
                    replay_from_block - reorg_delay - alchemy_max_block_fetch,
                    replay_from_block,
                )
            elif mgr.tenderly_fork_id:
                return (
                    mgr.w3_tenderly.eth.block_number
                    - reorg_delay
                    - alchemy_max_block_fetch,
                    mgr.w3_tenderly.eth.block_number,
                )
            else:
                return (
                    mgr.web3.eth.block_number - reorg_delay - alchemy_max_block_fetch,
                    None,
                )
        elif replay_from_block:
            return replay_from_block - 1, replay_from_block
        elif mgr.tenderly_fork_id:
            return (
                safe_int(max(block["last_updated_block"] for block in mgr.pool_data))
                - reorg_delay,
                mgr.w3_tenderly.eth.block_number,
            )
        else:
            return (
                safe_int(max(block["last_updated_block"] for block in mgr.pool_data))
                - reorg_delay,
                None,
            )

    @staticmethod
    def get_current_block(
        last_block: int,
        mgr: Any,
        reorg_delay: int,
        replay_from_block: int,
        tenderly_fork_id: str,
    ) -> int:
        """
        Get the current block number, then adjust to the block number reorg_delay blocks ago to avoid reorgs

        """
        if not replay_from_block and not tenderly_fork_id:
            current_block = mgr.web3.eth.block_number - reorg_delay
        elif last_block == 0 and replay_from_block:
            current_block = replay_from_block - reorg_delay
        elif tenderly_fork_id:
            current_block = mgr.w3_tenderly.eth.block_number
        else:
            current_block = last_block + 1
        return current_block

    def get_cached_events(self, mgr: Any) -> List[Any]:
        """
        Gets the cached events.
        """
        # read data from the json file latest_event_data.json
        self.mgr.logger.info("[events.utils] Using cached events...")
        path = "fastlane_bot/etl/staticdata/cached_event_data.json".replace(
            "./logs", "logs"
        )
        os.path.isfile(path)
        with open(path, "r") as f:
            latest_events = json.load(f)
        if not latest_events or len(latest_events) == 0:
            raise ValueError("No events found in the json file")
        mgr.logger.info(f"[events.utils] Found {len(latest_events)} new events")
        return latest_events

    @staticmethod
    def update_pools_from_events(n_jobs: int, mgr: Any, latest_events: List[Any]):
        """
        Updates the pools with the given events.
        """
        Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(mgr.update_from_event)(event=event) for event in latest_events
        )

    @staticmethod
    def write_pool_data_to_disk(
        cache_latest_only: bool, logging_path: str, mgr: Any, current_block: int
    ) -> None:
        """
        Writes the pool data to disk.

        Parameters
        ----------
        cache_latest_only : bool
            Whether to cache the latest pool data only.
        logging_path : str
            The logging path.
        mgr : Any
            The manager object.
        current_block : int
            The current block number.
        """
        if cache_latest_only:
            path = f"{logging_path}/latest_pool_data.json"
        else:
            if not os.path.isdir("pool_data"):
                os.mkdir("pool_data")
            path = f"pool_data/{mgr.SUPPORTED_EXCHANGES}_{current_block}.json"
        try:
            df = pd.DataFrame(mgr.pool_data)

            def remove_nan(row):
                return {col: val for col, val in row.items() if pd.notna(val)}

            # Apply the function to each row
            cleaned_df = df.apply(remove_nan, axis=1)
            cleaned_df.to_json(path, orient="records")
        except Exception as e:
            mgr.logger.error(f"Error writing pool data to disk: {e}")

    @staticmethod
    def handle_duplicates(mgr: Any):
        """
        Handles the duplicates in the pool data.

        Parameters
        ----------
        mgr : Any
            The manager object.

        """
        # check if any duplicate cid's exist in the pool data
        mgr.deduplicate_pool_data()
        cids = [pool["cid"] for pool in mgr.pool_data]
        assert len(cids) == len(set(cids)), "duplicate cid's exist in the pool data"

    def run_async_update_with_retries(self, mgr, current_block, max_retries=5):
        failed_async_calls = 0

        while failed_async_calls < max_retries:
            try:
                self.async_update_pools_from_contracts(mgr, current_block)
                return  # Successful execution
            except AsyncUpdateRetryException as e:
                failed_async_calls += 1
                mgr.logger.error(f"Attempt {failed_async_calls} failed: {e}")
                self.update_remaining_pools()

        # Handling failure after retries
        mgr.logger.error(
            f"[main run_bot.py] async_update_pools_from_contracts failed after "
            f"{len(mgr.pools_to_add_from_contracts)} attempts. List of failed pools: {mgr.pools_to_add_from_contracts}"
        )

        raise AsyncUpdateRetryException(
            "[run_bot.py] async_update_pools_from_contracts failed after maximum retries."
        )

    def update_remaining_pools(self):
        remaining_pools = []
        all_events = [pool[2] for pool in self.mgr.pools_to_add_from_contracts]
        for event in all_events:
            addr = Web3.to_checksum_address(event["address"])
            ex_name = self.mgr.exchange_name_from_event(event)
            if not ex_name:
                self.mgr.logger.warning(
                    "[run_async_update_with_retries] ex_name not found from event"
                )
                continue

            key, key_value = self.mgr.get_key_and_value(event, addr, ex_name)
            pool_info = self.get_pool_info(key, key_value, ex_name)

            if not pool_info:
                remaining_pools.append((addr, ex_name, event, key, key_value))

        random.shuffle(remaining_pools)
        self.mgr.pools_to_add_from_contracts = remaining_pools
