# coding=utf-8
"""
Contains the utils functions for events

(c) Copyright Bprotocol foundation 2023.
Licensed under MIT
"""
import base64
import json
import logging
import os
import platform
import sys
import time
from glob import glob
from typing import Any, Dict, Hashable, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import requests
from hexbytes import HexBytes
from joblib import Parallel, delayed
from web3 import Web3
from web3.datastructures import AttributeDict

from fastlane_bot.bot import CarbonBot
from fastlane_bot.constants import BASE_PATH, TOKENS_PATH
from fastlane_bot.etl.models.manager import BaseManager, Manager
from fastlane_bot.exceptions import ReadOnlyException
from fastlane_bot.helpers import TxHelpers


def complex_handler(obj: Any) -> Union[Dict, str, List, Set, Any]:
    """
    This function aims to handle complex data types, such as web3.py's AttributeDict, HexBytes, and native Python collections
    like dict, list, tuple, and set. It recursively traverses these collections and converts their elements into more "primitive"
    types, making it easier to work with these elements or serialize the data into JSON.

    Args:
        obj (Any): The object to be processed. This can be of any data type, but the function specifically handles AttributeDict,
        HexBytes, dict, list, tuple, and set.

    Returns:
        Union[Dict, str, List, Set, Any]: Returns a "simplified" version of the input object, where AttributeDict is converted
        into dict, HexBytes into str, and set into list. For dict, list, and tuple, it recursively processes their elements.
        If the input object does not match any of the specified types, it is returned as is.
    """
    if isinstance(obj, AttributeDict):
        return dict(obj)
    elif isinstance(obj, HexBytes):
        return obj.hex()
    elif isinstance(obj, bytes):
        return obj.hex()
    elif isinstance(obj, dict):
        return {k: complex_handler(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [complex_handler(i) for i in obj]
    elif isinstance(obj, set):
        return list(obj)
    else:
        return obj


def add_initial_pool_data(mgr: Any, n_jobs: int = -1):
    """
    Adds initial pool data to the manager.

    Parameters
    ----------
    cfg : Config
        The config object.
    mgr : Any
        The manager object.
    n_jobs : int, optional
        The number of jobs to run in parallel, by default -1

    """
    # Add initial pools for each row in the static_pool_data
    start_time = time.time()
    Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(mgr.add_pool_to_exchange)(row) for row in mgr.pool_data
    )
    mgr.logger.debug(
        f"[events.utils] Time taken to add initial pools: {time.time() - start_time}"
    )


class CSVReadError(Exception):
    """Raised when a CSV file cannot be read."""

    pass


def read_csv_file(filepath: str, low_memory: bool = False) -> pd.DataFrame:
    """Helper function to read a CSV file.

    Parameters
    ----------
    filepath : str
        The filepath of the CSV file.
    low_memory : bool, optional
        Whether to read the CSV file in low memory mode, by default False

    Returns
    -------
    pd.DataFrame
        The CSV data as a pandas DataFrame.

    Raises
    ------
    CSVReadError
        If the file does not exist or cannot be parsed.
    """
    if not os.path.isfile(filepath):
        raise CSVReadError(f"File {filepath} does not exist")
    try:
        return pd.read_csv(filepath, low_memory=low_memory)
    except pd.errors.ParserError as e:
        raise CSVReadError(f"Error parsing the CSV file {filepath}") from e


def get_tkn_symbol(tkn_address, tokens: pd.DataFrame) -> str:
    """
    Gets the token symbol for logging purposes
    :param tkn_address: the token address
    :param tokens: the Dataframe containing token information

    returns: str
    """
    try:
        return tokens.loc[tokens["address"] == tkn_address]["symbol"].values[0]
    except Exception:
        return tkn_address


def get_tkn_symbols(flashloan_tokens, tokens: pd.DataFrame) -> List:
    """
    Gets the token symbol for logging purposes
    :param flashloan_tokens: the flashloan token addresses
    :param tokens: the Dataframe containing token information

    returns: list
    """
    flashloan_tokens = flashloan_tokens.split(",")
    flashloan_tkn_symbols = []
    for tkn in flashloan_tokens:
        flashloan_tkn_symbols.append(get_tkn_symbol(tkn_address=tkn, tokens=tokens))
    return flashloan_tkn_symbols


def get_static_data(
    exchanges: List[str],
    blockchain: str,
    static_pool_data_filename: str,
    read_only: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Helper function to get static pool data, tokens, and Uniswap v2 event mappings.

    Parameters
    ----------
    cfg : Config
        The config object.
    exchanges : List[str]
        A list of exchanges to fetch data for.
    blockchain : str
        The name of the blockchain being used
    static_pool_data_filename : str
        The filename of the static pool data CSV file.
    read_only : bool, optional
        Whether to run the bot in read-only mode, by default False

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]
        A tuple of static pool data, tokens, and Uniswap v2 event mappings.

    """
    base_path = BASE_PATH.replace("{{blockchain}}", blockchain)
    # Read static pool data from CSV
    static_pool_data_filepath = os.path.join(
        base_path, f"{static_pool_data_filename}.csv"
    )
    static_pool_data = read_csv_file(static_pool_data_filepath)
    static_pool_data = static_pool_data[
        static_pool_data["exchange_name"].isin(exchanges)
    ]

    # Read Uniswap v2 event mappings and tokens
    uniswap_v2_filepath = os.path.join(base_path, "uniswap_v2_event_mappings.csv")
    uniswap_v2_event_mappings_df = read_csv_file(uniswap_v2_filepath)
    uniswap_v2_event_mappings = dict(
        uniswap_v2_event_mappings_df[["address", "exchange"]].values
    )

    # Read Uniswap v3 event mappings and tokens
    uniswap_v3_filepath = os.path.join(base_path, "uniswap_v3_event_mappings.csv")
    uniswap_v3_event_mappings_df = read_csv_file(uniswap_v3_filepath)
    uniswap_v3_event_mappings = dict(
        uniswap_v3_event_mappings_df[["address", "exchange"]].values
    )
    # Read Solidly v2 event mappings and tokens
    solidly_v2_filepath = os.path.join(base_path, "solidly_v2_event_mappings.csv")
    solidly_v2_event_mappings_df = read_csv_file(solidly_v2_filepath)
    solidly_v2_event_mappings = dict(
        solidly_v2_event_mappings_df[["address", "exchange"]].values
    )

    tokens_filepath = os.path.join(base_path, "tokens.csv")
    if not os.path.exists(tokens_filepath) and not read_only:
        df = pd.DataFrame(columns=["address", "symbol", "decimals"])
        df.to_csv(tokens_filepath)
    elif not os.path.exists(tokens_filepath) and read_only:
        raise ReadOnlyException(
            f"Tokens file {tokens_filepath} does not exist. Please run the bot in non-read-only mode to create it."
        )
    tokens = read_csv_file(tokens_filepath)
    tokens["address"] = tokens["address"].apply(lambda x: Web3.to_checksum_address(x))
    tokens = tokens.drop_duplicates(subset=["address"])
    tokens = tokens.dropna(subset=["decimals", "symbol", "address"])
    tokens["symbol"] = (
        tokens["symbol"]
        .str.replace(" ", "_")
        .str.replace("/", "_")
        .str.replace("-", "_")
    )

    def correct_tkn(tkn_address, keyname):
        try:
            return tokens[tokens["address"] == tkn_address][keyname].values[0]
        except IndexError:
            return np.nan

    static_pool_data["tkn0_address"] = static_pool_data["tkn0_address"].apply(
        lambda x: Web3.to_checksum_address(x)
    )
    static_pool_data["tkn1_address"] = static_pool_data["tkn1_address"].apply(
        lambda x: Web3.to_checksum_address(x)
    )
    static_pool_data["tkn0_decimals"] = static_pool_data["tkn0_address"].apply(
        lambda x: correct_tkn(x, "decimals")
    )
    static_pool_data["tkn1_decimals"] = static_pool_data["tkn1_address"].apply(
        lambda x: correct_tkn(x, "decimals")
    )

    static_pool_data["tkn0_symbol"] = static_pool_data["tkn0_address"].apply(
        lambda x: correct_tkn(x, "symbol")
    )
    static_pool_data["tkn1_symbol"] = static_pool_data["tkn1_address"].apply(
        lambda x: correct_tkn(x, "symbol")
    )
    static_pool_data["pair_name"] = (
        static_pool_data["tkn0_address"] + "/" + static_pool_data["tkn1_address"]
    )
    static_pool_data = static_pool_data.dropna(
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

    static_pool_data["descr"] = (
        static_pool_data["exchange_name"]
        + " "
        + static_pool_data["pair_name"]
        + " "
        + static_pool_data["fee"].astype(str)
    )
    # Initialize web3
    static_pool_data["cid"] = [
        Web3.keccak(text=f"{row['descr']}").hex()
        for index, row in static_pool_data.iterrows()
    ]

    static_pool_data = static_pool_data.drop_duplicates(subset=["cid"])
    static_pool_data.reset_index(drop=True, inplace=True)

    return (
        static_pool_data,
        tokens,
        uniswap_v2_event_mappings,
        uniswap_v3_event_mappings,
        solidly_v2_event_mappings,
    )


def handle_tenderly_event_exchanges(exchanges: str, tenderly_fork_id: str) -> List[str]:
    """
    Handles the exchanges parameter.
    """
    if not tenderly_fork_id:
        return []

    if not exchanges or exchanges == "None":
        return []

    return exchanges.split(",") if exchanges else []


def handle_target_tokens(
    flashloan_tokens: List[str],
    target_tokens: str,
) -> List[str]:
    """
    Handles the target tokens parameter.
    """

    if target_tokens:
        if target_tokens == "flashloan_tokens":
            target_tokens = flashloan_tokens
        else:
            target_tokens = target_tokens.split(",")

            # Ensure that the target tokens are a subset of the flashloan tokens
            for token in flashloan_tokens:
                if token not in target_tokens:
                    target_tokens.append(token)

    return target_tokens


def handle_flashloan_tokens(flashloan_tokens: str, tokens: pd.DataFrame) -> List[str]:
    """
    Handles the flashloan tokens parameter.
    """
    flashloan_tokens = flashloan_tokens.split(",")
    return [tkn for tkn in flashloan_tokens if tkn in tokens["address"].unique()]


def convert_to_serializable(data: Any) -> Any:
    if isinstance(data, bytes):
        return base64.b64encode(data).decode("ascii")
    elif isinstance(data, dict):
        return {key: convert_to_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    elif hasattr(data, "__dict__"):
        return convert_to_serializable(data.__dict__)
    else:
        return data


def save_events_to_json(
    mgr: BaseManager,
    latest_events: List[Any],
) -> None:
    """
    Saves the given events to a JSON file.
    """
    path = f"{mgr.logging_path}/latest_event_data.json"
    try:
        with open(path, "w") as f:
            # Remove contextId from the latest events
            latest_events = convert_to_serializable(latest_events)
            f.write(json.dumps(latest_events))
            mgr.logger.info(f"Saved events to {path}")
    except Exception as e:
        mgr.logger.warning(
            f"[events.utils.save_events_to_json]: {e}. "
            f"This will not impact bot functionality. "
            f"Skipping..."
        )
    mgr.logger.debug(f"[events.utils.save_events_to_json] Saved events to {path}")


def parse_non_multicall_rows_to_update(
    mgr: Any,
    rows_to_update: List[Hashable],
) -> List[Hashable]:
    """
    Parses the rows to update for Bancor v3 pools.

    Parameters
    ----------
    mgr : Any
        The manager object.
    rows_to_update : List[Hashable]
        A list of rows to update.

    Returns
    -------
    Tuple[List[Hashable], List[Hashable]]
        A tuple of the Bancor v3 pool rows to update and other pool rows to update.
    """

    return [
        idx
        for idx in rows_to_update
        if mgr.pool_data[idx]["exchange_name"] not in mgr.cfg.MULTICALLABLE_EXCHANGES
    ]


def update_pools_from_contracts(
    mgr: Any,
    n_jobs: int,
    rows_to_update: List[int] or List[Hashable],
    token_address: bool = False,
    current_block: int = None,
) -> None:
    """
    Updates the pools with the given indices by calling the contracts.

    Parameters
    ----------
    mgr : Any
        The manager object.
    n_jobs : int
        The number of jobs to run in parallel.
    rows_to_update : List[int]
        A list of rows to update.
    multicall_contract : MultiProviderContractWrapper or web3.contract.Contract
        The multicall contract.
    token_address : bool, optional
        Whether to update the token address, by default False
    current_block : int, optional
        The current block number, by default None

    """
    Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(mgr.update)(
            pool_info=mgr.pool_data[idx],
            block_number=current_block,
            token_address=token_address,
        )
        for idx in rows_to_update
    )


def handle_subsequent_iterations(
    arb_mode: str,
    bot: CarbonBot,
    flashloan_tokens: List[str],
    polling_interval: int,
    randomizer: int,
    run_data_validator: bool,
    target_tokens: List[str] = None,
    loop_idx: int = 0,
    logging_path: str = None,
    replay_from_block: int = None,
    tenderly_uri: str = None,
    mgr: Any = None,
    forked_from_block: int = None,
):
    """
    Handles the subsequent iterations of the bot.

    Parameters
    ----------
    arb_mode : str
        The arb mode.
    bot : CarbonBot
        The bot object.
    flashloan_tokens : List[str]
        A list of flashloan tokens.
    polling_interval : int
        The polling interval.
    randomizer : int
        The randomizer.
    run_data_validator : bool
        Whether to run the data validator.
    target_tokens : List[str], optional
        A list of target tokens, by default None
    loop_idx : int, optional
        The loop index, by default 0
    logging_path : str, optional
        The logging path, by default None
    replay_from_block : int, optional
        The block number to replay from, by default None
    tenderly_uri : str, optional
        The Tenderly URI, by default None
    mgr : Any
        The manager object.
    forked_from_block : int
        The block number to fork from.

    """
    if loop_idx > 0 or replay_from_block:
        bot.db.remove_unmapped_uniswap_v2_pools()
        bot.db.remove_zero_liquidity_pools()
        bot.db.remove_unsupported_exchanges()

        # Filter the target tokens
        if target_tokens:
            bot.db.filter_target_tokens(target_tokens)

        # Log the forked_from_block
        if forked_from_block:
            mgr.cfg.logger.info(
                f"[events.utils] Submitting bot.run with forked_from_block: {forked_from_block}, replay_from_block {replay_from_block}"
            )
            mgr.cfg.w3 = Web3(Web3.HTTPProvider(tenderly_uri))

        # Run the bot
        bot.run(
            polling_interval=polling_interval,
            flashloan_tokens=flashloan_tokens,
            mode="single",
            arb_mode=arb_mode,
            run_data_validator=run_data_validator,
            randomizer=randomizer,
            logging_path=logging_path,
            replay_mode=True if replay_from_block else False,
            tenderly_fork=tenderly_uri.split("/")[-1] if tenderly_uri else None,
            replay_from_block=forked_from_block,
        )


def verify_state_changed(bot: CarbonBot, initial_state: List[Dict[str, Any]], mgr: Any):
    """
    Verifies that the state has changed.

    Parameters
    ----------
    bot : CarbonBot
        The bot object.
    initial_state : Dict[str, Any]
        The initial state.
    mgr : Any
        The manager object.

    """
    # Compare the initial state to the final state, and update the state if it has changed
    final_state = mgr.pool_data.copy()
    final_state_bancor_pol = [
        final_state[i]
        for i in range(len(final_state))
        if final_state[i]["exchange_name"] == mgr.cfg.BANCOR_POL_NAME
    ]
    # assert bot.db.state == final_state, "\n *** bot failed to update state *** \n"
    if initial_state != final_state_bancor_pol:
        mgr.cfg.logger.debug("[events.utils.verify_state_changed] State has changed...")
    else:
        mgr.cfg.logger.warning(
            "[events.utils.verify_state_changed] State has not changed... This may indicate an error"
        )


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


def handle_initial_iteration(
    backdate_pools: bool,
    current_block: int,
    last_block: int,
    mgr: Any,
    n_jobs: int,
    start_block: int,
):
    """
    Handles the initial iteration of the bot.

    Parameters
    ----------
    backdate_pools : bool
        Whether to backdate the pools.
    current_block : int
        The current block number.
    last_block : int
        The last block number.
    mgr : Any
        The manager object.
    n_jobs : int
        The number of jobs to run in parallel.
    start_block : int
        The starting block number.

    """

    if last_block == 0:
        non_multicall_rows_to_update = mgr.get_rows_to_update(start_block)

        if backdate_pools:
            # Remove duplicates
            non_multicall_rows_to_update = list(set(non_multicall_rows_to_update))

            # Parse the rows to update
            other_pool_rows = parse_non_multicall_rows_to_update(
                mgr, non_multicall_rows_to_update
            )

            for rows in [other_pool_rows]:
                update_pools_from_contracts(
                    mgr,
                    n_jobs=n_jobs,
                    rows_to_update=rows,
                    current_block=current_block,
                )


def get_tenderly_events(
    mgr,
    start_block,
    current_block,
    tenderly_fork_id,
):
    """
    Gets the Tenderly POL events.

    Parameters
    ----------
    mgr: Any
        The manager object.
    start_block: int
        The starting block number.
    current_block: int
        The current block number.
    tenderly_fork_id: str
        The Tenderly fork ID.

    Returns
    -------
    List[Any]
        A list of Tenderly POL events.

    """
    # connect to the Tenderly fork
    mgr.cfg.logger.info(
        f"Connecting to Tenderly fork: {tenderly_fork_id}, current_block: {current_block}, start_block: {start_block}"
    )
    tenderly_events_all = []
    tenderly_exchanges = mgr.tenderly_event_exchanges
    for exchange in tenderly_exchanges:
        contract = mgr.tenderly_event_contracts[exchange]
        exchange_events = mgr.exchanges[exchange].get_events(contract)

        tenderly_events = [
            event.getLogs(fromBlock=current_block - 1000, toBlock=current_block)
            for event in exchange_events
        ]

        tenderly_events = [event for event in tenderly_events if len(event) > 0]
        tenderly_events = [
            complex_handler(event)
            for event in [complex_handler(event) for event in tenderly_events]
        ]
        tenderly_events_all += tenderly_events
    return tenderly_events_all


def safe_int(value: int or float) -> int:
    assert value == int(value), f"non-integer `float` value {value}"
    return int(value)


def get_tenderly_block_number(tenderly_fork_id: str) -> int:
    """
    Gets the Tenderly block number.

    Parameters
    ----------
    tenderly_fork_id : str
        The Tenderly fork ID.

    Returns
    -------
    int
        The Tenderly block number.

    """
    provider = Web3.HTTPProvider(f"https://rpc.tenderly.co/fork/{tenderly_fork_id}")
    web3 = Web3(provider)
    return web3.eth.block_number


def setup_replay_from_block(mgr: Any, block_number: int) -> Tuple[str, int]:
    """
    Setup a Tenderly fork from a specific block number.

    Parameters
    ----------
    mgr : Any
        The manager object
    block_number: int
        The block number to fork from.

    Returns
    -------
    str
        The web3 provider URL to use for the fork.

    """
    from web3 import Web3

    # The network and block where Tenderly fork gets created
    forkingPoint = {"network_id": "1", "block_number": block_number}

    # Define your Tenderly credentials and project info
    tenderly_access_key = os.getenv("TENDERLY_ACCESS_KEY")
    tenderly_user = os.getenv("TENDERLY_USER")
    tenderly_project = os.getenv("TENDERLY_PROJECT")

    # Base URL for Tenderly's API
    base_url = "https://api.tenderly.co/api/v1"

    # Define the headers for the request
    headers = {"X-Access-Key": tenderly_access_key, "Content-Type": "application/json"}

    # Define the project URL
    project_url = f"account/{tenderly_user}/project/{tenderly_project}"

    # Make the request to create the fork
    fork_response = requests.post(
        f"{base_url}/{project_url}/fork", headers=headers, json=forkingPoint
    )

    # Check if the request was successful
    fork_response.raise_for_status()

    # Parse the JSON response
    fork_data = fork_response.json()

    # Extract the fork id from the response
    fork_id = fork_data["simulation_fork"]["id"]

    # Log the fork id
    mgr.cfg.logger.info(
        f"[events.utils.setup_replay_from_block] Forked with fork id: {fork_id}"
    )

    # Create the provider you can use throughout the rest of your project
    provider = Web3.HTTPProvider(f"https://rpc.tenderly.co/fork/{fork_id}")

    mgr.cfg.logger.info(
        f"[events.utils.setup_replay_from_block] Forking from block_number: {block_number}, for fork_id: {fork_id}"
    )

    return provider.endpoint_uri, block_number


def set_network_connection_to_tenderly(
    mgr: Any,
    use_cached_events: bool,
    tenderly_uri: str,
    forked_from_block: int = None,
    tenderly_fork_id: str = None,
) -> Any:
    """
    Set the network connection to Tenderly.

    Parameters
    ----------
    mgr: Any (Manager)
        The manager object.
    use_cached_events: bool
        Whether to use cached events.
    tenderly_uri: str
        The Tenderly URI.
    forked_from_block: int
        The block number the Tenderly fork was created from.
    tenderly_fork_id: str
        The Tenderly fork ID.

    Returns
    -------
    Any (Manager object, Any is used to avoid circular import)
        The manager object.

    """
    assert (
        not use_cached_events
    ), "Cannot replay from block and use cached events at the same time"
    if not tenderly_uri and not tenderly_fork_id:
        return mgr, forked_from_block
    elif tenderly_fork_id:
        tenderly_uri = f"https://rpc.tenderly.co/fork/{tenderly_fork_id}"
        forked_from_block = None
        mgr.cfg.logger.info(
            f"[events.utils.set_network_connection_to_tenderly] Using Tenderly fork id: {tenderly_fork_id} at {tenderly_uri}"
        )
        mgr.cfg.w3 = Web3(Web3.HTTPProvider(tenderly_uri))
    elif tenderly_uri:
        mgr.cfg.logger.info(
            f"[events.utils.set_network_connection_to_tenderly] Connecting to Tenderly fork at {tenderly_uri}"
        )
        mgr.cfg.w3 = Web3(Web3.HTTPProvider(tenderly_uri))

    if tenderly_fork_id and not forked_from_block:
        forked_from_block = mgr.cfg.w3.eth.block_number

    assert (
        mgr.cfg.w3.provider.endpoint_uri == tenderly_uri
    ), f"Failed to connect to Tenderly fork at {tenderly_uri} - got {mgr.cfg.w3.provider.endpoint_uri} instead"
    mgr.cfg.logger.info(
        f"[events.utils.set_network_connection_to_tenderly] Successfully connected to Tenderly fork at {tenderly_uri}, forked from block: {forked_from_block}"
    )
    mgr.cfg.NETWORK = mgr.cfg.NETWORK_TENDERLY
    return mgr, forked_from_block


def set_network_connection_to_mainnet(
    mgr: Any, use_cached_events: bool, mainnet_uri: str
) -> Any:
    """
    Set the network connection to Mainnet.

    Parameters
    ----------
    mgr
    use_cached_events
    mainnet_uri

    Returns
    -------
    Any (Manager object, Any is used to avoid circular import)
        The manager object.

    """

    assert (
        not use_cached_events
    ), "Cannot replay from block and use cached events at the same time"

    mgr.cfg.w3 = Web3(Web3.HTTPProvider(mainnet_uri))

    assert (
        mgr.cfg.w3.provider.endpoint_uri == mainnet_uri
    ), f"Failed to connect to Mainnet at {mainnet_uri} - got {mgr.cfg.w3.provider.endpoint_uri} instead"
    mgr.cfg.logger.info(
        "[events.utils.set_network_connection_to_mainnet] Successfully connected to Mainnet"
    )
    mgr.cfg.NETWORK = mgr.cfg.NETWORK_MAINNET
    return mgr


def handle_limit_pairs_for_replay_mode(
    limit_pairs_for_replay: str,
    replay_from_block: int,
    static_pool_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Splits, validates, and logs the `limit_pairs_for_replay` for replay mode.

    Parameters
    ----------
    cfg: Config
        The config object.
    limit_pairs_for_replay: str
        A comma-separated list of pairs to limit replay to. Must be in the format
    replay_from_block: int
        The block number to replay from. (For debugging / testing)
    static_pool_data: pd.DataFrame
        The static pool data.

    Returns
    -------
    pd.DataFrame
        The static pool data.

    """
    if limit_pairs_for_replay and replay_from_block:
        limit_pairs_for_replay = limit_pairs_for_replay.split(",")
        static_pool_data = static_pool_data[
            static_pool_data["pair_name"].isin(limit_pairs_for_replay)
        ]
    return static_pool_data


def set_network_to_tenderly_if_replay(
    last_block: int,
    loop_idx: int,
    mgr: Any,
    replay_from_block: int,
    tenderly_uri: str or None,
    use_cached_events: bool,
    forked_from_block: int = None,
    tenderly_fork_id: str = None,
) -> Tuple[Any, str or None, int or None]:
    """
    Set the network connection to Tenderly if replaying from a block

    Parameters
    ----------
    last_block : int
        The last block that was processed
    loop_idx : int
        The current loop index
    mgr : Any
        The manager object
    replay_from_block : int
        The block to replay from
    tenderly_uri : str
        The Tenderly URI
    use_cached_events : bool
        Whether to use cached events
    forked_from_block : int
        The block number the Tenderly fork was created from.
    tenderly_fork_id : str
        The Tenderly fork id

    Returns
    -------
    mgr : Any
        The manager object
    tenderly_uri : str or None
        The Tenderly URI
    forked_from_block : int or None
        The block number the Tenderly fork was created from.
    """
    if not replay_from_block and not tenderly_fork_id:
        return mgr, None, None

    elif last_block == 0 and tenderly_fork_id:
        mgr.cfg.logger.info(
            f"[events.utils.set_network_to_tenderly_if_replay] Setting network connection to Tenderly idx: {loop_idx}"
        )
        mgr, forked_from_block = set_network_connection_to_tenderly(
            mgr=mgr,
            use_cached_events=use_cached_events,
            tenderly_uri=tenderly_uri,
            forked_from_block=forked_from_block,
            tenderly_fork_id=tenderly_fork_id,
        )
        tenderly_uri = mgr.cfg.w3.provider.endpoint_uri
        return mgr, tenderly_uri, forked_from_block

    elif replay_from_block and loop_idx > 0 and mgr.cfg.NETWORK != "tenderly":
        # Tx must always be submitted from Tenderly when in replay mode
        mgr.cfg.logger.info(
            f"[events.utils.set_network_to_tenderly_if_replay] Setting network connection to Tenderly idx: {loop_idx}"
        )
        mgr, forked_from_block = set_network_connection_to_tenderly(
            mgr=mgr,
            use_cached_events=use_cached_events,
            tenderly_uri=tenderly_uri,
            forked_from_block=forked_from_block,
        )
        mgr.cfg.w3.provider.endpoint_uri = tenderly_uri
        return mgr, tenderly_uri, forked_from_block

    else:
        tenderly_uri, forked_from_block = setup_replay_from_block(
            mgr=mgr, block_number=replay_from_block
        )
        mgr.cfg.NETWORK = mgr.cfg.NETWORK_TENDERLY
        return mgr, tenderly_uri, forked_from_block


def set_network_to_mainnet_if_replay(
    last_block: int,
    loop_idx: int,
    mainnet_uri: str,
    mgr: Any,
    replay_from_block: int,
    use_cached_events: bool,
):
    """
    Set the network connection to Mainnet if replaying from a block

    Parameters
    ----------
    last_block : int
        The last block that the bot processed
    loop_idx : int
        The current loop index
    mainnet_uri : str
        The URI of the Mainnet node
    mgr : Any
        The manager object
    replay_from_block : int
        The block to replay from
    use_cached_events : bool
        Whether to use cached events

    Returns
    -------
    mgr : Any
        The manager object

    """
    if (
        (replay_from_block or mgr.tenderly_fork_id)
        and mgr.cfg.NETWORK != "mainnet"
        and last_block != 0
    ):
        mgr.cfg.logger.info(
            f"[events.utils.set_network_to_mainnet_if_replay] Setting network connection to Mainnet idx: {loop_idx}"
        )
        mgr = set_network_connection_to_mainnet(
            mgr=mgr,
            use_cached_events=use_cached_events,
            mainnet_uri=mainnet_uri,
        )
    return mgr


def append_fork_for_cleanup(forks_to_cleanup: List[str], tenderly_uri: str):
    """
    Appends the fork to the forks_to_cleanup list if it is not None.

    Parameters
    ----------
    forks_to_cleanup : List[str]
        The list of forks to cleanup.
    tenderly_uri : str
        The tenderly uri.

    Returns
    -------
    forks_to_cleanup : List[str]
        The list of forks to cleanup.

    """
    if tenderly_uri is not None:
        forks_to_cleanup.append(tenderly_uri.split("/")[-1])
    return forks_to_cleanup


def delete_tenderly_forks(forks_to_cleanup: List[str], mgr: Any) -> List[str]:
    """
    Deletes the forks that were created on Tenderly.

    Parameters
    ----------
    forks_to_cleanup : List[str]
        List of Tenderly fork names to delete.
    mgr : Any
        The manager object.
    """

    forks_to_keep = [forks_to_cleanup[-1], forks_to_cleanup[-2]]
    forks_to_cleanup = [fork for fork in forks_to_cleanup if fork not in forks_to_keep]

    # Delete the forks
    for fork in forks_to_cleanup:
        # Define your Tenderly credentials and project info
        tenderly_access_key = os.getenv("TENDERLY_ACCESS_KEY")
        tenderly_project = os.getenv("TENDERLY_PROJECT")

        # Define the headers for the request
        headers = {
            "X-Access-Key": tenderly_access_key,
            "Content-Type": "application/json",
        }

        url = f"https://api.tenderly.co/api/v2/project/{tenderly_project}/forks/{fork}"

        # Make the request to create the fork
        fork_response = requests.delete(url, headers=headers)

        mgr.cfg.logger.info(
            f"[events.utils.delete_tenderly_forks] Delete Fork {fork}, Response: {fork_response.status_code}"
        )

    return forks_to_keep


def verify_min_bnt_is_respected(bot: CarbonBot, mgr: Any):
    """
    Verifies that the bot respects the min profit. Used for testing.

    Parameters
    ----------
    bot : CarbonBot
        The bot object.
    mgr : Any
        The manager object.

    """
    # Verify MIN_PROFIT_BNT is set and respected
    assert (
        bot.ConfigObj.DEFAULT_MIN_PROFIT_GAS_TOKEN
        == mgr.cfg.DEFAULT_MIN_PROFIT_GAS_TOKEN
    ), "bot failed to update min profit"
    mgr.cfg.logger.debug(
        "[events.utils.verify_min_bnt_is_respected] Bot successfully updated min profit"
    )


def handle_target_token_addresses(static_pool_data: pd.DataFrame, target_tokens: List):
    """
    Get the addresses of the target tokens.

    Parameters
    ----------
    static_pool_data : pd.DataFrame
        The static pool data.
    target_tokens : List
        The target tokens.

    Returns
    -------
    List
        The addresses of the target tokens.

    """
    # Get the addresses of the target tokens
    target_token_addresses = []
    if target_tokens:
        for token in target_tokens:
            target_token_addresses = (
                target_token_addresses
                + static_pool_data[static_pool_data["tkn0_address"] == token][
                    "tkn0_address"
                ].tolist()
            )
            target_token_addresses = (
                target_token_addresses
                + static_pool_data[static_pool_data["tkn1_address"] == token][
                    "tkn1_address"
                ].tolist()
            )
    target_token_addresses = list(set(target_token_addresses))
    return target_token_addresses


def handle_replay_from_block(replay_from_block: int) -> (int, int, bool):
    """
    Handle the replay from block flag.

    Parameters
    ----------
    replay_from_block : int
        The block number to replay from.

    Returns
    -------
    polling_interval : int
        The time interval at which the bot polls for new events.

    """
    if replay_from_block:
        assert (
            replay_from_block > 0
        ), "The block number to replay from must be greater than 0."
    reorg_delay = 0
    use_cached_events = False
    polling_interval = 0
    return polling_interval, reorg_delay, use_cached_events


def handle_static_pools_update(mgr: Any):
    """
    Handles the static pools update 1x at startup and then periodically thereafter upon terraformer runs.

    Parameters
    ----------
    mgr : Any
        The manager object.

    """
    uniswap_v2_event_mappings = pd.DataFrame(
        [
            {"address": k, "exchange_name": v}
            for k, v in mgr.uniswap_v2_event_mappings.items()
        ]
    )
    uniswap_v3_event_mappings = pd.DataFrame(
        [
            {"address": k, "exchange_name": v}
            for k, v in mgr.uniswap_v3_event_mappings.items()
        ]
    )
    solidly_v2_event_mappings = pd.DataFrame(
        [
            {"address": k, "exchange_name": v}
            for k, v in mgr.solidly_v2_event_mappings.items()
        ]
    )
    all_event_mappings = (
        pd.concat(
            [
                uniswap_v2_event_mappings,
                uniswap_v3_event_mappings,
                solidly_v2_event_mappings,
            ]
        )
        .drop_duplicates("address")
        .to_dict(orient="records")
    )
    if "uniswap_v2_pools" not in mgr.static_pools:
        mgr.static_pools["uniswap_v2_pools"] = []
    if "uniswap_v3_pools" not in mgr.static_pools:
        mgr.static_pools["uniswap_v3_pools"] = []
    if "solidly_v2_pools" not in mgr.static_pools:
        mgr.static_pools["solidly_v2_pools"] = []

    for ex in mgr.forked_exchanges:
        if ex in mgr.exchanges:
            exchange_pools = [
                e["address"] for e in all_event_mappings if e["exchange_name"] == ex
            ]
            mgr.cfg.logger.info(
                f"[events.utils.handle_static_pools_update] Adding {len(exchange_pools)} {ex} pools to static pools"
            )
            attr_name = f"{ex}_pools"
            mgr.static_pools[attr_name] = exchange_pools


def handle_tokens_csv(mgr: BaseManager, prefix_path: str, read_only: bool = False):
    try:
        token_data = pd.read_csv(TOKENS_PATH)
    except Exception as e:
        if read_only:
            raise ReadOnlyException(TOKENS_PATH) from e

        mgr.logger.info(
            f"[events.utils.handle_tokens_csv] Error reading token data: {e}... creating new file"
        )
        token_data = pd.DataFrame(mgr.tokens)
        token_data.to_csv(TOKENS_PATH, index=False)
    extra_info = glob(
        os.path.normpath(
            f"{prefix_path}fastlane_bot/data/blockchain_data/{mgr.blockchain}/token_detail/*.csv"
        )
    )
    if len(extra_info) > 0:
        extra_info_df = pd.concat(
            [pd.read_csv(f) for f in extra_info], ignore_index=True
        )
        token_data = pd.concat([token_data, extra_info_df], ignore_index=True)
        token_data = token_data.drop_duplicates(subset=["address"])

        if not read_only:
            token_data.to_csv(TOKENS_PATH, index=False)

            # delete all files in token_detail
            for f in extra_info:
                try:
                    os.remove(f)
                except FileNotFoundError:
                    pass

    mgr.tokens = token_data.to_dict(orient="records")

    mgr.logger.info(
        f"[events.utils.handle_tokens_csv] Updated token data with {len(extra_info)} new tokens"
    )


def self_funding_warning_sequence(logger: logging.Logger):
    """
    This function initiates a warning sequence if the user has specified to use their own funds.

    """
    logger.info(
        f"\n\n*********************************************************************************\n"
        f"*********************************   WARNING   *********************************\n\n"
    )
    logger.info(
        f"Arbitrage bot is set to use its own funds instead of using Flashloans.\n\n*****   This could put your funds "
        f"at risk.    ******\nIf you did not mean to use this mode, cancel the bot now.\n\nOtherwise, the bot will "
        f"submit token approvals IRRESPECTIVE OF CURRENT GAS PRICE for each token specified in Flashloan "
        f"tokens.\n\n*********************************************************************************"
    )
    time.sleep(5)
    logger.info("Submitting approvals in 15 seconds")
    time.sleep(5)
    logger.info("Submitting approvals in 10 seconds")
    time.sleep(5)
    logger.info("Submitting approvals in 5 seconds")
    time.sleep(5)
    logger.info(
        f"*********************************************************************************\n\nSelf-funding mode "
        f"activated."
    )
    logger.info(
        f"""\n\n
          _____
         |A .  | _____
         | /.\ ||A ^  | _____
         |(_._)|| / \ ||A _  | _____
         |  |  || \ / || ( ) ||A_ _ |
         |____V||  .  ||(_'_)||( v )|
                |____V||  |  || \ / |
                       |____V||  .  |
                              |____V|
    \n\n"""
    )


def find_unapproved_tokens(tokens: List, tx_helpers) -> List:
    """
    This function checks if tokens have been previously approved from the wallet address to the Arbitrage contract.
    If they are not already approved, it will submit approvals for each token specified in Flashloan tokens.
    :param tokens: the list of tokens to check/approve
    :param tx_helpers: the TxHelpers instantiated class

    returns: List of tokens that have not been approved

    """
    unapproved_tokens = []
    for tkn in tokens:
        if not tx_helpers.check_if_token_approved(token_address=tkn):
            unapproved_tokens.append(tkn)
    return unapproved_tokens


def check_and_approve_tokens(
    mgr: Manager, tokens: List, logger: logging.Logger
) -> bool:
    """
    This function checks if tokens have been previously approved from the wallet address to the Arbitrage contract.
    If they are not already approved, it will submit approvals for each token specified in Flashloan tokens.

    """
    _tokens = []
    for tkn in tokens:
        # If the token is a token key, get the address from the CHAIN_FLASHLOAN_TOKENS dict in the network.py config
        # file
        if "-" in tkn:
            try:
                _tokens.append(mgr.blockchain.flashloan_tokens[tkn])
            except KeyError:
                mgr.logger.info(f"could not find token address for tkn: {tkn}")
        else:
            _tokens.append(tkn)
    tokens = _tokens

    self_funding_warning_sequence(mgr.logger)
    tx_helpers = TxHelpers(mgr)
    unapproved_tokens = find_unapproved_tokens(tokens=tokens, tx_helpers=tx_helpers)

    if len(unapproved_tokens) == 0:
        return True

    for _tkn in unapproved_tokens:
        tx = tx_helpers.approve_token_for_arb_contract(token_address=_tkn)
        if tx is not None:
            continue
        else:
            assert (
                False
            ), f"Failed to approve token: {_tkn}. This can be fixed by approving manually, or restarting the bot to try again."

    unapproved_tokens = find_unapproved_tokens(
        tokens=unapproved_tokens, cfg=cfg, tx_helpers=tx_helpers
    )
    if len(unapproved_tokens) == 0:
        return True
    else:
        return False


def find_latest_timestamped_folder(logging_path=None):
    """
    Find the latest timestamped folder in the given directory or the default directory.

    Args:
        logging_path (str, optional): The custom logging path where the timestamped folders are. Defaults to None.

    Returns:
        str: Path to the latest timestamped folder, or None if no folder is found.
    """
    search_path = logging_path if logging_path else "."
    search_path = os.path.join(search_path, "logs/*")
    list_of_folders = glob(search_path)

    if not list_of_folders:
        return None

    list_of_folders.sort(reverse=True)  # Sort the folders in descending order
    return list_of_folders[0]  # The first one is the latest


def get_tokens_data(args, base_path):
    tokens_filepath = os.path.join(base_path, "tokens.csv")
    if not os.path.exists(tokens_filepath):
        if args.read_only:
            raise ReadOnlyException(tokens_filepath)

        df = pd.DataFrame(columns=["address", "decimals"])
        df.to_csv(tokens_filepath)
    return read_csv_file(tokens_filepath)


def get_logging_header(
    args,
    bot_version,
    exchanges,
    tenderly_event_exchanges,
):
    # Get the current python version used
    python_version = sys.version
    python_info = sys.version_info
    os_system = platform.system()

    # Log the run configuration
    logging_header = f"""
            +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            Starting fastlane bot with the following configuration:
            bot_version: {bot_version}
            os_system: {os_system}
            python_version: {python_version}
            python_info: {python_info}

            logging_path: {args.logging_path}
            arb_mode: {args.arb_mode}
            blockchain: {args.blockchain}
            default_min_profit_gas_token: {args.default_min_profit_gas_token}
            exchanges: {exchanges}
            flashloan_tokens: {args.flashloan_tokens}
            target_tokens: {args.target_tokens}
            use_specific_exchange_for_target_tokens: {args.use_specific_exchange_for_target_tokens}
            loglevel: {args.loglevel}
            backdate_pools: {args.backdate_pools}
            max_block_fetch: {args.max_block_fetch}
            static_pool_data_filename: {args.static_pool_data_filename}
            n_jobs: {args.n_jobs}
            use_cached_events: {args.use_cached_events}
            randomizer: {args.randomizer}
            timeout: {args.timeout}
            tenderly_fork_id: {args.tenderly_fork_id}
            tenderly_event_exchanges: {tenderly_event_exchanges}
            increment_time: {args.increment_time}
            increment_blocks: {args.increment_blocks}
            pool_data_update_frequency: {args.pool_data_update_frequency}
            prefix_path: {args.prefix_path}
            version_check_frequency: {args.version_check_frequency}
            self_fund: {args.self_fund}
            read_only: {args.read_only}

            +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            Copy and paste the above configuration when reporting a bug. Please also include the error message and stack trace below:

            <INSERT ERROR MESSAGE AND STACK TRACE HERE>

            Please direct all questions/reporting to the Fastlane Telegram channel: https://t.me/BancorDevelopers

            +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            """
    return logging_header


def check_version_of_arb_contract(bancor_arbitrage_contract) -> Optional[str]:
    """
    Args:
        bancor_arbitrage_contract: The BancorArbitrage contract instance

    Returns:
        str: The version of the arbitrage contract

    """
    arb_contract_version = None

    if bancor_arbitrage_contract is not None:
        try:
            arb_contract_version = bancor_arbitrage_contract.caller.version()
        except Exception as e:
            # Failed to get latest version of arbitrage contract
            print(
                f"Failed to get latest version of arbitrage contract due to exception: {e}"
            )
            return

    return arb_contract_version
