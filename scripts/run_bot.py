# coding=utf-8
"""
This is the main file for configuring the bot and running the fastlane bot.

(c) Copyright Bprotocol foundation 2024.
Licensed under MIT
"""
import argparse
import logging
import os
import time
from traceback import format_exc

import pandas as pd
from dotenv import load_dotenv
from web3 import AsyncHTTPProvider, AsyncWeb3, HTTPProvider, Web3

from fastlane_bot import __version__ as bot_version
from fastlane_bot.abi import (
    BANCOR_V3_NETWORK_INFO_ABI,
    FAST_LANE_CONTRACT_ABI,
    GAS_ORACLE_ABI,
)
from fastlane_bot.bot import CarbonBot
from fastlane_bot.constants import (
    BASE_PATH,
    SOLIDLY_V2_EVENT_MAPPINGS_PATH,
    TENDERLY_EVENT_EXCHANGES,
    TENDERLY_FORK_ID,
    UNISWAP_V2_EVENT_MAPPINGS_PATH,
    UNISWAP_V3_EVENT_MAPPINGS_PATH,
)
from fastlane_bot.etl.models.blockchain import Blockchain
from fastlane_bot.etl.models.manager import Manager
from fastlane_bot.etl.tasks.extract import ExtractTask
from fastlane_bot.etl.tasks.load import QueryInterface
from fastlane_bot.etl.utils import (
    add_initial_pool_data,
    check_and_approve_tokens,
    check_version_of_arb_contract,
    find_latest_timestamped_folder,
    get_logging_header,
    get_static_data,
    get_tokens_data,
    handle_flashloan_tokens,
    handle_static_pools_update,
    handle_subsequent_iterations,
    handle_target_token_addresses,
    handle_target_tokens,
    handle_tenderly_event_exchanges,
    handle_tokens_csv,
    verify_min_bnt_is_respected,
    verify_state_changed,
)

# from scripts.run_terraformer import terraform_blockchain

load_dotenv()


def main(args: argparse.Namespace) -> None:
    """
    Main function for running the fastlane bot.

    Args:
        args: Command line arguments. See the argparse.ArgumentParser in the __main__ block for details.
    """
    (
        args,
        last_block,
        last_block_queried,
        logger,
        loop_idx,
        mgr,
        provider_uri,
        start_timeout,
        total_iteration_time,
    ) = handle_setup(args)

    if args.is_args_test:
        return

    # Break if timeout is hit to test the bot flags
    if args.timeout == 1:
        logger.info("Timeout to test the bot flags")
        return

    while True:
        loop_idx += 1
        (
            iteration_start_time,
            last_block,
            initial_state,
            current_block,
        ) = perform_extraction_and_initialization(
            mgr,
            loop_idx,
            last_block,
            last_block_queried,
            total_iteration_time,
            start_timeout,
            args.provider_uri,
            args,
        )

        if iteration_start_time is None:
            continue  # Skip this iteration due to error

        bot = initialize_bot(mgr)
        handle_validations(bot, initial_state, mgr)
        handle_arb_contract_version_check(args, loop_idx, mgr)
        manage_tokens_and_iterations(bot, args, mgr, loop_idx)

        if check_timeout(args, start_timeout):
            break

        if loop_idx == 1:
            mgr.logger.info(
                "Finished first iteration of data sync. Now starting main loop."
            )

        manage_sleep_and_logging(args, mgr)


def process_arguments(args: argparse.Namespace) -> argparse.Namespace:
    """
    Process and transform command line arguments.

    Args:
        args: Command line arguments.

    Returns:
        args: Processed command line arguments.
    """

    def is_true(x):
        return x.lower() == "true"

    def int_or_none(x):
        return int(x) if x else None

    # Define the transformations for each argument
    transformations = {
        "backdate_pools": is_true,
        "n_jobs": int,
        "polling_interval": int,
        "alchemy_max_block_fetch": int,
        "reorg_delay": int,
        "use_cached_events": is_true,
        "run_data_validator": is_true,
        "randomizer": int,
        "limit_bancor3_flashloan_tokens": is_true,
        "timeout": int_or_none,
        "replay_from_block": int_or_none,
        "increment_time": int,
        "increment_blocks": int,
        "pool_data_update_frequency": int,
        "version_check_frequency": int,
        "self_fund": is_true,
        "read_only": is_true,
        "is_args_test": is_true,
    }

    # Apply the transformations
    for arg, transform in transformations.items():
        if hasattr(args, arg):
            setattr(args, arg, transform(getattr(args, arg)))

    return args


def check_timeout(args, start_timeout):
    return args.timeout is not None and time.time() - start_timeout > args.timeout


def manage_tokens_and_iterations(bot, args, mgr, loop_idx):
    # Optionally filter target tokens based on a specific exchange
    if args.use_specific_exchange_for_target_tokens is not None:
        target_tokens = bot.get_tokens_in_exchange(
            exchange_name=args.use_specific_exchange_for_target_tokens
        )
        mgr.logger.info(
            f"[main] Using only tokens in: {args.use_specific_exchange_for_target_tokens}, found {len(target_tokens)} tokens"
        )
    else:
        target_tokens = None  # or set to a default list of tokens if necessary

    # Handle tokens CSV file if not in read-only mode
    if not mgr.read_only:
        handle_tokens_csv(mgr, mgr.prefix_path)

    # Perform actions necessary for handling subsequent iterations
    handle_subsequent_iterations(
        arb_mode=args.arb_mode,
        bot=bot,
        flashloan_tokens=args.flashloan_tokens,
        polling_interval=args.polling_interval,
        randomizer=args.randomizer,
        run_data_validator=args.run_data_validator,
        target_tokens=target_tokens,
        loop_idx=loop_idx,
        logging_path=args.logging_path,
        replay_from_block=args.replay_from_block,
        tenderly_uri=args.tenderly_uri,
        mgr=mgr,
        forked_from_block=args.forked_from_block,
    )


def initialize_bot(mgr):
    bot = CarbonBot()
    bot.db = QueryInterface(mgr=mgr)
    return bot


def manage_sleep_and_logging(args, mgr):
    # Handles sleeping for the polling interval and logging
    if args.polling_interval > 0:
        mgr.logger.info(f"Sleeping for {args.polling_interval} seconds...")
        time.sleep(args.polling_interval)


def perform_extraction_and_initialization(
    mgr,
    loop_idx,
    last_block,
    last_block_queried,
    total_iteration_time,
    start_timeout,
    provider_uri,
    args,
) -> tuple:
    try:
        iteration_start_time, last_block, initial_state, current_block = ExtractTask(
            mgr=mgr,
            loop_idx=loop_idx,
            last_block=last_block,
            last_block_queried=last_block_queried,
            total_iteration_time=total_iteration_time,
            start_timeout=start_timeout,
            provider_uri=provider_uri,
        ).run(args)
        return iteration_start_time, last_block, initial_state, current_block
    except Exception as e:
        handle_extraction_error(e, mgr, args)
        return None, None, None, None


def handle_extraction_error(e, mgr, args):
    mgr.logger.error(f"Error in extraction and initialization: {format_exc()}")
    mgr.logger.error(
        f"[main] Error during extraction: {e}. Continuing... "
        f"Please report this error."
        f"{args.logging_header}"
    )


def handle_arb_contract_version_check(args, loop_idx, mgr):
    if (
        loop_idx % args.version_check_frequency == 0
        and args.version_check_frequency != -1
        and args.blockchain in "ethereum"
    ):
        # Check the version of the deployed arbitrage contract
        arb_contract_version = check_version_of_arb_contract(mgr.arb_contract)
        mgr.logger.info(
            f"[main] Checking latest version of Arbitrage Contract. Found version: {arb_contract_version}"
        )


def handle_tenderly_time_and_block(args):
    w3 = Web3(HTTPProvider(args.tenderly_uri))
    # Increase time and blocks
    params = [w3.to_hex(args.increment_time)]  # number of seconds
    w3.provider.make_request(method="evm_increaseTime", params=params)
    params = [w3.to_hex(args.increment_blocks)]  # number of blocks
    w3.provider.make_request(method="evm_increaseBlocks", params=params)


def handle_setup(args):
    args.tenderly_fork_id = TENDERLY_FORK_ID
    args.tenderly_event_exchanges = TENDERLY_EVENT_EXCHANGES
    args = process_arguments(args)
    logger = logging.getLogger(__name__)
    tokens = get_tokens_data(args, BASE_PATH.replace("{{blockchain}}", args.blockchain))
    blockchain = Blockchain(args.blockchain)
    if args.flashloan_tokens is None:
        args.flashloan_tokens = blockchain.flashloan_tokens
    else:
        args.flashloan_tokens = handle_flashloan_tokens(args.flashloan_tokens, tokens)
        blockchain._flashloan_tokens = args.flashloan_tokens
    # Search the logging directory for the latest timestamped folder
    args.logging_path = find_latest_timestamped_folder(args.logging_path)
    # Format the target tokens
    args.target_tokens = handle_target_tokens(args.flashloan_tokens, args.target_tokens)
    # Format the exchanges
    exchanges = args.exchanges.split(",") if args.exchanges else []
    # Format the tenderly event exchanges
    tenderly_event_exchanges = handle_tenderly_event_exchanges(
        args.tenderly_event_exchanges, args.tenderly_fork_id
    )
    logging_header = get_logging_header(
        args,
        bot_version,
        exchanges,
        tenderly_event_exchanges,
    )
    logger.info(logging_header)
    # Get the static pool data, tokens and uniswap v2 event mappings
    (
        static_pool_data,
        tokens,
        uniswap_v2_event_mappings,
        uniswap_v3_event_mappings,
        solidly_v2_event_mappings,
    ) = get_static_data(
        exchanges, args.blockchain, args.static_pool_data_filename, args.read_only
    )
    target_token_addresses = handle_target_token_addresses(
        static_pool_data, args.target_tokens
    )
    w3, w3_async, w3_tenderly = get_web3_instances(args)
    # Initialize data fetch manager
    mgr = Manager(
        _w3=w3,
        _w3_async=w3_async,
        _w3_tenderly=w3_tenderly,
        logger=logger,
        logging_path=args.logging_path,
        pool_data=static_pool_data.to_dict(orient="records"),
        tokens=tokens.to_dict(orient="records"),
        target_tokens=target_token_addresses,
        blockchain=args.blockchain,
        prefix_path=args.prefix_path,
        read_only=args.read_only,
        self_fund=args.self_fund,
        uniswap_v2_event_mappings=uniswap_v2_event_mappings,
        uniswap_v3_event_mappings=uniswap_v3_event_mappings,
        solidly_v2_event_mappings=solidly_v2_event_mappings,
    )
    if args.self_fund:
        check_and_approve_tokens(mgr=mgr, tokens=args.flashloan_tokens)
    # Add initial pool data to the manager
    add_initial_pool_data(mgr, args.n_jobs)
    # Set the main loop variables
    loop_idx = last_block = last_block_queried = total_iteration_time = 0
    start_timeout = time.time()
    provider_uri = mgr.w3.provider.endpoint_uri
    handle_static_pools_update(mgr)

    arb_contract = mgr.w3.eth.contract(
        address=mgr.blockchain.fastlane_contract_address,
        abi=FAST_LANE_CONTRACT_ABI,
    )
    mgr.arb_contract = arb_contract

    bancor_network_info_contract = mgr.w3.eth.contract(
        address=mgr.blockchain.bancor_network_info_contract_address,
        abi=BANCOR_V3_NETWORK_INFO_ABI,
    )
    mgr.bancor_network_info_contract = bancor_network_info_contract

    gas_oracle_contract = mgr.w3_async.eth.contract(
        address=mgr.blockchain.gas_oracle_address,
        abi=GAS_ORACLE_ABI,
    )
    mgr.gas_oracle_contract = gas_oracle_contract

    return (
        args,
        last_block,
        last_block_queried,
        logger,
        loop_idx,
        mgr,
        provider_uri,
        start_timeout,
        total_iteration_time,
    )


def handle_terraformer(
    args,
    current_block,
    iteration_start_time,
    last_block_queried,
    loop_idx,
    mgr,
    total_iteration_time,
):
    if (
        loop_idx % args.pool_data_update_frequency == 0
        and args.pool_data_update_frequency != -1
    ):
        mgr.logger.info(
            f"[main] Terraforming {args.blockchain}. Standby for oxygen levels."
        )
        sblock = (
            (current_block - (current_block - last_block_queried))
            if loop_idx > 1
            else None
        )
        # (
        #     exchange_df,
        #     uniswap_v2_event_mappings,
        #     uniswap_v3_event_mappings,
        #     solidly_v2_event_mappings,
        # ) = terraform_blockchain(
        #     network_name=args.blockchain,
        #     web3=mgr.w3,
        #     start_block=sblock,
        # )
        uniswap_v2_event_mappings = pd.read_csv(
            UNISWAP_V2_EVENT_MAPPINGS_PATH.replace("{{blockchain}}", args.blockchain)
        )
        uniswap_v3_event_mappings = pd.read_csv(
            UNISWAP_V3_EVENT_MAPPINGS_PATH.replace("{{blockchain}}", args.blockchain)
        )
        solidly_v2_event_mappings = pd.read_csv(
            SOLIDLY_V2_EVENT_MAPPINGS_PATH.replace("{{blockchain}}", args.blockchain)
        )
        mgr.uniswap_v2_event_mappings = dict(
            uniswap_v2_event_mappings[["address", "exchange"]].values
        )
        mgr.uniswap_v3_event_mappings = dict(
            uniswap_v3_event_mappings[["address", "exchange"]].values
        )
        mgr.solidly_v2_event_mappings = dict(
            solidly_v2_event_mappings[["address", "exchange"]].values
        )

    last_block_queried = current_block
    total_iteration_time += time.time() - iteration_start_time
    mgr.logger.info(
        f"\n\n********************************************\n"
        f"Average Total iteration time for loop {loop_idx}: {total_iteration_time / loop_idx}\n"
        f"bot_version: {bot_version}\n"
        f"\n********************************************\n\n"
    )
    return last_block_queried, total_iteration_time


def get_web3_instances(args):
    w3_tenderly = (
        Web3(HTTPProvider(f"https://rpc.tenderly.co/fork/{args.tenderly_fork_id}"))
        if args.tenderly_fork_id
        else None
    )
    # Initialize the web3 provider
    w3 = Web3(HTTPProvider(args.rpc_url))
    w3_async = AsyncWeb3(AsyncHTTPProvider(args.rpc_url))
    return w3, w3_async, w3_tenderly


def handle_validations(bot, initial_state, mgr):
    assert isinstance(
        bot.db, QueryInterface
    ), "QueryInterface not initialized correctly"
    # Verify that the state has changed
    verify_state_changed(bot=bot, initial_state=initial_state, mgr=mgr)
    # Verify that the minimum profit in BNT is respected
    verify_min_bnt_is_respected(bot=bot, mgr=mgr)


# # Run the main loop
# while True:
#     try:
#         (
#             iteration_start_time,
#             last_block,
#             initial_state,
#             current_block,
#         ) = ExtractTask(
#             mgr=mgr,
#             loop_idx=loop_idx,
#             last_block=last_block,
#             last_block_queried=last_block_queried,
#             total_iteration_time=total_iteration_time,
#             start_timeout=start_timeout,
#             provider_uri=provider_uri,
#         ).run(
#             args
#         )
#
#         # Re-initialize the bot
#         bot = CarbonBot()
#         bot.db = QueryInterface(
#             mgr=mgr,
#         )
#
#         handle_validations(bot, initial_state, mgr)
#
#         if args.use_specific_exchange_for_target_tokens is not None:
#             target_tokens = bot.get_tokens_in_exchange(
#                 exchange_name=args.use_specific_exchange_for_target_tokens
#             )
#             mgr.logger.info(
#                 f"[main] Using only tokens in: {args.use_specific_exchange_for_target_tokens}, found {len(target_tokens)} tokens"
#             )
#
#         if not mgr.read_only:
#             handle_tokens_csv(mgr, mgr.prefix_path)
#
#         # Handle subsequent iterations
#         handle_subsequent_iterations(
#             arb_mode=args.arb_mode,
#             bot=bot,
#             flashloan_tokens=args.flashloan_tokens,
#             polling_interval=args.polling_interval,
#             randomizer=args.randomizer,
#             run_data_validator=args.run_data_validator,
#             target_tokens=args.target_tokens,
#             loop_idx=loop_idx,
#             logging_path=args.logging_path,
#             replay_from_block=args.replay_from_block,
#             tenderly_uri=args.tenderly_uri,
#             mgr=mgr,
#             forked_from_block=args.forked_from_block,
#         )
#
#         # Sleep for the polling interval
#         if not args.replay_from_block and args.polling_interval > 0:
#             mgr.logger.info(
#                 f"[main] Sleeping for polling_interval={args.polling_interval} seconds..."
#             )
#             time.sleep(args.polling_interval)
#
#         # Check if timeout has been hit, and if so, break the loop for tests
#         if args.timeout is not None and time.time() - start_timeout > args.timeout:
#             mgr.logger.info("[main] Timeout hit... stopping bot")
#             break
#
#         if loop_idx == 1:
#             mgr.logger.info(
#                 """
#               +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#               +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
#               Finished first iteration of data sync. Now starting main loop arbitrage search.
#
#               +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#               +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#               """
#             )
#
#         if args.tenderly_fork_id:
#             handle_tenderly_time_and_block(args)
#
#         handle_arb_contract_version_check(arb_contract, args, loop_idx, mgr)
#
#         last_block_queried, total_iteration_time = handle_terraformer(
#             args,
#             current_block,
#             iteration_start_time,
#             last_block_queried,
#             loop_idx,
#             mgr,
#             total_iteration_time,
#         )
#
#     except Exception as e:
#         mgr.logger.error(f"Error in main loop: {format_exc()}")
#         mgr.logger.error(
#             f"[main] Error in main loop: {e}. Continuing... "
#             f"Please report this error to the Fastlane Telegram channel if it persists."
#             f"{args.logging_header}"
#         )
#         time.sleep(args.polling_interval)
#         if args.timeout is not None and time.time() - start_timeout > args.timeout:
#             mgr.logger.info("Timeout hit... stopping bot")
#             mgr.logger.info("[main] Timeout hit... stopping bot")
#             break
