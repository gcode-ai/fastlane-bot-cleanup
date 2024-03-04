# coding=utf-8
"""
This is the main file for configuring the bot and running the fastlane bot.

(c) Copyright Bprotocol foundation 2024.
Licensed under MIT
"""
import argparse
import importlib

from fastlane_bot.constants import CommonEthereumTokens

ethereum_tokens = CommonEthereumTokens()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--script",
        default="run_bot",
        help="The script to run. See the README for more information.",
    )
    parser.add_argument(
        "--backdate_pools",
        default="False",
        help="Set to False for faster testing / debugging",
    )
    parser.add_argument(
        "--max_block_fetch",
        default=2000,
        help="The maximum number of blocks to fetch.",
    )
    parser.add_argument(
        "--static_pool_data_filename",
        default="static_pool_data",
        help="Filename of the static pool data.",
    )
    parser.add_argument(
        "--arb_mode",
        default="multi_pairwise_all",
        help="See arb_mode in bot.py",
        choices=[
            "single",
            "multi",
            "triangle",
            "multi_triangle",
            "b3_two_hop",
            "multi_pairwise_pol",
            "multi_pairwise_all",
        ],
    )
    parser.add_argument(
        "--flashloan_tokens",
        default=None,
        help="A comma-separated string of tokens to flashloan. If not set, the bot will flashloan all tokens for the "
        "blockchain, as specified in the constants.py file.",
    )
    parser.add_argument("--n_jobs", default=-1, help="Number of parallel jobs to run")
    parser.add_argument(
        "--exchanges",
        default=None,
        help="Comma separated external exchanges. If not set, the bot will use all supported exchanges for the "
        "blockchain.",
    )
    parser.add_argument("--logging_path", default="", help="The logging path.")
    parser.add_argument(
        "--loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="The logging level.",
    )
    parser.add_argument(
        "--use_cached_events",
        default="False",
        help="Set to True for debugging / testing. Set to False for production.",
    )
    parser.add_argument(
        "--randomizer",
        default="3",
        help="Set to the number of arb opportunities to pick from.",
    )
    parser.add_argument(
        "--default_min_profit_gas_token",
        default="0.01",
        help="Set to the default minimum profit in gas token.",
    )
    parser.add_argument(
        "--timeout",
        default=None,
        help="Set to the timeout in seconds. Set to None for no timeout.",
    )
    parser.add_argument(
        "--target_tokens",
        default=None,
        help="A comma-separated string of tokens to target.",
    )
    parser.add_argument(
        "--increment_time",
        default=1,
        help="If tenderly_fork_id is set, this is the number of seconds to increment the fork time by for each "
        "iteration.",
    )
    parser.add_argument(
        "--increment_blocks",
        default=1,
        help="If tenderly_fork_id is set, this is the number of blocks to increment the block number "
        "by for each iteration.",
    )
    parser.add_argument(
        "--blockchain",
        default="ethereum",
        help="A blockchain from the list. Blockchains not in this list do not have a deployed Fast Lane contract and "
        "are not supported.",
        choices=["ethereum", "coinbase_base", "fantom"],
    )
    parser.add_argument(
        "--pool_data_update_frequency",
        default=-1,
        help="How frequently pool data should be updated, in main loop iterations.",
    )
    parser.add_argument(
        "--use_specific_exchange_for_target_tokens",
        default=None,
        help="If an exchange is specified, this will limit the scope of tokens to the tokens found on the exchange",
    )
    parser.add_argument(
        "--prefix_path",
        default="",
        help="Prefixes the path to the write folders (used for deployment)",
    )
    parser.add_argument(
        "--version_check_frequency",
        default=1,
        help="How frequently pool data should be updated, in main loop iterations.",
    )
    parser.add_argument(
        "--self_fund",
        default="False",
        help="If True, the bot will attempt to submit arbitrage transactions using funds in your "
        "wallet when possible.",
    )
    parser.add_argument(
        "--read_only",
        default="True",
        help="If True, the bot will skip all operations which write to disk. Use this flag if you're "
        "running the bot in an environment with restricted write permissions.",
    )
    parser.add_argument(
        "--is_args_test",
        default="False",
        help="The logging path.",
    )
    parser.add_argument(
        "--rpc_url",
        default=None,
        help="Custom RPC URL. If not set, the bot will use the default Alchemy RPC URL for the blockchain (if "
        "available).",
    )

    # Process the arguments
    args = parser.parse_args()
    script = importlib.import_module(f"scripts.{args.script}")
    script.main(args)
