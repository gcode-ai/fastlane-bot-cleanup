from typing import Dict

import pandas as pd
from hexbytes import HexBytes

from fastlane_bot.constants import MULTICHAIN_ADDRESS_PATH


def count_bytes(data: HexBytes) -> (int, int):
    """
    This function counts the number of zero and non-zero bytes in a given input data.
    """
    zero_bytes = len([byte for byte in data if byte == 0])
    non_zero_bytes = len(data) - zero_bytes
    return zero_bytes, non_zero_bytes


def int_prefix(string: str) -> int:
    return int(string[: -len(string.lstrip("0123456789"))])


def num_format(number):
    try:
        return "{0:.4f}".format(number)
    except Exception:
        return number


def num_format_float(number):
    try:
        return float("{0:.4f}".format(number))
    except Exception:
        return number


def log_format(log_data: {}, log_name: str = "new"):
    now = datetime.datetime.now()
    time_ts = str(int(now.timestamp()))  # timestamp (epoch)
    time_iso = now.isoformat().split(".")[0]

    return f"[{time_iso}::{time_ts}] |{log_name}| == {log_data}"


def get_multichain_addresses(network: str) -> pd.DataFrame:
    """
    Create dataframe of addresses for the selected network

    Args:
        network (str): the network to get the addresses for

    Returns:
        pd.DataFrame: the dataframe of addresses for the selected network
    """
    df = pd.read_csv(MULTICHAIN_ADDRESS_PATH)
    return df.loc[df["chain"] == network]


def get_fork_map(df: pd.DataFrame, fork_name: str) -> Dict:
    """
    Gets a dictionary mapping of the exchange forks based on CSV staticdata.

    Args:
        df (pd.DataFrame): the dataframe containing exchange details
        fork_name (str): the fork name

    Returns:
        Dict: a dictionary containing the exchange name and router address
    """
    fork_map = {}
    for exchange_name, fork, router_address in df[
        ["exchange_name", "fork", "router_address"]
    ].values:
        if fork in fork_name:
            fork_map[exchange_name] = router_address
    return fork_map


def get_factory_map(df: pd.DataFrame, fork_names: [str]) -> Dict:
    """
    Gets a dictionary mapping of the factory addresses based on CSV staticdata.

    Args:
        df (pd.DataFrame): the dataframe containing exchange details
        fork_names ([str]): the fork names

    Returns:
        Dict: a dictionary containing the factory address and exchange name
    """
    fork_map = {}
    for row in df.iterrows():
        exchange_name = row[1]["exchange_name"]
        fork = row[1]["fork"]
        factory_address = row[1]["factory_address"]
        if fork in fork_names:
            fork_map[factory_address] = exchange_name
            fork_map[exchange_name] = factory_address
    return fork_map


def get_fee_map(df: pd.DataFrame, fork_name: str) -> Dict:
    """
    Gets a dictionary mapping of the exchange fees based on CSV staticdata.

    Args:
        df (pd.DataFrame): the dataframe containing exchange details
        fork_name (str): the fork name

    Returns:
        Dict: a dictionary containing the exchange name and fee
    """
    fork_map = {}
    for row in df.iterrows():
        exchange_name = row[1]["exchange_name"]
        fork = row[1]["fork"]
        fee = row[1]["fee"]
        if fork in fork_name:
            fork_map[exchange_name] = fee
    return fork_map


def get_exchange_from_address(address: str, df: pd.DataFrame) -> str or None:
    """
    Get the exchange name from the given address.

    Args:
        address (str): the address to search for
        df (pd.DataFrame): the dataframe containing exchange details

    Returns:
        str or None: the exchange name if found, otherwise None
    """
    # Filter the DataFrame to find a row where either 'router_address' or 'factory_address' matches the given address.
    filtered_df = df[
        (df["router_address"] == address) | (df["factory_address"] == address)
    ]

    # If a matching row is found, return the 'exchange_name'; otherwise, return None.
    if not filtered_df.empty:
        return filtered_df.iloc[0]["exchange_name"]
    return None


def get_router_address_for_exchange(
    exchange_name: str, fork: str, df: pd.DataFrame
) -> str:
    """
    Get the router address for the given exchange and fork.

    Args:
        exchange_name (str): the name of the exchange
        fork (str): the fork name
        df (pd.DataFrame): the dataframe containing exchange details

    Returns:
        str: the router address
    """
    # Filter the DataFrame for the given exchange name and fork.
    df_filtered = df[(df["exchange_name"] == exchange_name) & (df["fork"] == fork)]

    # Check if any rows are found after filtering.
    if df_filtered.empty:
        raise ExchangeInfoNotFound(
            f"Router address could not be found for exchange: {exchange_name}, fork of: {fork}. Exchange must be mapped in fastlane_bot/staticdata/multichain_addresses.csv"
        )

    # Return the first router address found.
    return df_filtered["router_address"].values[0]


def get_fee_for_exchange(
    exchange_name: str, fork: str, df: pd.DataFrame
) -> float or None:
    """
    Get the fee for the given exchange and fork.

    Args:
        exchange_name (str): the name of the exchange
        fork (str): the fork name
        df (pd.DataFrame): the dataframe containing exchange details

    Returns:
        float or None: the fee if found, otherwise None
    """
    # Filter the DataFrame for the specified exchange name and fork.
    df_filtered = df[(df["exchange_name"] == exchange_name) & (df["fork"] == fork)]

    # If no matching rows are found, raise an exception.
    if df_filtered.empty:
        raise ExchangeInfoNotFound(
            f"Fee could not be found for exchange: {exchange_name}, fork of: {fork}. Exchange must be mapped in fastlane_bot/staticdata/multichain_addresses.csv"
        )

    # Return the fee from the first matching row.
    return df_filtered["fee"].values[0]
