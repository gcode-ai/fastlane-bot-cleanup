# coding=utf-8
"""
Contains the factory class for exchanges. This class is responsible for creating exchanges.

(c) Copyright Bprotocol foundation 2024.
Licensed under MIT
"""
from typing import Dict, Any
from fastlane_bot.etl.models.exchange import BaseExchange
from fastlane_bot.etl.models.pool import BasePool


class ExchangeFactory:
    """
    Factory class for exchanges
    """

    def __init__(self):
        self._creators = {}

    def register_exchange(self, key, creator):
        """
        Register an exchange with the factory

        Parameters
        ----------
        key : str
            The key to use for the exchange
        creator : Exchange
            The exchange class to register
        """
        self._creators[key] = creator

    def get_exchange(self, key, cfg: Any) -> BaseExchange:
        """
        Get an exchange from the factory

        Parameters
        ----------
        key : str
            The key to use for the exchange
        cfg : Any
            The Config object
        exchange_initialized : bool
            If the exchange has been initialized - this flag signals if an exchange that has staticdata updated through events has already been initialized in order to avoid duplicate event filters.
        Returns
        -------
        Exchange
            The exchange class
        """
        creator = self._creators.get(key)
        if not creator:
            fork_name = cfg.network.exchange_name_base_from_fork(exchange_name=key)
            if fork_name in key:
                raise ValueError(key)
            else:
                creator = self._creators.get(fork_name)

        # args = self.get_fork_extras(exchange_name=key, cfg=cfg, exchange_initialized=exchange_initialized)
        return creator


class PoolFactory:
    """
    Factory class for creating pools.
    """

    def __init__(self):
        self._creators = {}

    def register_format(self, format_name: str, creator: BasePool) -> None:
        """
        Register a pool type.

        Parameters
        ----------
        format_name : str
            The name of the pool type.
        creator : Pool
            The pool class.
        """
        self._creators[format_name] = creator

    def get_pool(self, format_name: str, cfg: Any) -> BasePool:
        """
        Get a pool.

        Parameters
        ----------
        format_name : str
            The name of the pool type.
        cfg : Any
            The config object
        """
        exchange_base = cfg.network.exchange_name_base_from_fork(exchange_name=format_name)
        creator = self._creators.get(exchange_base)

        if not creator:
            if format_name:
                raise ValueError(format_name)
        return creator


# create an instance of the factory
pool_factory = PoolFactory()
