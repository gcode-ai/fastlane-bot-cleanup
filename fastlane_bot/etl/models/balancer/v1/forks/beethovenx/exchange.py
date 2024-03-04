# coding=utf-8
"""
Contains the exchange class for Balancer weighted pools.

(c) Copyright Bprotocol foundation 2024.
Licensed under MIT
"""
from dataclasses import dataclass

from fastlane_bot.etl.models.balancer.v1.exchange import BalancerV1Exchange


@dataclass
class BeethovenxExchange(BalancerV1Exchange):
    """
    Beethovenx exchange class.
    """

    name: str = "beethovenx"
