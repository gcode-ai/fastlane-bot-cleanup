# coding=utf-8
"""
Contains the pool class for Balancer.

(c) Copyright Bprotocol foundation 2024.
Licensed under MIT
"""
from dataclasses import dataclass

from fastlane_bot.etl.models.balancer.v1.pool import BalancerV1Pool


@dataclass
class BeethovenxPool(BalancerV1Pool):
    """
    Beethovenx exchange class.
    """
