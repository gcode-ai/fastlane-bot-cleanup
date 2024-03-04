from dataclasses import dataclass

from fastlane_bot.etl.models.carbon.v1.pool import CarbonV1Pool


@dataclass
class CarbonV1ForkPool(CarbonV1Pool):
    """
    Class representing a fork of Carbon v1 pool (for testing purposes).
    """
