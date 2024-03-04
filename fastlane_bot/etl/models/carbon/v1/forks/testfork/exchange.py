from dataclasses import dataclass

from fastlane_bot.etl.models.carbon.v1.exchange import CarbonV1Exchange


@dataclass
class CarbonV1ForkExchange(CarbonV1Exchange):
    """
    Class representing a fork of Carbon v1 (for testing purposes).
    """

    name: str = "carbon_v1_fork"
