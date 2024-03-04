from dataclasses import dataclass

from fastlane_bot.etl.models.exchanges.solidly.v2.pool import SolidlyV2Pool


@dataclass
class ScaleV2Pool(SolidlyV2Pool):
    """
    ScaleV2 pool class
    """
