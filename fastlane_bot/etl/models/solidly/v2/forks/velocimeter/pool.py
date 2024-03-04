from dataclasses import dataclass

from fastlane_bot.etl.models.exchanges.solidly.v2.pool import SolidlyV2Pool


@dataclass
class VelocimeterV2Pool(SolidlyV2Pool):
    """
    VelocimeterV2 pool class
    """
