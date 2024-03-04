from dataclasses import dataclass

from fastlane_bot.etl.models.exchanges.solidly.v2.pool import SolidlyV2Pool


@dataclass
class AerodromeV2Pool(SolidlyV2Pool):
    """
    Aerodrome pool class
    """
