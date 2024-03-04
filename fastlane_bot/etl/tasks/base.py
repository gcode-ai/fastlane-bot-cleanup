from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

from fastlane_bot.etl.models.manager import BaseManager


@dataclass
class BaseTask(ABC):
    """
    Base task class
    """

    mgr: BaseManager

    @abstractmethod
    def run(self, args: Any):
        """
        Run the task.
        """
        pass

    @property
    def w3(self):
        return self.mgr.w3

    @property
    def w3_async(self):
        return self.mgr.w3_async

    @property
    def w3_tenderly(self):
        return self.mgr.w3_tenderly
