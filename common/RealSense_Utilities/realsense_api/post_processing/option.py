from typing import TypedDict
from dataclasses import dataclass
from enum import Enum, auto
from abc import ABC, abstractmethod


class OptionType(Enum):
    MAGNITUDE = auto()
    SMOOTH_ALPHA = auto()
    SMOOTH_DELTA = auto()
    HOLE_FILLING = auto()
    PERSISTENCY_INDEX = auto()


@dataclass
class OptionValues:
    option_value: float
    option_value_increment: float
    option_min_value: float
    option_max_value: float


# class OptionDict(TypedDict):
#     option: OptionType
#     properties: OptionValues


class FilterOptions(ABC):
    @abstractmethod
    def increment(self, option: OptionValues) -> None:
        pass
