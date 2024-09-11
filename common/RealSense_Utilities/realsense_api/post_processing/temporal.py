# https://dev.intelrealsense.com/docs/post-processing-filters
from option import OptionValues, OptionType, OptionDict, FilterOptions


class TemporalOptions(FilterOptions):
    def __init__(self) -> None:
        self.options: OptionDict = {
            OptionType.SMOOTH_ALPHA: OptionValues(
                option_value=0.4,
                option_value_increment=0.1,
                option_min_value=0.0,
                option_max_value=1.0
            ),
            OptionType.SMOOTH_DELTA: OptionValues(
                option_value=20,
                option_value_increment=10,
                option_min_value=0,
                option_max_value=100
            ),
            OptionType.HOLE_FILLING: OptionValues(
                option_value=3,
                option_value_increment=1,
                option_min_value=0,
                option_max_value=8
            )
        }

    def increment(self, option: OptionValues) -> None:
        option.option_value += option.option_value_increment
        if option.option_value > option.option_max_value:
            option.option_value = option.option_min_value
