# https://dev.intelrealsense.com/docs/post-processing-filters
from option import OptionValues, OptionType, FilterOptions


class DecimationOptions(FilterOptions):
    def __init__(self) -> None:
        self.options: OptionDict = {
            OptionType.MAGNITUDE: OptionValues(
                option_value=2,
                option_value_increment=1,
                option_min_value=1,
                option_max_value=8
            )
        }

    def increment(self, option: OptionValues) -> None:
        option.option_value += option.option_value_increment
        if option.option_value > option.option_max_value:
            option.option_value = option.option_min_value
