# https://dev.intelrealsense.com/docs/post-processing-filters
from RealSense_Utilities.realsense_api.post_processing.option import OptionValues, OptionType, FilterOptions


#############################################################
# -------------------- DECIMATION OPTIONS -------------------#
#############################################################
class DecimationOptions(FilterOptions):
    def __init__(self) -> None:
        self.options: dict = {
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


#############################################################
# --------------------- SPATIAL OPTIONS ---------------------#
#############################################################
class SpatialOptions(FilterOptions):
    def __init__(self) -> None:
        self.options: dict = {
            OptionType.MAGNITUDE: OptionValues(
                option_value=2,
                option_value_increment=1,
                option_min_value=0,
                option_max_value=5
            ),
            OptionType.SMOOTH_ALPHA: OptionValues(
                option_value=0.5,
                option_value_increment=0.25,
                option_min_value=0.0,
                option_max_value=1.0
            ),
            OptionType.SMOOTH_DELTA: OptionValues(
                option_value=2,
                option_value_increment=1,
                option_min_value=0,
                option_max_value=5
            ),
            OptionType.HOLE_FILLING: OptionValues(
                option_value=0,
                option_value_increment=1,
                option_min_value=0,
                option_max_value=5
            )
        }

    def increment(self, option: OptionValues):
        option.option_value += option.option_value_increment
        if option.option_value > option.option_max_value:
            option.option_value = option.option_min_value


#############################################################
# --------------------- TEMPORAL OPTIONS --------------------#
#############################################################
class TemporalOptions(FilterOptions):
    def __init__(self) -> None:
        self.options: dict = {
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
            OptionType.PERSISTENCY_INDEX: OptionValues(
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


#############################################################
# ------------------- HOLE FILLING OPTIONS ------------------#
#############################################################
class HoleFillingOptions(FilterOptions):
    def __init__(self) -> None:
        self.options: dict = {
            OptionType.HOLE_FILLING: OptionValues(
                option_value=1,
                option_value_increment=1,
                option_min_value=0,
                option_max_value=2
            )
        }

    def increment(self, option: OptionValues) -> None:
        option.option_value += option.option_value_increment
        if option.option_value > option.option_max_value:
            option.option_value = option.option_min_value
