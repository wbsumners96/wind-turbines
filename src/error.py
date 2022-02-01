class MissingTurbineError(Exception):
    """
    Given turbine IDs could not be retrieved from the provided data.
    """

    def __init__(self, missing_turbines):
        self.missing_turbines = missing_turbines


class MissingTimeError(Exception):
    """
    Given times could not be retrieved from the provided data.
    """

    def __init__(self, missing_times):
        self.missing_times = missing_times
