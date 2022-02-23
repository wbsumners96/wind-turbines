import numpy as np


def lies_in_wake(turbine_position, other_position, wind_heading,
        blade_diameter=80):
    """
    Determine if turbine 'other' lies in the wake of turbine 'turbine'
    according to IEC 61400-12-2.

    Parameters
    ----------
    turbine_position: length 2 numpy vector
        Position of turbine, with easting (in meters) in the first component 
        and northing (in meters) in the second.

    other_position: length 2 numpy vector
        Position of other.

    wind_heading: float
        Direction of the wind in degrees from north.

    blade_diameter: float (default = 80)
        Diameter of turbine blades in meters.
    """
    def iec_function(blade_diameters):
        return (180*1.3/np.pi)*np.arctan(2.5/blade_diameters + 0.15) + 10

    wind_vector = np.array([np.sin(np.pi*wind_heading/180),
                            np.cos(np.pi*wind_heading/180)])
    turbine_displacement = other_position - turbine_position
    angle_to_wind = np.dot(turbine_displacement, 
            wind_vector)/np.linalg.norm(turbine_displacement)

    turbine_distance = np.linalg.norm(turbine_displacement)/blade_diameter 

    if turbine_distance <= 2:
        return True
    if turbine_distance > 20:
        return False

    return angle_to_wind <= iec_function(turbine_distance)

