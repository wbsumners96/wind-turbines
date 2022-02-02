import numpy as np
import itertools


def frontness(turbine_position, wind_direction):
    wind_vector = np.array(np.sin(wind_direction), np.cos(wind_direction))
    
    return -np.dot(wind_vector, turbine_position)


def lies_in_shadow(shadow_turbine_position, other_turbine_position, 
        wind_direction, shadow_radius):
    turbine_displacement = other_turbine_position - shadow_turbine_position
    wind_vector = np.array(np.sin(wind_direction), np.cos(wind_direction))
    if turbine_displacement == 0:
        displacement_angle = 0
    else:
        displacement_angle = np.arccos(np.dot(wind_vector,
            turbine_displacement)/np.linalg.norm(turbine_displacement))

    return -shadow_radius < displacement_angle < shadow_radius


def shadowing_matrix(turbine_positions, wind_direction, shadow_radius=np.pi/4):
    return np.array([turbine.shadows(other, wind_direction, shadow_radius) for 
            turbine, other in itertools.product(turbine_positions,
                turbine_positions)])


def find_frontmost(turbine_positions, wind_direction, count, 
        shadow_radius=np.pi/4):
    """
    Maximize the total frontness of a subset of turbines, such that no turbine
    lies in the shadow of another.

    Parameters
    ----------
    turbine_positions: np.array
        Position of the turbines, each position with format [easting, northing].
    wind_direction: float
        Heading of wind, in radians relative to true north.
    count: int
        Size of resulting subset.
    shadow_radius: float
        Size of shadow, in radians.

    Returns
    -------
    optimal_indices: list of int | None
        Indices of frontmost turbine positions if a suitable subset was found,
        or None otherwise.
    """
    shadows =  shadowing_matrix(turbine_positions, wind_direction, 
            shadow_radius)
    frontnesses = np.array([turbine.frontness(wind_direction) for turbine in
        turbine_positions])

    max_frontness = -np.inf
    optimal_indices = None
    for indices in itertools.combinations(range(len(turbine_positions)), count):
        subarray = shadows[indices, indices]
        if not any(subarray) and sum(frontnesses[indices]) > max_frontness:
            max_frontness = sum(frontnesses[indices])
            optimal_indices = indices

    return optimal_indices

