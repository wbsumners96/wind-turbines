using Dates
using Combinatorics
using LinearAlgebra
using Plots

struct Turbine
    id::String
    easting::Float64
    northing::Float64
end

Base.transpose(turbine::Turbine) = turbine

function displacement(turbine::Turbine, other::Turbine)
    return [turbine.easting - other.easting, turbine.northing - other.northing]
end

position(turbine::Turbine) = [turbine.easting, turbine.northing]

id(turbine) = turbine.id

easting(turbine) = turbine.easting

northing(turbine) = turbine.northing

struct Environment
    wind_heading::Float64
    windspeed::Float64
    temperature::Float64
    turbulence::Float64
end

struct Entry
    turbine::Turbine
    timestamp::DateTime
    environment::Environment
    power::Float64
    normal::Bool
end

function frontness(turbine::Turbine, wind_direction::Real)
    wind_vector = [sin(wind_direction), cos(wind_direction)]

    return -dot(position(turbine), wind_vector)
end

function shadows(turbine::Turbine, other::Turbine, wind_direction::Real;  
        shadow_radius::Real = pi/4)
    delta = displacement(turbine, other)
    wind_vector = [sin(wind_direction), cos(wind_direction)]
    if delta == 0
        angle = 0
    else 
        angle = acos(dot(delta, wind_vector)/norm(delta))
    end

    return -shadow_radius < angle < shadow_radius
end

function shadowing_matrix(turbines::AbstractVector{Turbine}, 
        wind_direction::Real; shadow_radius::Real = pi/4) 
    BitMatrix(shadows.(turbines, transpose(turbines), wind_direction; 
                       shadow_radius))
end

"""
    find_frontmost(turbines, wind_direction, count; shadow_radius)

Find the closest turbines to the origin of wind, subject to the condition that 
no turbine lie in the shadow of another.
"""
function find_frontmost(turbines::AbstractVector{Turbine}, wind_direction::Real,
        count::Integer; shadow_radius::Real=pi/4)
    shadowings = shadowing_matrix(turbines, wind_direction; shadow_radius)
    frontnesses = frontness.(turbines, wind_direction)
    
    max_frontness = -Inf
    optimal_indices = Nothing
    for indices in combinations(1:length(turbines), count)
        subarray = view(shadowings, indices, indices)
        if !any(subarray) && sum(frontnesses[indices]) > max_frontness
            max_frontness = sum(frontnesses[indices])
            optimal_indices = indices
        end
    end

    return view(turbines, optimal_indices)
end

# turbines = [Turbine("nicki", 0, 0)
#             Turbine("cardi", 32, 10)
#             Turbine("megan", 70, 70)
#             Turbine("tyler", 23, 100)
#             Turbine("wazingo", 318, -50)
#             Turbine("jogjgj", 7, -300)
#             Turbine("gorbulango", -328, -29)]
# frontmost_turbines = find_frontmost(turbines, pi, 3)
# print(frontmost_turbines)
# plot(easting.(frontmost_turbines), northing.(frontmost_turbines);
#      series_annotations=text.(id.(frontmost_turbines); pointsize=8, 
#                               rotation=30, valign=:top),
#      seriestype=:scatter, 
#      color=:red, 
#      label="Frontmost three turbines",
#      legend_position=:outerbottom)
# backmost_turbines = setdiff(turbines, frontmost_turbines)
# plot!(easting.(backmost_turbines), northing.(backmost_turbines); 
#      series_annotations=text.(id.(backmost_turbines); pointsize=5, 
#                               rotation=30, valign=:top),
#       seriestype=:scatter, color=:blue, label="Other turbines")
# quiver!([-300], [100]; quiver=([0], [-50]))
# 
# savefig("frontmost_turbines.png")
