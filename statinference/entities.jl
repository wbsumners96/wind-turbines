using DataFrames
using Dates

struct Turbine
    id::String
    easting::Float64
    northing::Float64
end

struct Measurement
    time::DateTime
    turbine_id::String
    turbulence_intensity::Float64
    windspeed::Float64
    power::Float64
    ambient_temperature::Float64
    wind_heading::Float64
    flag::Bool
end

function convert(::Type{A}, turbine_data) where A <: AbstractVector{Turbine}
    turbines = similar(A, nrow(turbine_data))
    turbines .= Turbine.(turbine_data[:, "id"], turbine_data[:, "easting"],
                         turbine_data[:, "northing"])

    return turbines
end

function convert(::Type{A}, measurement_data) where
        A <: AbstractVector{Measurement}
    measurements = similar(A, nrow(measurement_data))
    measurements .= Measurement.(measurement_data[:, "time"],
                                 measurement_data[:, "id"],
                                 measurement_data[:, "turbulence_intensity"],
                                 measurement_data[:, "windspeed"],
                                 measurement_data[:, "power"],
                                 measurement_data[:, "ambient_temperature"],
                                 measurement_data[:, "wind_heading"],
                                 measurement_data[:, "flag"])

    return measurements
end

