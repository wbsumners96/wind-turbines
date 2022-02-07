using Base.Filesystem
using CSV
using DataFrames
using Dates
using Printf
using Query
using XLSX

mutable struct FarmData
    datapath::String
    code::String
    turbines::Union{Nothing, DataFrame}
    measurements::Union{Nothing, DataFrame}
end

FarmData(datapath, code) = FarmData(datapath, code, nothing, nothing)

function turbines(farm::FarmData)
    if !isnothing(farm.turbines)
        return farm.turbines
    end
    
    filename = "$(farm.code) Turbine Positions.xlsx"
    filepath = joinpath(farm.datapath, filename)
    turbines = DataFrame(XLSX.readtable(filepath, "positions")...)

    colnames = ["id", "location_type", "easting", "northing", "hub_height",
                "diameter", "altitude"]
    rename!(turbines, colnames)

    farm.turbines = turbines

    return turbines
end

_as_instanceID(farm::FarmData, i) = @sprintf "%s_WTG%02d" farm.code i

function turbine(farm::FarmData, id::String)
    return @from turbine in turbines(farm) begin
           @where turbine.id == id
           @select turbine
           @collect DataFrame
    end
end

function turbine(farm::FarmData, id::Integer)
    return turbine(farm, _as_instanceID(farm, id))
end

function measurements(farm::FarmData)
    if !isnothing(farm.measurements)
        return farm.measurements
    end
    
    data_filename = "$(farm.code)_Data.csv"
    data_filepath = joinpath(farm.datapath, data_filename)
    data = DataFrame(CSV.File(data_filepath))

    flag_filename = "$(farm.code)_Flag.csv"
    flag_filepath = joinpath(farm.datapath, flag_filename)
    flags = DataFrame(CSV.File(flag_filepath))

    measurements = innerjoin(data, flags, on=[:instanceID, :ts])

    colnames = ["time", "id", "turbulence_intensity", "windspeed", "power", 
                "ambient_temperature", "wind_heading", "flag"]
    rename!(measurements, colnames)

    farm.measurements = measurements

    return measurements
end

function measurements(farm::FarmData, turbine_id::String; normal::Bool=true)
    farm_measurements = measurements(farm)

    condition(mt) = mt.id == turbine_id && (normal ? mt.flag == 1 : true) 
    return @from mt in farm_measurements begin
           @where condition(mt)
           @select mt
           @collect DataFrame
    end
end

function measurements(farm::FarmData, turbine_id::Integer; normal::Bool=true)
    return measurements(farm, _as_instanceID(farm, turbine_id); normal)
end

function windspeeds(condition, farm::FarmData, turbine_id; normal::Bool=true)
    return @from mt in measurements(farm, turbine_id; normal) begin
           @where condition(mt)
           @select mt.windspeed
           @collect
    end
end

function windspeeds(farm, turbine_id; normal=true) 
    windspeeds(_ -> true, farm, turbine_id; normal)
end

function powers(condition, farm::FarmData, turbine_id; normal::Bool=true)
    return @from mt in measurements(farm, turbine_id; normal) begin
           @where condition(mt)
           @select mt[:power]
           @collect
    end
end

function powers(farm::FarmData, turbine_id::Integer; normal::Bool=true)
    return powers(_ -> true, farm, turbine_id; normal)
end
