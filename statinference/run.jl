using Blink
using Distributions
using Interact
using Query
import Plots: histogram
using Printf
using StatsPlots

include("data.jl")
include("entities.jl")
include("stats.jl")
include("turbine_selector.jl")

farm = Nothing

function load_farm!(datapath, code)
    global farm 
    farm = FarmData(datapath, code)
end

function find_frontmost(farm::FarmData, wind_heading, count;
        shadow_radius = π/4)
    ts = convert(Vector{Turbine}, turbines(farm))
    frontmost_ids = [t.id for t in find_frontmost(ts, wind_heading, count;
                                                  shadow_radius)]

    return subset(turbines(farm), :id => ByRow(id -> id ∈ frontmost_ids))
end

function plot_frontmost(farm::FarmData, wind_heading, count; shadow_radius=π/4)
    fm = find_frontmost(farm, wind_heading, count; shadow_radius)

    function ray!(start, heading)
        ts = [0, 2000]
        xs = start[1] .+ ts .* sin(heading)
        ys = start[2] .+ ts .* cos(heading)
        
        plot!(xs, ys, lw=2, color=:gray)
    end

    plot(aspect_ratio=:equal,
         xticks=-1000:200:1000,
         yticks=-1200:200:400,
         xlims=(-1000, 1000),
         ylims=(-1200, 400))

    starts = [fm[:, :easting] fm[:, :northing]]
    for start in eachslice(starts; dims=1)
        ray!(start, wind_heading - shadow_radius)
        ray!(start, wind_heading + shadow_radius)
    end

    @df turbines(farm) scatter!(:easting, :northing, aspect_ratio=:equal,
                               xticks=-1000:200:1000,
                               yticks=-1200:200:400,
                               markersize=1)
    @df fm scatter!(:easting, :northing, color=:red, marksersize=2)
end

function frontmost_widget(farm)
    widget = @manipulate for h in 0:5:360, c in 1:5
        plot_frontmost(farm, π*h/180, c; shadow_radius=π/6)
    end

    w = Window()
    body!(w, widget)
end

function modelpredict(weighting, farm::FarmData, targets, references, times)
    targets = subset(turbines(farm), :id => ByRow(id -> in(id, targets)))
    references = subset(turbines(farm), :id => ByRow(id -> in(id, references)))

    function weight(easting_tar, easting_ref, northing_tar, northing_ref)
        d = sqrt((easting_tar - easting_ref)^2 +
                 (northing_tar - northing_ref)^2)

        return weighting(d)
    end

    weights = crossjoin(targets, references, makeunique=true)
    select!(weights, [:id, :id_1, :easting,
                      :easting_1, :northing, :northing_1] =>
            ByRow((i1, i2, e1, e2, n1, n2) ->
                  (i1, i2, weight(e1, e2, n1, n2))) => 
            [:target_id, :reference_id, :weight])

    println(weights)
end

function infer(condition, estimator_type, turbine_id)
    ps = windspeeds(condition, farm, turbine_id; normal=true)
    ps .+= 0.01 - minimum(ps) 

    mle = fit(estimator_type, ps)

    histogram(ps, normalize=true)
    xs = 0:0.1:20
    plot!(xs, pdf.(mle, xs))
end

function power_histogram(farm, turbine, windspeed, heading)
    ms = measurements(farm, turbine)
    subset!(ms, 
            [:windspeed, :wind_heading] => 
            (ws, h) -> ws .== windspeed .&& heading - 30 .<= h .< heading + 30)

    baseline = parse(DateTime, "2019-08-01")
    phase1 = parse(DateTime, "2020-09-10")

    configs = []
    push!(configs, subset(ms, :time => time -> time .<= baseline))
    push!(configs, subset(ms, :time => time -> baseline .< time .<= phase1))
    push!(configs, subset(ms, :time => time -> time .> phase1))
    
    colors = [:red, :yellow, :green]
    labels = ["Baseline", "Intermediate", "Phase 1"]
    title = @sprintf("turbine = %s,\nwindspeed = %.1f m/s,\nheading = %d°",
                     _as_instanceID(farm, turbine),
                     windspeed,
                     heading)
    plt = plot(; xlabel="Power (kW)",
               ylabel="Frequency",
               ylims=(0, 0.02),
               xlims=(0, 1500),
               title)
    for i in 1:length(configs)
        if !isempty(configs[i])
            @df configs[i] histogram!(plt, :power,
                                      color=colors[i],
                                      fillalpha=0.5,
                                      label=labels[i],
                                      normalize=true,
                                      ylims=(0, 0.02),
                                      xlims=(0, 2000))
        end
    end

    return plt
end

function animate_power_histogram(filename, farm, turbine)
    animation = @animate for ws in 0:0.1:20
        power_histogram(farm, turbine, ws)
    end

    gif(animation, filename)
end

function windspeed_widget(farm)
    plt = @manipulate for ws in 0:0.1:20, h in 0:5:360, turbine in 1:1:15 
        power_histogram(farm, turbine, ws, h)
    end

    w = Window()
    body!(w, plt)
end

infer(estimator_type, turbine_id) = infer(_ -> true, estimator_type, turbine_id)

load_farm!("../../Data/", "ARD")

