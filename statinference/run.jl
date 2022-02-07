using Distributions
using Plots

include("data.jl")
include("stats.jl")

farm = Nothing

function load_farm!(datapath, code)
    global farm 
    farm = FarmData(datapath, code)
end

function infer(condition, estimator_type, turbine_id)
    ps = windspeeds(condition, farm, turbine_id; normal=true)
    ps .+= 0.01 - minimum(ps) 

    mle = fit(estimator_type, ps)

    histogram(ps, normalize=true)
    xs = 0:0.1:20
    plot!(xs, pdf.(mle, xs))
end

infer(estimator_type, turbine_id) = infer(_ -> true, estimator_type, turbine_id)

load_farm!("Data/", "ARD")

