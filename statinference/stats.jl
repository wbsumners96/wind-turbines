using Distributions
using Roots
using Statistics

function estimator(::Type{LogNormal}, xs) 
    μ̂(xs) = mean(log.(xs))
    σ̂(xs) = mean((log.(xs) .- μ̂(xs)) .^ 2)

    return LogNormal(μ̂(xs), σ̂(xs))
end

function estimator(::Type{Weibull}, xs) 
    f(α) = sum(@. xs^α * log(xs))/sum(xs .^ α) - 1/α - mean(log.(xs))
    α̂ = find_zero(f, 1)
    θ̂(α) = mean(xs .^ α)^(1/α)

    return Weibull(α̂, θ̂(α̂))
end

function power_pdf(wb, k, p)
    Z = 1/(3*cbrt(k*p^2))
    f(w) = pdf(wb, w)

    return Z*f(cbrt(p/k))
end

