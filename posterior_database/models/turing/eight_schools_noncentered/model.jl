module EightSchoolsNoncentered

using FillArrays
using LinearAlgebra
using Turing

struct EightSchoolsNoncenteredModel{T<:AbstractVector{<:Real}}
    """number of schools"""
    J::Int
    """variance of estimated effect"""
    sigma2::T
end

@model function (model::EightSchoolsNoncenteredModel)()
    theta_trans ~ MvNormal(Zeros(model.J), I)  # transformation of theta
    mu ~ Normal(0, 5)  # hyper-parameter of mean, non-informative prior
    tau ~ Cauchy(0, 5)  # hyper-parameter of sd
    theta = theta_trans .* tau .+ mu  # original theta
    y ~ MvNormal(theta, Diagonal(model.sigma2))  # estimated treatment
    return (; theta=theta)
end

function model(data::AbstractDict{String})
    J = data["J"]
    sigma2 = data["sigma"].^2
    y = data["y"]
    return EightSchoolsModel(J, sigma2) | (; y=y)
end

end # module
