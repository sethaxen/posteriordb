module EightSchoolsNoncentered

using LinearAlgebra, Turing

@model function model(
    data;
    J=data["J"],  # number of schools
    y=data["y"],  # estimated treatment
    sigma=data["sigma"]  # std of estimated effect
)
    theta_trans ~ filldist(Normal(0, 1), J)  # transformation of theta
    mu ~ Normal(0, 5)  # hyper-parameter of mean, non-informative prior
    tau ~ Cauchy(0, 5)  # hyper-parameter of sd
    theta = theta_trans .* tau .+ mu  # original theta
    y .~ Normal.(theta, sigma)
    return (; theta=theta)
end

end # module
