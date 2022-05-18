module Chain

using Statistics
using LinearAlgebra
using Random

function dataset(n::Int, p::Int)
    trainset = randn(n, p)
    trainlabels = rand((-1,1), p)
    return trainset, trainlabels
end

function teacher_dataset(n::Int, p::Int, teacher::Vector)
    trainset = randn(n, p)
    mult_mat = teacher.*trainset
    trainlabels = [sign(sum(mult_mat[:,i])) for i=1:p]
    return trainset, trainlabels
end

function energy(w::Vector, K::Int, trainset::Matrix, trainlabels::Vector)
    n = length(w)
    p = length(trainlabels)
    @assert n % K == 0
    @assert size(trainset) == (n, p)
    nh = n ÷ K # inputs per hidden unit

    W = reshape(w, nh, K)
    X = reshape(trainset, nh, K, p)
    # indices = Int[]
    errors = 0
    @inbounds for μ = 1:p
        Δ = 0
        for j = 1:K
            Δj = 0.0
            for i = 1:nh
                Δj += W[i, j] * X[i, j, μ]
            end
            Δ += sign(Δj)
        end
        outμ = sign(Δ)
        if outμ ≠ trainlabels[μ]
            # push!(indices, μ)
            errors += 1
        end
    end

    ## vectorized broadcasted version: slower than the explicit for loops, and allocates
    ## TODO? try Tullio.jl

    # errors2 = sum(vec(sign.(sum(sign.(sum(W .* X, dims=1)), dims=2))) .≠ trainlabels)
    # @assert errors == errors2

    # println("error indices: $indices")
    return errors
end


function gen_candidate(
        c::Vector,
        d::Float64,           # pairwise points distance
        trainset::Matrix,
        trainlabels::Vector,
        K::Int,
    )
    n = length(c)

    ## distance from barycenter (expected)
    ρ = d * norm(c) / √(1 - d^2 / 2)

    ## generate a new random direction
    x = ρ / √n * randn(n)

    ## choose the best of the two directions
    p_new1 = c + x
    p_new2 = c - x
    E_new1 = energy(p_new1, K, trainset, trainlabels)
    E_new2 = energy(p_new2, K, trainset, trainlabels)
    ## new point
    p_new, E_new = E_new1 ≤ E_new2 ? (p_new1, E_new1) : (p_new2, E_new2)

    return p_new, E_new
end

function simplex_chain(
        n::Int,
        p::Int,
        y::Int;
        K::Int = 3,
        d₀::Float64 = 1.0,
        seed::Int = 411068089483816338,
        iters::Int = 1_000,
        rescale_factor::Float64 = 0.99,
        max_failures::Integer = 5,
        verbose::Bool = false,
        teacher::Bool = false,
        p_test::Int = 1000,
    )

    @assert 0 ≤ d₀ ≤ √2

    d = d₀

    if verbose
        dirname = "runs" * (teacher ? "_teacher" : "")
        mkpath(dirname)
        filename = joinpath(dirname, "run_n_$(n)_p_$(p)_y_$(y)_k_$(K)_d_$(d)_df_$(rescale_factor).txt")
        open(filename, "a") do io
            println(io, "#it, d, Ec, Eworst, Ebest, Ec" * (teacher ? " test" : ""))
        end
    end

    seed > 0 && Random.seed!(seed)

    if teacher
        W_teacher = randn(n)
        trainset, trainlabels = teacher_dataset(n, p, W_teacher)
        testset, testlabels = teacher_dataset(n, p_test, W_teacher)
    else
        trainset, trainlabels = dataset(n, p)
        testset, testlabels = zeros(n,0), zeros(0)
    end

    c = √((1 - d^2 / 2) / n) * randn(n)

    Ec = energy(c, K, trainset, trainlabels)
    testEc = teacher ? energy(c, K, testset, testlabels) : 0.0

    Es = [typemax(Int) for j = 1:y]
    norms = ones(y)

    E_best, E_worst = extrema(Es)

    @info "it = 0 d = $d Ec = $(Ec) Es = $(mean(Es)) ± $(std(Es)) [$(extrema(Es))]" * (teacher ? " $testEc" : "")
    println("norms: $(mean(norms)) ± $(std(norms))")

    ind = 1

    it = 0
    failures = 0
    while all(Es .> 0) && Ec > 0
        it += 1
        E_worst = maximum(Es)
        verbose && open(filename, "a") do io
            println(io, "$it $d $Ec $E_worst $(minimum(Es))" * (teacher ? " $testEc" : ""))
        end
        failed = true
        for attempt = 1:iters
            p_new, E_new = gen_candidate(c, d, trainset, trainlabels, K)
            if E_new < E_worst
                Es[ind] = E_new
                E_worst = maximum(Es)
                norms[ind] = norm(p_new)
                c = (y - 1) / y * c + 1 / y * p_new
                Ec = energy(c, K, trainset, trainlabels)
                testEc = teacher ? energy(c, K, testset, testlabels) : 0.0
                failed = false
                ind = mod1(ind + 1, y)
                if Ec == 0
                    if verbose
                        open(filename, "a") do io
                            println(io, "$it $d $Ec $E_worst $(minimum(Es))" * teacher ? " $testEc" : "")
                        end
                    end
                    break
                end
            end
        end

        if failed
            failures += 1
        else
            failures = 0
        end

        scale = norm(c) / √(1 - d^2 / 2)
        c ./= scale

        @info "it = $it [$failures] d = $d Ec = $(Ec) Es = $(mean(Es)) ± $(std(Es)) [$(extrema(Es))]" * (teacher ? " testEc = $testEc" : "")
        println("norms: $(mean(norms)) ± $(std(norms))")

        d *= rescale_factor

        failures < max_failures && continue

        if failures ≥ max_failures
            @info "failed $failures times, give up"
            break
        end

        if d < 1e-4
            @info "min dist reached, give up"
            break
        end
    end

    return c, Ec, Es
end

end # module Chain
