module SimplexSearch

using Statistics
using LinearAlgebra
using Random
using StatsBase

function dataset(n::Int, p::Int)
    trainset = randn(n, p)
    trainlabels = rand((-1,1), p)
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

function gen_candidate_reflected(
        ps::Matrix,           # input points (in a matrix, each column is a point)
        vsum::Vector,         # sum of all the points
        Es::Vector,           # energy of each point
        i_worst::Int,         # index of the point to substitute
        d::Float64,           # pairwise points distance
        trainset::Matrix,
        trainlabels::Vector,
        K::Int = 3,
    )
    n, y = size(ps)
    @assert y ≥ 2
    @assert 1 ≤ i_worst ≤ y

    ## old point
    old_p = ps[:,i_worst]

    ## barycenter, including the point to be removed

    center = vsum / y

    ## barycenter, excluding the point to be removed
    vcav = vsum - old_p
    c = vcav / (y - 1)

    ## new point
    new_p = 2c - old_p
    new_E = energy(new_p, K, trainset, trainlabels)

    return new_p, new_E
end


function gen_candidate(
        ps::Matrix,           # input points (in a matrix, each column is a point)
        vsum::Vector,         # sum of all the points
        ws::Vector,
        wz::Float64,
        i_sub::Int,         # index of the point to substitute
        d::Float64,           # pairwise points distance
        trainset::Matrix,
        trainlabels::Vector,
        K::Int = 3,
    )
    n, y = size(ps)
    @assert y ≥ 2
    @assert 1 ≤ i_sub ≤ y

    p_old = ps[:,i_sub]
    w_old = ws[i_sub]

    ## barycenter, excluding the point to be removed
    vcav = vsum - (p_old .* w_old)
    wzcav = wz - w_old
    c = vcav / wzcav

    # scale = norm(c) / (d / √2)

    ## distance from barycenter (expected)
    ρ = d * norm(c) * √(y / (y - 1))

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


function simplex_opt(
        n::Int,
        p::Int,
        y::Int;
        K::Int = 3,
        d::Float64 = Float64(n),
        seed::Int = 411068089483816338,
        max_attempts::Int = 100,
        rescale_factor::Float64 = 0.99,
        β₀::Float64 = 1.0,
        Δβ::Float64 = 0.1,
        α::Float64 = 1.0
    )

    seed > 0 && Random.seed!(seed)

    filename = "run.n_$n.p_$p.y_$y.d_$d.df_$rescale_factor.txt"

    trainset, trainlabels = dataset(n, p)
    # n here is the input size, for a tree committee with hL_size = 3 we need 3*n parameters

    ## Create the initial y points
    ## the factors are such that the points all have norm d and
    ## distance d between them (in expectation)
    c0 = 1 / √(2n) * randn(n)
    ps = c0 .+ (d / √(2n)) .* randn(n, y)

    ## Print norms and distances, as a check
    norms = [norm(ps[:,i]) for i=1:y]
    println("norm of replicas: $(mean(norms)) ± $(std(norms))")
    dists = [norm(ps[:,i] - ps[:,j]) for i = 1:y for j = (i+1):y]
    println("dists of replicas: $(mean(dists)) ± $(std(dists))")

    β = β₀

    ## Energies
    Es = [energy(ps[:,i], K, trainset, trainlabels) for i = 1:y]
    E_best, E_worst = extrema(Es)

    ws = [exp(-β * (E-E_worst)) for E in Es]
    wz = sum(ws.^α)

    ## Pre-compute the weighted sum of all points
    vsum = vec(sum(ps .* ((ws').^α), dims=2))

    ## Barycenter
    c = vsum / wz

    Ec = energy(c, K, trainset, trainlabels)
    @info "it = 0 d = $d β = $β Ec = $(Ec/p) Emin = $(E_best/p) Es = $(Es/p)"

    it = 0
    while all(Es .> 0) && Ec > 0
        it += 1
        E_worst = maximum(Es)
        # i_sub = rand(findall(Es .== E_worst))
        ## Find a new point with lower energy
        failed = true
        for attempt = 1:max_attempts
            i_sub = sample(1:n, Weights(1 ./ ws))
            E_sub = Es[i_sub]
            # @show ws / sum(ws)
            # @show E_sub, E_worst, Es
            # @assert E_sub == E_worst
            # if attempt == max_attempts
            #     p_new, E_new = gen_candidate_reflected(ps, vsum, Es, i_sub, d, trainset, trainlabels, K)
            # else
                p_new, E_new = gen_candidate(ps, vsum, ws, wz, i_sub, d, trainset, trainlabels, K)
            # end
            if E_new < E_worst # || rand() < exp(-β * p * (E_new - E_sub))

                # p_old, E_old = ps[:,i_sub], E_sub
                Es[i_sub] = E_new
                E_worst = maximum(Es)

                ps[:, i_sub] = p_new
                ws = [exp(-β * (E - E_worst)) for E in Es]
                wz = sum(ws.^α)
                vsum = vec(sum(ps .* ((ws').^α), dims=2))
                c = vsum / wz
                Ec = energy(c, K, trainset, trainlabels)

                # w_new, w_old = exp(-β * (E_new - E_worst)), ws[i_sub]
                # vsum .+= (p_new .* w_new.^α) .- (p_old .* w_old.^α)
                # wz += w_new.^α - w_old.^α
                # ps[:, i_sub] = p_new
                failed = false
                # break
                # i_sub = rand(findall(Es .== E_worst))
            end
        end

        # if it % 100 == 0
            β += Δβ
            ws = [exp(-β * (E - E_worst)) for E in Es]
            wz = sum(ws.^α)
            vsum = vec(sum(ps .* ((ws').^α), dims=2))
            c = vsum / wz
            Ec = energy(c, K, trainset, trainlabels)
        # end

        E_best = minimum(Es)
        @info "it = $it d = $d β = $β Ec = $(Ec/p) Emin = $(E_best/p) Es = $(Es/p)"
        norms = [norm(ps[:,i]) for i=1:y]
        println("norm of replicas: $(mean(norms)) ± $(std(norms))")
        dists = [norm(ps[:,i] - ps[:,j]) for i = 1:y for j = (i+1):y]
        println("dists of replicas: $(mean(dists)) ± $(std(dists))")

        d *= rescale_factor

        failed || continue

        # if d ≤ 1e-4
        #     @info "failed, giving up"
        #     break
        # end

        # @info "give up"
        # break

        if it ≥ 1_000
            @info "failed, giving up"
            break
        end

        ## Random attempts failed: rescale simplex

        # @info "resampling failed, rescale simplex $d -> $(d * rescale_factor)"

        # d *= rescale_factor
        # ref = Ec ≤ E_best ? c : ps[:,i_best]
        # for j = 1:y
        #     ps[:,j] .= ref .+ rescale_factor .* (ps[:,j] .- ref)
        # end
        @info "resampling failed, reshuffle"
        c0 = c / (√2 * norm(c))
        ps = c0 .+ (d / √(2n)) .* randn(n, y)

        norms = [norm(ps[:,i]) for i=1:y]
        println("norm of replicas: $(mean(norms)) ± $(std(norms))")
        dists = [norm(ps[:,i] - ps[:,j]) for i = 1:y for j = (i+1):y]
        println("dists of replicas: $(mean(dists)) ± $(std(dists))")

        Es = [energy(ps[:,i], K, trainset, trainlabels) for i = 1:y]

        E_best, E_worst = extrema(Es)

        ws = [exp(-β * (E-E_worst)) for E in Es]
        wz = sum(ws.^α)
        vsum = vec(sum(ps .* ((ws').^α), dims=2))
        c = vsum / wz
        Ec = energy(c, K, trainset, trainlabels)

        # open(filename, "a") do io
        #     println(io, "it = $it d = $d Ec = $Ec Es = $Es")
        # end

    end

    return ps, Ec, Es
end





end # module
