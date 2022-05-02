module SimplexSearch

using Statistics
using LinearAlgebra
using Random

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
    return errors / p
end

function replace_point_lin!(
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
    @assert y+1 ≤ n
    @assert y ≥ 2
    @assert 1 ≤ i_worst ≤ y


    ## old point
    old_p = ps[:,i_worst]

    ## barycenter, including the point to be removed

    center = vsum / y

    ## barycenter, excluding the point to be removed
    vcav = vsum - old_p
    c = vcav / (y - 1)

    ## generate direction from center to worst point
    x = c - old_p 

    ## rescale x so that its length corresponds
    ## to the height of a regular simplex with
    ## y points and size d
    normalize!(x)
    x .*= d * √(y / (2*(y-1)))

    ## new point
    new_p = c + x

    ## update the input structures
    ps[:,i_worst] = new_p
    vsum .= vcav .+ new_p

    Es[i_worst] = energy(new_p, K, trainset, trainlabels)

    return
end


function replace_point!(
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

    ## barycenter, excluding the point to be removed
    vcav = vsum - ps[:,i_worst]
    c = vcav / (y - 1)

    ## distance from barycenter (expected)
    ρ = d * √(y / (2 * (y - 1)))

    ## generate a new random direction
    x = ρ / √n * randn(n)

    old_p = ps[:,i_worst]

    ## choose the best of the two directions
    new_p1 = c + x
    new_p2 = c - x
    new_E1 = energy(new_p1, K, trainset, trainlabels)
    new_E2 = energy(new_p2, K, trainset, trainlabels)
    ## new point
    new_p = new_E1 ≤ new_E2 ? new_p1 : new_p2

    ## update the input structures
    ps[:,i_worst] = new_p
    vsum .= vcav .+ new_p

    Es[i_worst] = energy(new_p, K, trainset, trainlabels)

    return
end


function simplex_opt(
        n::Int,
        p::Int,
        y::Int;
        K::Int = 3,
        d::Float64 = Float64(n),
        seed::Int = 411068089483816338,
        max_attempts::Int = 100,
        rescale_factor::Float64 = 0.5
    )

    seed > 0 && Random.seed!(seed)

    filename = "run.n_$n.p_$p.y_$y.d_$d.df_$rescale_factor.txt"

    trainset, trainlabels = dataset(n, p)
    # n here is the input size, for a tree committee with hL_size = 3 we need 3*n parameters

    ## Create the initial y points
    c0 = d / √(2n) * randn(n)
    ps = hcat((c0 + d / √(2n) * randn(n) for i = 1:y)...)

    ## Check
    # println([norm(ps[:,i] - ps[:,j]) for i = 1:y, j = 1:y])
    # println([norm(ps[:,i]) for i = 1:y])

    ## Pre-compute the sum of all points
    vsum = vec(sum(ps, dims=2))

    ## Energies
    Es = [energy(ps[:,i], K, trainset, trainlabels) for i = 1:y]
    Ec = energy(vsum / y, K, trainset, trainlabels)
    E_best = minimum(Es)

    @info "it = 0 Ec = $Ec Emin = $E_best Es = $Es"

    it = 0
    while all(Es .> 0) && Ec > 0
        it += 1
        E_worst, i_worst = findmax(Es)
        @info "replacing $i_worst"
        ## Find a new point with lower energy
        E_new = E_worst
        p_bk, E_bk = ps[:,i_worst], Es[i_worst]
        for attempt = 1:max_attempts
            replace_point!(ps, vsum, Es, i_worst, d, trainset, trainlabels, K)
            # replace_point_lin!(ps, vsum, Es, i_worst, d, trainset, trainlabels, K)
            ## temporary consistency check
            # dists = [norm(ps[:,i] - ps[:,j]) for i = 1:y for j = (i+1):y]
            # println("dists: $(mean(dists)) $(std(dists))")
            E_new = Es[i_worst]
            E_new < E_worst && break
        end

        success = E_new < E_worst
        if !success
            ## restore the previous worst point
            ps[:,i_worst] = p_bk
            Es[i_worst] = E_bk
            vsum = vec(sum(ps, dims=2))
        end

        c = vsum / y
        Ec = energy(c, K, trainset, trainlabels)
        E_best, i_best = findmin(Es)
        @info "it = $it d = $d Ec = $Ec Emin = $E_best Es = $Es"
        norms = [norm(ps[:,i]) for i=1:y]
        println("norm of replicas: $(mean(norms))±$(std(norms))")
        success && continue

        if d ≤ 1e-4
            @info "failed, giving up"
            break
        end

        ## Random attempts failed: rescale simplex

        @info "resampling failed, rescale simplex $d -> $(d * rescale_factor)"

        d *= rescale_factor
        ref = Ec ≤ E_best ? c : ps[:,i_best]
        for j = 1:y
            ps[:,j] .= ref .+ rescale_factor .* (ps[:,j] .- ref)
        end
        vsum = vec(sum(ps, dims=2))
        Es = [energy(ps[:,i], K, trainset, trainlabels) for i = 1:y]
        Ec = energy(vsum / y, K, trainset, trainlabels)

        # open(filename, "a") do io
        #     println(io, "it = $it d = $d Ec = $Ec Es = $Es")
        # end

    end

    return ps, Ec, Es
end





end # module
