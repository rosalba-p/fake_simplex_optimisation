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
    indices = Int[]
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
            push!(indices, μ)
        end
        errors += (outμ ≠ trainlabels[μ])   
    end

    ## vectorized broadcasted version: slower than the explicit for loops, and allocates
    ## TODO? try Tullio.jl

    # errors2 = sum(vec(sign.(sum(sign.(sum(W .* X, dims=1)), dims=2))) .≠ trainlabels)
    # @assert errors == errors2

    #println("gli indici sono: ", indices)
    return errors / p
end

function new_point(ps::Matrix, d::Float64)
    n, y = size(ps)
    @assert y ≤ n

    ## barycenter
    c = mean(ps, dims=2)

    ## create a new orthonormal basis
    ## that spans the y input points
    ## (only y-1 points are needed)
    b = zeros(n, y-1)
    for i = 1:(y-1)
        di = reduce(vcat, ps[:,i] - c)
        if !all(di .== 0)
            normalize!(di)
        end
        for j = 1:(i-1)
            dj = b[:,j]
            di .-= (di ⋅ dj) * dj
        end
        normalize!(di)
        b[:,i] = di
    end

    ## generate a random direction
    x = randn(n)

    ## subtract the components from the
    ## orthonormal basis
    for i = 1:(y-1)
        dir = b[:,i]
        x .-= (x ⋅ dir) * dir
    end

    ## rescale x so that its length corresponds
    ## to the height of a regular simplex with
    ## y+1 points and size d
    normalize!(x)
    x .*= d * √((y + 1) / 2y)

    ## new point
    new_p = c + x

    return new_p
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
vcent = vsum
center = vcent / y

## barycenter, excluding the point to be removed
vcav = vsum - old_p
c = vcav / (y - 1)

## generate direction from center to worst point
x = center - old_p 

## rescale x so that its length corresponds
## to the height of a regular simplex with
## y points and size d
normalize!(x)
x .*= d * √(y / (2*(y-1)))

## new point
new_p = center + x

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
    @assert y+1 ≤ n
    @assert y ≥ 2
    @assert 1 ≤ i_worst ≤ y

    ## barycenter, excluding the point to be removed
    vcav = vsum - ps[:,i_worst]
    c = vcav / (y - 1)

    ## create a new orthonormal basis
    ## that spans the y-1 remaining input points
    ## (only y-2 points are needed; we loop over
    ## all y points but we skip the i_worst one and
    ## we also just get out after y-2 points)
    b = zeros(n, y-2)
    k = 0
    for i = 1:y
        i == i_worst && continue
        k += 1
        di = ps[:,i] - c
        if !all(di .== 0)
            normalize!(di)
        end
        for j = 1:(k-1)
            dj = b[:,j]
            di .-= (di ⋅ dj) * dj
        end
        normalize!(di)
        b[:,k] = di
        k == y-2 && break
    end
    ## debug checks
    # @assert all(norm(b[:,i]) ≈ 1 for i = 1:(y-2)) # normalization
    # @assert maximum(abs.([b[:,i] ⋅ b[:,j] for i = 1:(y-2), j = 1:(y-2)] - I)) < 1e-12 # orthogonality

    ## generate a random direction
    x = randn(n)

    ## subtract the components from the
    ## orthonormal basis
    for i = 1:(y-2)
        dir = b[:,i]
        x .-= (x ⋅ dir) * dir
    end
    # @assert maximum(abs.(x ⋅ b[:,i] for i = 1:(y-2))) < 1e-12

    ## old point
    old_p = ps[:,i_worst]

    ## try to go to the opposite direction than the previous point
    if x ⋅ (old_p - c) > 0
        x = -x
    end

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

    original_distance = d

    filename = "run.n_$n.p_$p.y_$y.d_$d.df_$rescale_factor.txt"

    trainset, trainlabels = dataset(n, p)
    # n here is the input size, for a tree committee with hL_size = 3 we need 3*n parameters

    ## Create the initial y points
    ## Generate the first point at random
    ps = reshape(randn(n), (n,1))

    ## Add the remaining y-1 points
    for i = 2:y
        ps = hcat(ps, new_point(ps, d))
    end

    ## Check that the distances are all d off-diagonal here
    @assert all([norm(ps[:,i] - ps[:,j]) ≈ (i==j ? 0.0 : d) for i = 1:y, j = 1:y])
    

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
            replace_point_lin!(ps, vsum, Es, i_worst, d, trainset, trainlabels, K)
            ## temporary consistency check
            @assert all([norm(ps[:,i] - ps[:,j]) ≈ (i==j ? 0.0 : d) for i = 1:y, j = 1:y])
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
        println("norm of replicas:", mean([norm(ps[:,i]) for i=1:y]))
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
        @assert all([norm(ps[:,i] - ps[:,j]) ≈ (i==j ? 0.0 : d) for i = 1:y, j = 1:y])

        # open(filename, "a") do io
        #     println(io, "it = $it d = $d Ec = $Ec Es = $Es")
        # end

    end

    return ps, Ec, Es
end





end # module
