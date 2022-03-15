module SimplexSearch

using Statistics
using LinearAlgebra
using Random

function dataset(n::Int, p::Int)
    trainset = randn(n, p)
    trainlabels = [rand(1:2) > 1 ? -1 : 1 for i in 1:p]
    return trainset, trainlabels
end

## TODO: implement a real energy function, this is a silly placeholder
function energy(p::Vector, hL_size, trainset, trainlabels)
    fakep = reshape(p, (Int(length(p)/hL_size), hL_size))
    errors = 0
    for i in 1:length(trainlabels)
        out = sign(sum([sum(trainset[:,i].*fakep[:,j]) for j in 1:hL_size]))  
        if out != trainlabels[i]
            errors += 1
        end
    end
    return errors/length(trainlabels)
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


function replace_point!(
        ps::Matrix,   # input points (in a matrix, each column is a point)
        vsum::Vector, # sum of all the points
        Es::Vector,   # energy of each point
        i_worst::Int, # index of the point to substitute
        d::Float64,    # pairwise points distance
        trainset,
        trainlabels,
        hL_size::Int = 3,
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
    @assert all(norm(b[:,i]) ≈ 1 for i = 1:(y-2)) # normalization
    @assert maximum(abs.([b[:,i] ⋅ b[:,j] for i = 1:(y-2), j = 1:(y-2)] - I)) < 1e-12 # orthogonality

    ## generate a random direction
    x = randn(n)

    ## subtract the components from the
    ## orthonormal basis
    for i = 1:(y-2)
        dir = b[:,i]
        x .-= (x ⋅ dir) * dir
    end
    @assert maximum(abs.(x ⋅ b[:,i] for i = 1:(y-2))) < 1e-12

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

    Es[i_worst] = energy(new_p, hL_size, trainset, trainlabels)

    return
end


function simplex_opt(n::Int, 
    p::Int, y::Int, d::Float64, 
    seed::Int = -1, hL_size::Int = 3;
    rescale::Int = 100, rescale_factor::Float64 = 0.5) ## TODO: add arguments etc.


    filename = string("run_n_", n, "_p_", p, "_d_", d, "_y_", y, "_dscale_",rescale_factor, ".txt")
    trainset, trainlabels = dataset(n,p)
    #n here is the input size, for a tree committee with hL_size = 3 we need 3*n parameters
    n = Int32(3*n)  


    seed > 0 && Random.seed!(seed)
    #n = 10
    #y = 5
    #d = 3.0

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
    Es = [energy(ps[:,i], hL_size, trainset, trainlabels) for i = 1:y]
    center_energy = energy(vsum/y,hL_size, trainset, trainlabels )
  


    @info "Center energy: $center_energy Es before: $Es"

    counter = 0
    
    ## One step of the algorithm (TODO: put this in an optimization loop)
    while !any([x == 0 for x in Es])
        counter += 1
        

        E_worst, i_worst = findmax(Es)
        E_best, i_best = findmax(Es)
        @info "replacing $i_worst"
        ## Find a new point with lower energy
        E_new = E_worst
        stop_count = 0
        while E_new ≥ E_worst
            stop_count += 1
            stop_count == rescale && break
            replace_point!(ps, vsum, Es, i_worst, d, trainset, trainlabels)
            
            ## temporary consistency check
            #@show [norm(ps[:,i] - ps[:,j]) for i = 1:y, j = 1:y]
            @assert all([norm(ps[:,i] - ps[:,j]) ≈ (i==j ? 0.0 : d) for i = 1:y, j = 1:y])
            E_new = Es[i_worst]
        end
        #stop_count >= 10*n && (println("solution not found") && break)

        ## Rescale simplex 

        if stop_count % rescale == 0
            c = vsum/y
            d *= rescale_factor
            println("simplex size:", d)
            for j=1:y
                #ps[:,j] = normalize!(ps[:,j])
                #ps[:,j] .= c .+ rescale_factor.*(ps[:,j] .- c) ##shrink towards the center
                ps[:,j] .= ps[:,i_best] .+ rescale_factor.*(ps[:,j] .- ps[:,i_best]) ##shrink towards the center
            end
            vsum = vec(sum(ps, dims=2))
            Es = [energy(ps[:,i], hL_size, trainset, trainlabels) for i = 1:y]
            center_energy = energy(vsum/y,hL_size, trainset, trainlabels )
            #@show [norm(ps[:,i] - ps[:,j]) for i = 1:y, j = 1:y]
            @assert all([norm(ps[:,i] - ps[:,j]) ≈ (i==j ? 0.0 : d) for i = 1:y, j = 1:y])
        end

        io = open(filename, "a")
        println(io, counter, " ", mean(Es))
        close(io)

        @info "Center energy: $center_energy Es after: $Es"
    end
    
    ## TODO ...
    return Es
end





end # module
