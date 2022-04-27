module polytope_MC

#export simplex_opt

using Statistics
using LinearAlgebra
using Random





function new_point_MC(old_points::Matrix, rho::Float64; precision::Float64 = 0.01)
    
    delta= 0.01
    
    n, npoints = size(old_points)
    old_dist = [99. for i=1:npoints]
    new_point = mean(old_points, dims=2)
    normalize!(new_point)
    @assert any(old_dist.> precision)

    while any(old_dist.> precision)
        #println("ciao")
        xtemp = deepcopy(new_point)
        xtemp = new_point + (delta/n)*randn(n)
        normalize!(xtemp)
        new_dist = [abs(dot(xtemp, old_points[:,i]) - rho) for i=1:npoints]
        if any(new_dist.<= precision)
            bitvec = new_dist .<= precision
            for k=1:npoints
                !(bitvec[k]) && (new_dist[k]<old_dist[k]) && (bitvec[k] = 1)
            end
            if all(bitvec) 
                new_point = deepcopy(xtemp)
                old_dist = new_dist
            end
        elseif all(new_dist .<= old_dist )
            new_point = deepcopy(xtemp)
            old_dist = new_dist 
        end
    end #while
    return new_point
end

function make_polytope(n::Int, y::Int, d::Float64)

    ##d is the overlap now 

    #ps = normalize!(randn(n,1))
    ps = reshape(normalize!(randn(n)), (n,1))
    ## Add the remaining y-1 points
    for i = 2:y
        ps = hcat(ps, new_point_MC(ps, d))
        println(i)
       
    end
    #@assert all([dot(ps[:,i], ps[:,j]) ≈ (i==j ? 0.0 : d) for i = 1:y, j = 1:y])
    return ps
end
    


end # module
