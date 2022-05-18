module Chain

using Statistics
using LinearAlgebra
using Random

function dataset(n::Int, p::Int)
    trainset = randn(n, p)
    trainlabels = rand((-1,1), p)
    return trainset, trainlabels
end

function teacher_dataset(n::Int, p::Int, K::Int, teacher::Vector)
    trainset = randn(n, p)
    #mult_mat = teacher.*trainset
    #trainlabels = [sign(sum(mult_mat[:,i])) for i=1:p]
    #return trainset, trainlabels

    @assert(n == length(teacher))
    @assert n % K == 0

    nh = n ÷ K # inputs per hidden unit

    W = reshape(teacher, nh, K)
    X = reshape(trainset, nh, K, p)
    trainlabels = []


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
        append!(trainlabels, outμ)

    end
    return trainset, trainlabels#, pre_activations
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
    #pre_activations = []
    smallest_stability = -1000
    smallest_stability_index = 1
    @inbounds for μ = 1:p
        Δ = 0
        for j = 1:K
            Δj = 0.0
            for i = 1:nh
                Δj += W[i, j] * X[i, j, μ]
            end
            #append!(pre_activations, Δj)
            (Δj*trainlabels[μ] < 0) && (Δj*trainlabels[μ] > smallest_stability) && (smallest_stability = Δj*trainlabels[μ])
            smallest_stability_index = j 
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
    return errors, smallest_stability, smallest_stability_index #, pre_activations
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
    E_new1, _, _ = energy(p_new1, K, trainset, trainlabels)
    E_new2, _, _ = energy(p_new2, K, trainset, trainlabels)
    ## new point
    p_new, E_new = E_new1 ≤ E_new2 ? (p_new1, E_new1) : (p_new2, E_new2)

    return p_new, E_new
end


function LaL(
        n::Int,
        p::Int;
        K::Int = 3,
        seed::Int = 411068089483816338,
        iters::Int = 1_000,
        verbose::Bool = true,
        p_test::Int = 1000, 
        eta::Float64 = 0.1,
        c::Vector, 
        teacher::Vector,
        trainset, 
        trainlabels
    )
    
    if verbose
        filename = "runs_LaL/run_n_$(n)_p_$(p)_k_$(K).txt"
        open(filename, "a") do io
        println(io, "#it, Ec")
        end
    end

    seed > 0 && Random.seed!(seed)

    #teacher = randn(n)
    #trainset, trainlabels = teacher_dataset(n, p, K, teacher)
    #testset, testlabels = teacher_dataset(n, p_test, K, teacher)
    
    
    
    nh = n ÷ K # inputs per hidden unit

    #c = randn(n) #random initialisation


    #teacher_activations = []
    #for μ = 1:p
    #    data = reshape(trainset[:,μ], n, 1)
    #    Eteacher, teacher_activation = energy(teacher, K, data, [trainlabels[μ]])
    #    @assert(Eteacher == 0)
    #    append!(teacher_activations, [teacher_activation])
    #end

    #zeroth epoch
    Ec, smallest_stability, idx  = energy(c, K, trainset, trainlabels)
    @info "it = 0 Ec = $Ec"
    if verbose 
        open(filename, "a") do io
            println(io, "0 $Ec")
        end
    end

    for epoch=1:iters
        
        errors = 0

        for μ = 1:p
            pattern = reshape(trainset[:,μ], n, 1)
            label = trainlabels[μ]
            #teacher_act = teacher_activations[μ]

            

            Ec, smallest_stability, idx  = energy(c, K, pattern, [label])
            Ec == 0 && continue 
            if Ec == 1
                errors +=1 
                c = reshape(c, nh, K)
                pattern = reshape(pattern, nh,K)
                #min, idx = findmin(broadcast(abs, teacher_act - pre_act))
                for i = 1:nh
                    c[i, idx] = c[i, idx] + 2*eta*label*pattern[i, idx]
                end
                c = reshape(c, n)
                pattern = reshape(pattern,n)
            end
        
        end

        @info "it = $epoch Ec = $(errors)"
        if verbose 
            open(filename, "a") do io
                println(io, "$epoch $errors")
            end
        end
        errors ==0 && break
    end

end 



function simplex_chain(
        n::Int,
        p::Int,
        y::Int;
        K::Int = 3,
        K_teacher::Int = 1,
        d₀::Float64 = 1.0,
        seed::Int = 411068089483816338,
        iters::Int = 1_000,
        rescale_factor::Float64 = 0.99,
        max_failures::Integer = 5,
        verbose::Bool = true,
        p_test::Int = 1000
    )

    #@assert 0 ≤ d₀ ≤ √2

    d = d₀

    if verbose
        filename = "runs_teacher/run_n_$(n)_p_$(p)_y_$(y)_k_$(K)_d_$(d)_df_$(rescale_factor).txt"
        open(filename, "a") do io
        println(io, "#it, d, Ec, Eworst, Ebest, Ec test")
        end
    end

    seed > 0 && Random.seed!(seed)


    teacher = randn(n)
    trainset, trainlabels = teacher_dataset(n, p, K_teacher, teacher)
    testset, testlabels = teacher_dataset(n, p_test, K_teacher, teacher)

    
    c = √((1 - d^2 / 2) / n) * randn(n)

    Ec, _, _  = energy(c, K, trainset, trainlabels)
    testEc, _, _ = energy(c, K, testset, testlabels)

    Es = [typemax(Int) for j = 1:y]
    norms = ones(y)

    E_best, E_worst = extrema(Es)

    @info "it = 0 d = $d Ec = $(Ec) Es = $(mean(Es)) ± $(std(Es)) [$(extrema(Es))]"
    println("norms: $(mean(norms)) ± $(std(norms))")

    ind = 1

    it = 0
    failures = 0
    while all(Es .> 0) && Ec > 0
        it += 1
        E_worst = maximum(Es)
        open(filename, "a") do io
        println(io, "$it $d $Ec $E_worst $(minimum(Es)) $testEc")
        end
        failed = true
        for attempt = 1:iters
            p_new, E_new = gen_candidate(c, d, trainset, trainlabels, K)
            if E_new < E_worst
                Es[ind] = E_new
                E_worst = maximum(Es)
                norms[ind] = norm(p_new)
                c = (y - 1) / y * c + 1 / y * p_new
                Ec, _, _ = energy(c, K, trainset, trainlabels)
                testEc, _, _ = energy(c, K, testset, testlabels)
                failed = false
                ind = mod1(ind + 1, y)
                if Ec == 0
                    if verbose
                        open(filename, "a") do io
                        println(io, "$it $d $Ec $E_worst $(minimum(Es)) $testEc")
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

        @info "it = $it [$failures] d = $d Ec = $(Ec) Es = $(mean(Es)) ± $(std(Es)) [$(extrema(Es))] testEc = $testEc"
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

    return c, Ec, Es, teacher, trainset, trainlabels  
end

end # module Chain
