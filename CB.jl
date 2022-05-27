module BinChain

using Statistics
using LinearAlgebra
using Random

function dataset(n::Int, p::Int, k::Int = 1; fc = false)
    @assert n % k == 0
    if fc
        nk = n ÷ k
        trainset = repeat(rand((-1,1), (nk, p)), outer=(k,1))
    else
        trainset = rand((-1,1), (n, p))
    end
    trainlabels = rand((-1,1), p)
    return trainset, trainlabels
end

function teacher_dataset(n::Int, p::Int, teacher::Vector, k::Int = 1; fc = false)
    if fc
        nk = n ÷ k
        trainset1 = rand((-1,1), (nk, p))
        trainset = repeat(trainset1, outer=(k,1))
        mult_mat = teacher .* trainset1
    else
        trainset = rand((-1,1), (n, p))
        mult_mat = teacher .* trainset
    end
    trainlabels = [sign(sum(mult_mat[:,i])) for i=1:p]
    return trainset, trainlabels
end

function energy(w::Vector{T}, K::Int, trainset::AbstractMatrix{S}, trainlabels::AbstractVector) where {T,S}
    n = length(w)
    p = length(trainlabels)
    @assert n % K == 0
    @assert size(trainset) == (n, p)
    nh = n ÷ K # inputs per hidden unit

    W = reshape(w, nh, K)
    X = reshape(trainset, nh, K, p)
    # indices = Int[]
    errors = 0
    TT = promote_type(T, S)
    @inbounds for μ = 1:p
        Δ = 0
        for j = 1:K
            Δj = zero(TT)
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
        trainset::AbstractMatrix,
        trainlabels::AbstractVector,
        K::Int,
    )
    n = length(c)

    ## distance from barycenter (expected)
    # ρ = d * norm(c) / √(1 - d^2 / 2)

    @assert all(-1 ≤ c[i] ≤ 1 for i in 1:n)
    nc = norm(c)^2 / n
    # println(d^2 / 2)
    # println("nc=$nc (1-d^2/2)=$(1-(d^2)/2)")
    # @show nc
    # @show 1 - d^2 / 2
    # x = (1 - (1 - (d^2) / 2) / nc) / 4
    # println("x=$x")
    # @assert -1e-5 ≤ x ≤ 1 x
    # x = max(0.0, x)
    # x = 0.0

    # dcs = Float64[]
    # qcs = Float64[]
    # for j = 1:10_000
        p_new = [rand() < (1+c[i])/2 ? 1 : -1 for i = 1:n]
        # for i = 1:n
        #     if rand() < x
        #         p_new[i] *= -1
        #     end
        # end
        # dc = norm(c - p_new) / √n
        # dc2 = norm(c - p_new)^2 / n
        # dc2 = (norm(c)^2 + norm(p_new)^2 - 2 * (c ⋅ p_new)) / n
        # dc2 = (nc + 1 - 2 * (c ⋅ p_new) / n)
        qc = (p_new ⋅ c) / n
        # dc2 = (nc + 1 - 2 * qc)
    #     push!(dcs, dc)
    #     push!(qcs, qc)
    # end

    # println("nc=$(norm(c)/√n) dc=$(mean(dcs)) exp=$(d/√2) qc=$(mean(qcs)) exp=$(nc * (1 - 2x))")
    # println("nc=$(norm(c)/√n) dc=$(mean(dcs)) exp=$(nc + 1 - 2 * (nc * (1 - 2x)) ) qc=$(mean(qcs)) exp=$(nc * (1 - 2x))")
    # println("nc=$(norm(c)/√n) dc=$(mean(dcs)) exp=$(1 - nc + 4nc * x ) qc=$(mean(qcs)) exp=$(nc * (1 - 2x))")
    # println("nc=$(norm(c)/√n) dc=$(mean(dcs)) exp=$(d / √2) qc=$(mean(qcs)) exp=$(nc * (1 - 2x))")
    # println("nc=$(norm(c)/√n) dc=$dc exp=$(d/√2)")

    # q = (p_new ⋅ c) / n
    # println("q=$(mean(qs)) exp=$(norm(c)^2 / n)")

    ## generate a new random direction
    # x = d/√2 * randn(n)

    ## choose the best of the two directions
    # p_new1 = sign.(c + x)
    # p_new2 = sign.(c - x)

    # dc1 = norm(c - p_new1) / √n
    # dc2 = norm(c - p_new2) / √n
    # db1 = norm(sign.(c) - p_new1) / √n
    # db2 = norm(sign.(c) - p_new2) / √n
    # println("nc=$(norm(c)/√n) dc1=$dc1 dc2=$dc2 exp=$(d/√2) db1=$db1 db2=$db2")



    E_new = energy(p_new, K, trainset, trainlabels)
    # E_new2 = energy(p_new2, K, trainset, trainlabels)
    ## new point
    # p_new, E_new = E_new1 ≤ E_new2 ? (p_new1, E_new1) : (p_new2, E_new2)


    return p_new, E_new
end

function simplex_chain(
        n::Int,
        p::Int,
        y::Int;
        K::Int = 3,
        d₀::Float64 = 1.414,
        seed::Int = 411068089483816338,
        iters::Int = 1_000,
        # rescale_factor::Float64 = 0.99,
        steps::Int = 100,
        max_failures::Integer = 5,
        verbose::Bool = false,
        teacher::Bool = false,
        p_test::Int = 1000,
        B::Int = 1,
        fc::Bool = false
    )

    @assert 0 ≤ d₀ ≤ √2

    d = d₀
    mq₀ = d₀^2 / 2
    δq = mq₀ / steps

    if verbose
        dirname = "runs" * (teacher ? "_teacher" : "")
        mkpath(dirname)
        # filename = joinpath(dirname, "run_n_$(n)_p_$(p)_y_$(y)_k_$(K)_d_$(d)_df_$(rescale_factor).txt")
        filename = joinpath(dirname, "run_n_$(n)_p_$(p)_y_$(y)_k_$(K)_d_$(d)_st_$(steps)_it_$(iters).txt")
        open(filename, "a") do io
            println(io, "#it, d, Ec, Eworst, Ebest, Ec" * (teacher ? " test" : ""))
        end
    end

    seed > 0 && Random.seed!(seed)

    if teacher
        if fc
            @assert n % K == 0
            nk = n ÷ K
            W_teacher = rand((-1,1), nk)
        else
            W_teacher = rand((-1,1), n)
        end
        trainset, trainlabels = teacher_dataset(n, p, W_teacher, K; fc)
        testset, testlabels = teacher_dataset(n, p_test, W_teacher, K; fc)
    else
        trainset, trainlabels = dataset(n, p, K; fc)
        testset, testlabels = zeros(n,0), zeros(0)
    end

    # c = √(1 - d^2 / 2) * randn(n)
    c = (2 * rand(n) .- 1)
    c *= √((1 - d^2 / 2) * n) / norm(c)
    while !all(-1.0 .< c .< 1.0)
        c = clamp.(c, -0.5, 0.5) # XXX!!!!!!!!!
        c *= √((1 - d^2 / 2) * n) / norm(c)
        # @show c
        # break
    end
    # c = (2 * rand(n) .- 1) # XXX
    # @show norm(c)^2 / n
    # @show 1 - d^2 / 2
    # c = map(x->sign(x)*abs(x)^0.1, (2 * rand(n) .- 1)) # XXX
    # @show c

    Ec = energy(sign.(c), K, trainset, trainlabels)
    testEc = teacher ? energy(sign.(c), K, testset, testlabels) / p_test : 0.0
    testEcc = teacher ? energy(c, K, testset, testlabels) / p_test : 0.0
    normc2 = norm(c)^2 / n

    Esb = [Int[typemax(Int) ÷ B for b = 1:B] for j = 1:y]
    Es = Int[sum(Esb[j]) for j = 1:y]

    E_best, E_worst = extrema(Es)

    @info "it = 0 d = $d Ec = $(Ec) Es = $(mean(Es)) ± $(std(Es)) [$(extrema(Es))]" * (teacher ? " testEc = $testEc ($testEcc)" : "")
    # println("norm(c)^2: $normc2 should be: $(1 - d^2 / 2)")

    @assert p % B == 0
    bs = p ÷ B
    rn(k) = (1:bs) .+ (k - 1) * bs
    trainset_b = [@view(trainset[:,rn(k)]) for k = 1:B]
    trainlabels_b = [@view(trainlabels[rn(k)]) for k = 1:B]

    ind = 1
    it = 0
    failures = 0
    # while Ec > 0
    for mq in LinRange(mq₀, δq, steps)
        it += 1
        E_worst = maximum(Es)
        acc = 0
        cl = 0
        attempt = 0
        while (acc < iters && attempt < 10 * iters) # || (acc > 0.3 * attempt && attempt < 100 * iters)
        # while attempt < iters || acc > 0.3 * attempt
        # for attempt = 1:iters

            # ind = argmax(Es)
            attempt += 1
            if attempt > 1 && attempt % iters == 1
                print(".")
            end
            b = rand(1:B)
            p_new, E_new_b = gen_candidate(c, d, trainset_b[b], trainlabels_b[b], K)
            E_new = Es[ind] - Esb[ind][b] + E_new_b
            # if E_new_b * (p / length(trainlabels_b[b])) > E_worst
            #     continue
            # end
            # E_new = E_new_b
            # for b1 = 1:B
            #     b1 == b && continue
            #     E_new += energy(p_new, K, trainset_b[b1], trainlabels_b[b1])
            # end
            if E_new < E_worst
                Esb_tmp = [b1 == b ? E_new_b : energy(p_new, K, trainset_b[b1], trainlabels_b[b1]) for b1 = 1:B]
                E_new = sum(Esb_tmp)
            end
            if E_new < E_worst
                # Es[ind] = E_new
                # Esb_tmp = [b1 == b ? E_new_b : energy(p_new, K, trainset_b[b1], trainlabels_b[b1]) for b1 = 1:B]
                # E_new = sum(Esb_tmp)
                # E_new ≥ E_worst && continue
                Esb[ind] = Esb_tmp
                # Esb[ind][b] = E_new_b
                # # @assert sum(Esb[ind]) == E_new
                # for b1 = 1:B
                #     b1 == b && continue
                #     Esb[ind][b1] = energy(p_new, K, trainset_b[b1], trainlabels_b[b1])
                # end
                # E_new = sum(Esb[ind])
                # E_new ≥ E_worst && continue
                Es[ind] = E_new
                E_worst = maximum(Es)
                nc = norm(c)^2 / n
                pc = p_new ⋅ c / n
                r = pc / nc
                # α = 1/y * (√((1 + y^2) - 1/nc) - 1)
                # β = 1/y
                # α = 1/y * (√((r^2 + y^2) - 1/nc) - r)

                # β = exp(-E_new) / sum(exp.(-Es))
                β = 1/y
                k = r^2 - 1 / nc
                # k < 0 && (β = clamp(β, -√(-1/k)+1e-5, √(-1/k)-1e-5))
                # @show β, k, r^2, - 1 / nc
                α = -r * β + √(1 + k * β^2)
                # @show α

                c = α * c + β * p_new
                @assert norm(c)^2 / n ≈ nc
                if !all(-1 .< c .< 1)
                    c = clamp.(c, -1.0, 1.0)
                    cl += 1
                end
                acc += 1
                # if Ec == 0
                #     if verbose
                #         testEc = teacher ? energy(sign.(c), K, testset, testlabels) / p_test : 0.0
                #         open(filename, "a") do io
                #             println(io, "$it $d $Ec $E_worst $(minimum(Es))" * teacher ? " $testEc" : "")
                #         end
                #     end
                #     break
                # end
            end
            ind = mod1(ind + 1, y)
        end
        attempt > iters && println()

        if acc == 0
            failures += 1
        else
            failures = 0
        end

        normc2 = norm(c)^2 / n

        Ec = energy(sign.(c), K, trainset, trainlabels)
        testEc = teacher ? energy(sign.(c), K, testset, testlabels) / p_test : 0.0
        testEcc = teacher ? energy(c, K, testset, testlabels) / p_test : 0.0
        @info "it = $it [$failures] d = $d q = $(1-d^2/2) Ec = $(Ec) Es = $(mean(Es)) ± $(std(Es)) [$(extrema(Es))]" * (teacher ? " testEc = $testEc ($testEcc)" : "")
        println("acc.rate = $(acc / attempt) cl.rate = $(cl / attempt)")
        # println("norm(c)^2: $normc2 should be: $(1 - d^2 / 2)")

        verbose && open(filename, "a") do io
            println(io, "$it $d $Ec $E_worst $(minimum(Es))" * (teacher ? " $testEc" : ""))
        end

        if Ec == 0
            @info "solved"
            break
        end

        # d *= rescale_factor
        mq = d^2 / 2 - δq

        if mq < δq / 2
            @info "min dist reached, give up"
            break
        end

        d = √(2 * mq)

        c *= √((1 - d^2 / 2) * n) / norm(c)
        c = clamp.(c, -1.0, 1.0)

        failures < max_failures && continue

        if failures ≥ max_failures
            @info "failed $failures times, give up"
            break
        end

    end

    return c, Ec, Es
end

end # module BinChain
