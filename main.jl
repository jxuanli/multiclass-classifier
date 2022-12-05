using CSV, DataFrames, CategoricalArrays, Statistics, SpecialFunctions, Random, JuMP, HiGHS, Convex, SCS;
include("./knn.jl")
include("./nbc.jl")
include("./rf.jl")
include("./svm.jl")
include("./fetcher.jl")

function get_normalization_info(X::Matrix)
    means = []
    stds = []
    for i in axes(X)[2]
        # push!(means, length(levels(X[:, i])) < 10 ? 0 : mean(X[:, i]))
        # push!(stds,  length(levels(X[:, i])) < 10 ? 0 :  std(X[:, i]))
        push!(means, mean(X[:, i]))
        push!(stds,  std(X[:, i]))
    end
    reshape(means, (1, length(means))), reshape(stds, (1, length(stds)))
end

function most_frequent(vec::Vector)
    dict = Dict()
    max_frequency = 0
    key = vec[1]
    for e in vec 
        dict[e] = haskey(dict, e) ? dict[e] + 1 : 1
        if dict[e] > max_frequency
            max_frequency = dict[e]
            key = e
        end
    end
    key
end

mutable struct Atomic
    @atomic x::Float64
end

function cv_helper(X_train, y_train::Vector, func::Function, arg, vec, n::Int64, name::String, regularization_factor=0)
    shuffled_indices = shuffle(1:length(y_train))
    batch = floor(length(y_train) / n)
    losses = repeat([0.0], length(vec))
    Threads.@threads :dynamic for idx in eachindex(vec)
        e = vec[idx]
        loss = Atomic(e * regularization_factor)
        Threads.@threads :dynamic for i in 1:n
            start = trunc(Int, (i - 1) * batch) + 1
            finish = i == n ? length(y_train) : trunc(Int, i * batch)
            bitmap = repeat([0], length(y_train))
            bitmap[start:finish] .= 1
            bitmap = BitArray(bitmap)
            args = [arg e]
            actual = func(X_train[shuffled_indices[(!).(bitmap)], :], y_train[shuffled_indices[(!).(bitmap)]], X_train[shuffled_indices[bitmap], :], args)
            @atomic loss.x += sum(abs.(y_train[shuffled_indices[bitmap]] .- actual)) / n
        end
        println(name, ": ", e, " loss: ", round(loss.x, digits=3))
        losses[idx] = round(loss.x, digits=3)
    end
    vec[argmin(losses)], 1 - minimum(losses) / batch
end

# 77.8% (22.2%)
# 78.5% (11.3%)

function main()
    Random.seed!(1)
    X_train, y_train = fetcher("../data/train.csv")
    X_test, _,  = fetcher("../data/test.csv")
    println(".......................................................................")
    @time "knn" knn_res, knn_accuracy = knn(X_train, y_train, X_test)
    println(".......................................................................")
    @time "svm" svm_res, svm_accuracy = svm(X_train, y_train, X_test)
    println(".......................................................................")
    X_train, y_train = fetcher("../data/train.csv", true)
    X_test, _ = fetcher("../data/test.csv", true)
    println(".......................................................................")
    @time "rf" rf_res, rf_accuracy = rf(X_train, y_train, X_test)
    println(".......................................................................")
    println(".......................................................................")
    @time "nbc" nbc_res, nbc_accuracy = nbc(X_train, y_train, X_test)
    println(".......................................................................")
    println("knn: ", sum(knn_res) / length(knn_res))
    println("rf: ", sum(rf_res) / length(rf_res))
    println("nbc: ", sum(nbc_res) / length(nbc_res))
    println("svm: ", sum(svm_res) / length(svm_res))
    println(".......................................................................")
    println(".......................................................................")
    println("knn vs rf: ", sum(abs.(knn_res - rf_res)))
    println("nbc vs rf: ", sum(abs.(nbc_res - rf_res)))
    println("nbc vs knn: ", sum(abs.(nbc_res - knn_res)))
    println("svm vs rf: ", sum(abs.(svm_res - rf_res)))
    println("svm vs nbc: ", sum(abs.(svm_res - nbc_res)))
    println("svm vs knn: ", sum(abs.(svm_res - knn_res)))
    accuracy_sum = knn_accuracy + svm_accuracy + rf_accuracy + nbc_accuracy
    res = rf_res * (rf_accuracy / accuracy_sum) + svm_res * (svm_accuracy / accuracy_sum) + knn_res * (knn_accuracy / accuracy_sum) + nbc_res * (nbc_accuracy / accuracy_sum) 
    res = map(x -> x > 0.5 ? 1 : 0, res)
    println(".......................................................................")
    println(".......................................................................")
    println("res vs knn: ", sum(abs.(res - knn_res)))
    println("res vs rf: ", sum(abs.(res - rf_res)))
    println("res vs nbc: ", sum(abs.(res - nbc_res)))
    println("res vs svm: ", sum(abs.(res - svm_res)))
    println(".......................................................................")
    println(".......................................................................")
    tmp = DataFrame(PassengerId=892:1309, Survived=Vector{Int64}(res))
    println(first(tmp, 10))
    # CSV.write("gender_submission.csv", tmp)
    CSV.write("res2.csv", tmp)
end

@time "main" main()