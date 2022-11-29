using CSV, DataFrames, CategoricalArrays, Statistics, SpecialFunctions, Random;
include("./knn.jl")
include("./nbc.jl")
include("./rf.jl")
include("./fetcher.jl")

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

function cv_helper(X_train, y_train::Vector, func::Function, arg, vec, n::Int64, name::String, regularization_factor=0)
    shuffled_indices = shuffle(1:length(y_train))
    batch = floor(length(y_train) / n)
    losses = []
    for e in vec
        loss = e * regularization_factor
        for i in n
            start = trunc(Int, (i - 1) * batch) + 1
            finish = i == n ? length(y_train) : trunc(Int, i * batch)
            bitmap = repeat([0], length(y_train))
            bitmap[start:finish] .= 1
            bitmap = BitArray(bitmap)
            args = [arg e]
            actual = func(X_train[shuffled_indices[(!).(bitmap)], :], y_train[shuffled_indices[(!).(bitmap)]], X_train[shuffled_indices[bitmap], :], args)
            loss += sum(abs.(y_train[shuffled_indices[bitmap]] .- actual)) / n
        end
        loss = round(loss, digits=3)
        println(name, ": ", e, " loss: ", loss)
        push!(losses, loss)
    end
    vec[argmin(losses)]
end

# 77.8% (22.2%)
# 78.5% (11.3%)

function main()
    Random.seed!(1)
    X_train, X_means, X_stds, y_train = fetcher("../data/train.csv")
    X_test, _, _, _ = fetcher("../data/test.csv")
    X_train_normalized = (X_train .- X_means) ./ X_stds
    X_test_normalized = (X_test .- X_means) ./ X_stds
    X_train, _, _, y_train = fetcher("../data/train.csv", true)
    X_test, _, _, _ = fetcher("../data/test.csv", true)
    println(".......................................................................")
    @time "knn" knn_res = knn(X_train_normalized, y_train, X_test_normalized)
    @time "rf" rf_res = rf(X_train, y_train, X_test)
    println(".......................................................................")
    println(".......................................................................")
    @time "nbc" nbc_res = nbc(X_train, y_train, X_test)
    println(".......................................................................")
    println("knn: ", sum(knn_res) / length(knn_res))
    println("rf: ", sum(rf_res) / length(rf_res))
    println("nbc: ", sum(nbc_res) / length(nbc_res))
    println(".......................................................................")
    println(".......................................................................")
    println("knn vs rf: ", sum(abs.(knn_res - rf_res)))
    println("nbc vs rf: ", sum(abs.(nbc_res - rf_res)))
    println("nbc vs knn: ", sum(abs.(nbc_res - knn_res)))
    res = rf_res + nbc_res + knn_res
    res = map(x -> x > 1 ? 1 : 0, res)
    println(".......................................................................")
    println(".......................................................................")
    println("res vs knn: ", sum(abs.(res - knn_res)))
    println("res vs rf: ", sum(abs.(res - rf_res)))
    println("res vs nbc: ", sum(abs.(res - nbc_res)))
    println(".......................................................................")
    println(".......................................................................")
    tmp = DataFrame(PassengerId=892:1309, Survived=Vector{Int64}(res))
    println(first(tmp, 10))
    CSV.write("res1.csv", tmp)
    # CSV.write("gender_submission.csv", tmp)
end

@time "main" main()