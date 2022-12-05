
function find_kth(vec::Vector{Float64}, k::Int64)
    mids = Vector{Float64}()
    if (length(vec) >= 5) 
        i = 1
        while i + 4 <= length(vec)
            push!(mids, sort(vec[i:i+4])[3])
            i += 5
        end
        p = find_kth(mids, ceil(Int64, length(mids) / 2))
        lesser = Vector{Float64}()
        greater = Vector{Float64}()
        has_same = false
        for i in eachindex(vec) 
            if vec[i] > p || (vec[i] == p && has_same)
                push!(greater, vec[i])
            elseif vec[i] < p 
                push!(lesser, vec[i])
            elseif vec[i] == p && !has_same
                has_same = true
            end
        end
        if k - 1 == length(lesser)
            return p
        elseif length(lesser) > k - 1 
            return find_kth(lesser, k)
        else 
            return find_kth(greater, k - 1 - length(lesser))
        end
    else
        return sort(vec)[k]
    end
end

function cv_knn_gen(X_train::Matrix, y_train::Vector, x_test::Matrix, k::Int64)
    dist = vec(sqrt.(sum((X_train .- x_test).^2, dims=2)))
    kth = find_kth(dist, k)
    indices = Vector{Int64}()
    for i in eachindex(dist)
        if dist[i] <= kth 
            push!(indices, i)
        end
        if length(indices) == k
            break
        end
    end
    most_frequent(vec(y_train[reshape(indices, (1, length(indices)))]))
end

function knn_cv(X_train::Matrix, y_train::Vector, n::Int64)
    ks = 10:2:40
    println("...........................training knn ...............................")
    k, accuracy = cv_helper(X_train, y_train, knn_train, nothing, ks, n, "k")
    println("opt_k: ", k, " with accuracy: ", accuracy)
    println(".......................................................................")
    k, accuracy
end

function knn_train(X_train::Matrix, y_train::Vector, X_test::Matrix, arg) 
    k = arg[2]
    res = []
    X_means, X_stds = get_normalization_info(X_train)
    X_test_normalized = (X_test .- X_means) ./ X_stds
    for i in axes(X_test)[1]
        push!(res, cv_knn_gen((X_train .- X_means) ./ X_stds, y_train, reshape(X_test_normalized[i, :], (1, length(X_test[i, :]))), k))
    end
    res
end

function knn(X_train::Matrix, y_train::Vector, X_test::Matrix, n=5)
    k, accuracy = knn_cv(X_train, y_train, n)
    pred = []
    X_means, X_stds = get_normalization_info(X_train)
    X_test_normalized = (X_test .- X_means) ./ X_stds
    for i in axes(X_test)[1]
        push!(pred, cv_knn_gen((X_train .- X_means) ./ X_stds, y_train, reshape(X_test_normalized[i, :], (1, length(X_test[i, :]))), k))
    end
    pred, accuracy
end
