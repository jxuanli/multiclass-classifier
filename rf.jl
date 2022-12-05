
function rf(X_train::Matrix, y_train::Vector, X_test::Matrix, n=5)
    opt_depth, opt_reg, accuracy = rf_cv(X_train, y_train, n)
    rf_tree = rf_helper(X_train, y_train, 0, opt_depth, opt_reg)
    res = []
    for i in axes(X_test)[1]
        tmp = predict(rf_tree, X_test[i, :])
        push!(res, tmp)
    end
    res, accuracy
end

function rf_cv(X_train::Matrix, y_train::Vector, n)
    depths = trunc(Int, length(X_train[1, :])/2):trunc(Int, length(X_train[1, :])*3/20):length(X_train[1, :])*2
    regularizations = [0.01 0.03 0.05 0.08 0.1 0.3 0.5 0.7 0.9 1.0 1.2 1.4 1.8]
    println("...........................training rf depth...........................")
    depth, _ = cv_helper(X_train, y_train, rf_train, nothing, depths, n, "depth", 1 / n)
    println("opt_depth: ", depth)
    println(".......................................................................")
    println(".....................training rf regularization........................")
    reg, accuracy = cv_helper(X_train, y_train, rf_train, depth, regularizations, n, "reg")
    println("opt_regularization: ", reg, " with accuracy: ", accuracy)
    println(".......................................................................")
    depth, reg, accuracy
end

function rf_train(X_train::Matrix, y_train::Vector, X_test::Matrix, arg)
    max_depth = arg[1]
    regularization = 0.0
    if isnothing(max_depth)
        max_depth = arg[2]
    else
        regularization = arg[2]
    end
    rf_tree = rf_helper(X_train, y_train, 0, trunc(Int, max_depth), regularization)
    res = []
    for i in axes(X_test)[1]
        tmp = predict(rf_tree, X_test[i, :])
        push!(res, tmp)
    end
    res
end

function rf_helper(X::Matrix, y::Vector, depth::Int64, max_depth::Int64, regularization::Float64)
    if length(y) == 0
        return nothing
    end
    if depth > max_depth || length(y) <= 10
        dict = Dict()
        pred = nothing
        tmp = 0
        for e in y
            dict[e] = haskey(dict, e) ? dict[e] + 1 : 1
            if dict[e] > tmp
                tmp = dict[e]
                pred = e
            end
        end
        return Tree(nothing, nothing, pred, nothing, nothing, nothing)
    end
    num_features = length(X[1, :])
    min_entropy = typemax(Int)
    feature = 1
    value = 0
    is_cat = 0
    for f in shuffle(1:num_features)
        tmp_value, is_cat, tmp_entropy = rf_step(X[:, f], y)
        if tmp_entropy + regularization < min_entropy
            min_entropy = tmp_entropy
            value = tmp_value
            feature = f
        end
    end
    indices = is_cat == 1 ? X[:, feature] .== value : X[:, feature] .>= value
    Tree(value, feature, nothing, is_cat, rf_helper(X[indices, :], y[indices], depth + 1, max_depth, regularization), rf_helper(X[(!).(indices), :], y[(!).(indices)], depth + 1, max_depth, regularization))
end

function rf_step(x::Vector, y::Vector)
    return length(levels(x)) > 10 ? rf_step_helper(sort(x), x, y, 0) : rf_step_helper(levels(x), x, y, 1)
end

function rf_step_helper(vec::Vector, x::Vector, y::Vector, is_cat::Int64)
    min_entropy = typemax(Int)
    value = vec[1]
    for e in vec
        func(x) = isnan(e) ? e.===x : (is_cat == 1 ? e.==x : e.>=x);
        tmp = entropy(y[func(x)]) + entropy(y[(!).(func(x))])
        if tmp < min_entropy
            min_entropy = tmp
            value = e
        end
    end
    value, is_cat, min_entropy
end

struct Tree
    value::Union{Float64, Nothing}
    feature::Union{Int64, Nothing}
    pred::Union{Int64, Nothing}
    is_cat::Union{Int64, Nothing}
    left::Union{Tree, Nothing}
    right::Union{Tree, Nothing}
end

function predict(t::Tree, x::Vector)
    if !isnothing(t.pred)
        return t.pred
    end
    if isnan(t.value)
        return x[t.feature] === t.value ? (isnothing(t.left) ? predict(t.right, x) : predict(t.left, x)) : (isnothing(t.right) ? predict(t.left, x) : predict(t.right, x))
    end
    if t.is_cat == 1
        return x[t.feature] == t.value ? (isnothing(t.left) ? predict(t.right, x) : predict(t.left, x)) : (isnothing(t.right) ? predict(t.left, x) : predict(t.right, x))
    end
    x[t.feature] >= t.value ? (isnothing(t.left) ? predict(t.right, x) : predict(t.left, x)) : (isnothing(t.right) ? predict(t.left, x) : predict(t.right, x))
end

function entropy(pred_y::Vector)
    if length(pred_y) == 0
        return 1
    end
    dict = Dict()
    for e in pred_y
        if haskey(dict, e)
            dict[e] += 1
        else
            dict[e] = 1
        end
    end
    res = 0
    for (_, v) in dict
        res -= v / length(pred_y) * log(v / length(pred_y))
    end
    res
end
