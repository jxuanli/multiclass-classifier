
function nbc(X_train, y_train, X_test)
    labels = levels(y_train)
    priors = compute_priors(y_train)
    cat_map = cat_nbc(X_train, y_train)
    gau_map = gau_nbc(X_train, y_train)
    is_cat_list = check_cat(X_train)
    res = []
    for i in axes(X_test)[1]
        prob_per_label = []
        for l in labels 
            tmp = priors[l]
            for j in eachindex(X_test[i, :]) 
                if is_cat_list[j] == 1 
                    if haskey(cat_map[l][j].map, X_test[i, j])
                        tmp *= cat_map[l][j].map[X_test[i, j]]
                    end
                else
                    tmp *= compute_gaussian(X_test[i, j], gau_map[y_train[i]], j)
                end
            end
            push!(prob_per_label, tmp)
        end
        push!(res, labels[argmax(prob_per_label)])
    end
    res
end

function compute_priors(y::Vector)
    dict_helper(y)
end

struct Cat_Info 
    map::Dict
end

function cat_nbc(x::Matrix, y::Vector)
    res = Dict()
    for i in eachindex(y) 
        if !haskey(res, y[i])
            res[y[i]] = []
            for j in axes(x)[2]
                push!(res[y[i]], Cat_Info(dict_helper(x[y .== y[i], j], true)))
            end
        end 
    end
    res
end

function dict_helper(vec::Vector, regularize=false)
    dict = Dict()
    for e in vec
        if !isnan(e)
            dict[e] = haskey(dict, e) ? dict[e] + 1 : 1
        end
    end
    res = Dict()
    for (k, v) in dict 
        res[k] = regularize ? v / length(vec) : (v + 1) / (length(vec) + length(dict))
    end
    res
end

struct Gaussian_Info 
    mean::Vector{Float64}
    std::Vector{Float64}
end

function gau_nbc(x::Matrix, y::Vector)
    dict = Dict()
    for i in eachindex(y) 
        if !isnan(y[i])
            if haskey(dict, y[i])
                push!(dict[y[i]], i)
            else 
                dict[y[i]] = [i]
            end 
        end
    end
    res = Dict()
    for (k, v) in dict 
        tmp = []
        for _ in axes(x)[2]
            push!(tmp, [])
        end
        for e in v
            for j in axes(x)[2]
                if !isnan(x[e, j]) 
                    push!(tmp[j], x[e, j])
                end
            end
        end
        means = [] 
        stds = []
        for e in tmp
            push!(means, mean(e))
            push!(stds, std(e))
        end
        res[k] = Gaussian_Info(means, stds)
    end
    res
end

function compute_gaussian(x, info::Gaussian_Info, label)
    erfc((x-info.mean[label])/info.std[label])
end

function check_cat(X_train)
    res = Dict()
    for j in axes(X_train)[2]
        tmp = Dict()
        counter = 0
        for i in eachindex(X_train[:, j])
            if !haskey(tmp, X_train[i, j]) 
                tmp[X_train[i, j]] = 0
                counter += 1
            end
        end
        res[j] = counter < 10 ? 1 : 0
    end
    res
end
