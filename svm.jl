# function lp(X_train::Matrix, y::Vector, λ=100)
#     map(x -> x == 0 ? -1 : 1, y)
#     # labels = levels(y)
#     X = [repeat([1], length(X_train[:,1])) X_train] 
#     model = Model(HiGHS.Optimizer)
#     @variable(model, w[1:length(X[1, :])])
#     @variable(model, w_0)
#     @variable(model, ζ[1:length(X[:, 1])] .≥ 0)
#     # @variable(model, W[1:length(labels), 1:length(X[1, :])])
#     # @variable(model, ζ[1:length(X[:, 1]), 1:length(labels)] .≥ 0)
#     for i in axes(X)[1]
#         # for j in 1:length(labels)
#         #     if j != y[i] + 1
#         #         @constraint(model, W[y[i]+1, :] .* X[i, :] .+ ζ[i, j] .≥ 1 .+ W[j, :] .* X[i, :])
#         #     end
#         # end
#         @constraint(model, y[i]*((w * X[i, :]) + w_0) ≥ 1 - ζ[i])
#     end
#     @objective(model, Min, sum(w.^2)/2 + λ*sum(ζ))
#     optimize!(model)
#     value.(w), value.(ζ), value.(w_0)
# end

function svm(X_train::Matrix, y_train::Vector, X_test::Matrix, n=5)
    λ, accuracy = svm_cv(X_train, y_train, n)
    means, stds = get_normalization_info(X_train)
    svm_predict([repeat([1], length(X_train[:,1])) (X_train .- means) ./ stds], y_train, (X_test .- means) ./ stds, λ), accuracy
end

function svm_cv(X_train::Matrix, y_train::Vector, n::Int64)
    λs = [0.01 0.03 0.1 0.3 1 3 10 30 100 300]
    println("...........................training svm ...............................")
    λ, accuracy = cv_helper(X_train, y_train, svm_train, nothing, λs, n, "λ")
    println("opt_λ: ", λ, " with accuracy: ", accuracy)
    println(".......................................................................")
    λ, accuracy
end

function svm_train(X_train::Matrix, y_train::Vector, X_test::Matrix, arg)
    means, stds = get_normalization_info(X_train)
    svm_predict([repeat([1], length(X_train[:,1])) (X_train .- means) ./ stds], y_train, (X_test .- means) ./ stds, arg[2])
end

function svm_predict(X::Matrix, y::Vector, X_test::Matrix, λ)
    W = Variable(length(1:length(levels(y))), length(1:length(X[1, :])))
    obj = sumsquares(W)/2 * λ 
    for i in axes(X)[1]
        for j in axes(W)[1]
            obj += y[i] + 1 == j ? 0.0 : max(0, 1-W[y[i]+1,:]*X[i,:]+W[j,:]*X[i,:])
        end
    end
    p = minimize(obj)
    solve!(p, SCS.Optimizer; silent_solver=true)
    w = evaluate(W)
    tmp = [repeat([1], length(X_test[:,1])) X_test] * w' 
    res = [] 
    for i in axes(tmp)[1]
        push!(res, argmax(tmp[i, :]) - 1)
    end
    res
end