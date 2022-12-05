
using DataFrames, CSV

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

function get_default(vec::Vector)
    if typeof(vec) == typeof([missing, ""]) 
        return ""
    end
    NaN
end

function fill!(df::DataFrame, is_generative::Bool) 
    for i in 1:ncol(df) 
        e = is_generative ? get_default(Vector(df[:, i])) : most_frequent(collect(skipmissing(df[:, i])))
        df[:, i] = coalesce.(df[:, i], e)
    end
end

# function fetcher(path::String, is_generative::Bool=false) 
#     @time df = DataFrame(CSV.File(path, stringtype=String))
#     println(first(df, 10))
#     dict = Dict()
#     for e in df[:, :Cabin]
#         k = ismissing(e) ? missing : e[1:1]
#         dict[k] = haskey(dict, k) ? dict[k] + 1 : 1
#     end
#     println(unique(df[:, :Cabin]))
#     println(length(unique(df[:, :Cabin])), " ..... ", length(collect(skipmissing(df[:, :Cabin]))))
#     print_dict(dict)
#     println(length(dict))
#     println(".......................................................................")
#     println(".......................................................................")
#     prefixes = [r"A",]
#     dict = Dict()
#     for e in df[:, :Ticket]
#         k = Int(e[1]) <= 57 && Int(e[1]) >= 49 ? missing : e[1:(isnothing(findfirst(' ', e)) ? length(e) : findfirst(' ', e) - 1)]
#         dict[k] = haskey(dict, k) ? dict[k] + 1 : 1
#     end
#     println(unique(df[:, :Ticket]))
#     println(length(unique(df[:, :Ticket])), " ..... ", length(collect(skipmissing(df[:, :Ticket]))))
#     print_dict(dict)
#     println(length(dict))
# end



function convert_cabin(s::Union{String, Missing})
    res = repeat([0], 7)
    if !ismissing(s) && length(s) > 0
        res[1] = 1
        k = s[1]
        if k == 'A'
            res[2] = 1
        elseif k == 'B'
            res[3] = 1
        elseif k == 'C'
            res[4] = 1
        elseif k == 'D'
            res[5] = 1
        elseif k == 'E'
            res[6] = 1
        elseif k == 'F'
            res[7] = 1
        else 
            res[1] = 1
        end
    end
    res
end

function convert_ticket(s::String)
    res = repeat([0], 10)
    if s[1] == 'A'
        res[1] = 1
        res[2] = 1
    elseif s[1] == 'C'
        res[1] = 1
        res[3] = 1
    elseif length(s) >= 2 && s[1:2] == "PC"
        res[1] = 1
        res[4] = 1
    elseif length(s) >= 5 && s[1:5] == "SOTON"
        res[1] = 1
        res[5] = 1
    elseif length(s) >= 4 && s[1:4] == "STON"
        res[1] = 1
        res[6] = 1
    elseif s[1] == 'W'
        res[1] = 1
        res[7] = 1
    elseif s[1] == 'S'
        res[1] = 1
        res[8] = 1
    elseif !(Int(s[1]) <= 57 && Int(s[1]) >= 49)
        res[1] = 1
        res[9] = 1
    else 
        res[10] = 1
    end
    res
end

function fetcher(path::String, is_generative::Bool=false) 
    @time df = DataFrame(CSV.File(path, stringtype=String))
    y = [] 
    if columnindex(df, :Survived) != 0
        y = df[:, :Survived]
    end 
    transform!(df, AsTable(:) .=> ByRow.([r -> convert_cabin(r.Cabin)[1], 
                                          r -> convert_cabin(r.Cabin)[2], 
                                          r -> convert_cabin(r.Cabin)[3], 
                                          r -> convert_cabin(r.Cabin)[4], 
                                          r -> convert_cabin(r.Cabin)[5], 
                                          r -> convert_cabin(r.Cabin)[6], 
                                          r -> convert_cabin(r.Cabin)[7], 
                                          r -> convert_ticket(r.Ticket)[1], 
                                          r -> convert_ticket(r.Ticket)[2], 
                                          r -> convert_ticket(r.Ticket)[3], 
                                          r -> convert_ticket(r.Ticket)[4], 
                                          r -> convert_ticket(r.Ticket)[5], 
                                          r -> convert_ticket(r.Ticket)[6], 
                                          r -> convert_ticket(r.Ticket)[7], 
                                          r -> convert_ticket(r.Ticket)[8], 
                                          r -> convert_ticket(r.Ticket)[9], 
                                          r -> convert_ticket(r.Ticket)[10]])
                                          .=> [:Cabin1, :Cabin2, :Cabin3, :Cabin4, :Cabin5, :Cabin6, :Cabin7,
                                                :Ticket1, :Ticket2, :Ticket3, :Ticket4, :Ticket5, :Ticket6, :Ticket7, :Ticket8, :Ticket9, :Ticket10])
    # transform!(df, AsTable(:) .=> ByRow.([r -> ismissing(r.Cabin) ? 0 : (Int(r.Cabin[1]) <= 57 && Int(r.Cabin[1]) >= 49 ? 1 : 2)])
    #                                         .=> [:Cabin])
    fill!(df, is_generative)
    dropmissing!(df)
    select!(df, :Sex, :Pclass, :Age, :SibSp, :Parch, :Fare, :Embarked,
                :Cabin1, :Cabin2, :Cabin3, :Cabin4, :Cabin5, :Cabin6, :Cabin7,
                :Ticket1, :Ticket2, :Ticket3, :Ticket4, :Ticket5, :Ticket6, :Ticket7, :Ticket8, :Ticket9, :Ticket10) 
    transform!(df, AsTable(:) .=> ByRow.([r -> r.Sex == "" ? NaN : (r.Sex == "male" ? 1 : 0), 
                                          r -> r.Embarked == "" ? NaN : (r.Embarked == "S" ? 0 : (r.Embarked == "Q" ? 1 : 2))])
                                          .=> [:Sex, :Embarked])
    X = reshape(Vector(df[1, :]),(1,length(df[1, :])))
    for i in 2:rownumber(df[end, :])  
        X = [X; reshape(Vector(df[i, :]),(1,length(df[1, :])))]
    end
    reshape(X, (rownumber(df[end, :]), length(df[1, :]))), y
end


function print_dict(d::Dict)
    vec = sort(collect(keys(d)))
    for k in vec 
        println("(", k, ", ", d[k], ")")
    end
end

# using JuMP, HiGHS

# function tst()
#     A = [10 10; 10 10]
#     K = [1 2; 3 4; 5 6]
#     model = Model(HiGHS.Optimizer)
#     @variable(model, X[1:2,1:3]>=0)
#     @objective(model, Max, sum(X*K))
#     @constraint(model, con, A - X*K .>= 0)
#     optimize!(model)
#     value.(X)
# end
# println(tst())

mutable struct A 
    @atomic x::Int64
end

function compute(s, i) 
    start = time_ns()
    while (time_ns() - start) / 1e9 < s
    end
    # println(i * 10)
    i * 10
end

@time begin
    # a = Threads.@spawn compute(1)
    # b = wait(a)
    dict = [0 0 0 0 0]
    println(Threads.nthreads())
    Threads.@threads :dynamic for i in 1:5
        res = A(0)
        Threads.@threads :dynamic for j in 1:5
            @atomic res.x += compute(1, j)
        end
        dict[i] = res.x
        println(res.x)
    end
    println(dict)
end
