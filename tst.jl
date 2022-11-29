
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

function fetcher(path::String, is_generative::Bool=false) 
    @time df = DataFrame(CSV.File(path, stringtype=String))
    println(first(df, 10))
    dict = Dict()
    for e in df[:, :Cabin]
        k = ismissing(e) ? missing : e[1:1]
        dict[k] = haskey(dict, k) ? dict[k] + 1 : 1
    end
    println(unique(df[:, :Cabin]))
    println(length(unique(df[:, :Cabin])), " ..... ", length(collect(skipmissing(df[:, :Cabin]))))
    print_dict(dict)
    println(length(dict))
    println(".......................................................................")
    println(".......................................................................")
    prefixes = [r"A",]
    dict = Dict()
    for e in df[:, :Ticket]
        k = Int(e[1]) <= 57 && Int(e[1]) >= 49 ? missing : e[1:(isnothing(findfirst(' ', e)) ? length(e) : findfirst(' ', e) - 1)]
        dict[k] = haskey(dict, k) ? dict[k] + 1 : 1
    end
    println(unique(df[:, :Ticket]))
    println(length(unique(df[:, :Ticket])), " ..... ", length(collect(skipmissing(df[:, :Ticket]))))
    print_dict(dict)
    println(length(dict))
end

function print_dict(d::Dict)
    vec = sort(collect(keys(d)))
    for k in vec 
        println("(", k, ", ", d[k], ")")
    end
end

fetcher("../data/train.csv", true)
