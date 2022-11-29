
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

function convert_cabin(s::Union{String, Missing})
    res = 0
    if !ismissing(s) && length(s) > 0
        k = s[1]
        if k == 'A'
            res = 7
        elseif k == 'B'
            res = 6
        elseif k == 'C'
            res = 5
        elseif k == 'D'
            res = 4
        elseif k == 'E'
            res = 3
        elseif k == 'F'
            res = 2
        else 
            res = 1
        end
    end
    res
end

function convert_ticket(s::String)
    res = 0
    if s[1] == 'A'
        res = 1
    elseif s[1] == 'C'
        res = 2
    elseif length(s) >= 2 && s[1:2] == "PC"
        res = 3
    elseif length(s) >= 5 && s[1:5] == "SOTON"
        res = 4
    elseif length(s) >= 4 && s[1:4] == "STON"
        res = 5
    elseif s[1] == 'W'
        res = 6
    elseif s[1] == 'S'
        res = 7
    elseif !(Int(s[1]) <= 57 && Int(s[1]) >= 49)
        res = 8
    else 
        res = 0
    end
    res
end

function fetcher(path::String, is_generative::Bool=false) 
    @time df = DataFrame(CSV.File(path, stringtype=String))
    y = [] 
    if columnindex(df, :Survived) != 0
        y = df[:, :Survived]
    end 
    # transform!(df, AsTable(:) .=> ByRow.([r -> convert_cabin(r.Cabin), 
    #                                       r -> convert_cabin(r.Ticket)])
    #                                       .=> [:Cabin, :Ticket])
    transform!(df, AsTable(:) .=> ByRow.([r -> ismissing(r.Cabin) ? 0 : (Int(r.Cabin[1]) <= 57 && Int(r.Cabin[1]) >= 49 ? 1 : 2)])
                                            .=> [:Cabin])
    fill!(df, is_generative)
    dropmissing!(df)
    select!(df, :Pclass, :Sex, :Age, :SibSp, :Parch, :Fare, :Embarked, :Cabin, :Ticket) 
    transform!(df, AsTable(:) .=> ByRow.([r -> r.Sex == "" ? NaN : (r.Sex == "male" ? 1 : 0), 
                                          r -> r.Embarked == "" ? NaN : (r.Embarked == "S" ? 0 : (r.Embarked == "Q" ? 1 : 2)), 
                                          r -> Int(r.Ticket[1]) <= 57 && Int(r.Ticket[1]) >= 49 ? 0 : 1])
                                          .=> [:Sex, :Embarked, :Ticket])
    X = reshape(Vector(df[1, :]),(1,length(df[1, :])))
    for i in 2:rownumber(df[end, :])  
        X = [X; reshape(Vector(df[i, :]),(1,length(df[1, :])))]
    end
    reshape(X, (rownumber(df[end, :]), length(df[1, :]))), mean(X, dims=1), std(X, dims=1), y
end
