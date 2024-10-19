module Utils

export ensure_path, find_valleys

function ensure_path(path::AbstractString)
    if !isdir(path)
        mkpath(path)
    end
end


"""
    find_valleys(arr::Vector{<:Real}, n::Int)::Vector{Int}

Finds all the local minima in the array `arr` within a window of size `2n+1` and returns their indices.
"""
function find_valleys(arr::Vector{<:Real}, n::Int)::Vector{Int}
    len = length(arr)
    valley_indices = Int[]

    for i in (n + 1):(len - n)
        is_valley = true
        for j in (i-n):(i+n)
            if arr[j] < arr[i]
                is_valley = false
                break
            end
        end
        if is_valley
            push!(valley_indices, i)
        end
    end

    return valley_indices
end


end