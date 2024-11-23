module Utils

export ensure_path, find_valleys, get_closest_value, max_index_in_dir

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

    for i ∈ (n+1):(len-n)
        is_valley = true
        for j ∈ (i-n):(i+n)
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

"""
	get_closest_value(val::Real, arr::Vector{<:Real})::Real

Finds the value in `arr` that is closest to `val`.
"""
function get_closest_value(val::Real, arr::Vector{<:Real})::Real
    closest_val = arr[1]
    closest_dist = abs(val - closest_val)
    for i = 2:length(arr)
        dist = abs(val - arr[i])
        if dist < closest_dist
            closest_val = arr[i]
            closest_dist = dist
        end
    end
    return closest_val
end


"""
	max_index_in_dir(dir_path::AbstractString)::Int

Finds the maximum index in the directory `dir_path` by parsing the directory names.
Assumes that the directory names are of the form `xxx_#n` where `n` is an integer.

Returns -1 by default.

# Example:

	dir_path
	├── xxx_#0
	├── xxx_#2
	└── xxx_#3

	@assert max_index_in_dir(dir_path) == 3
"""
function max_index_in_dir(dir_path::AbstractString)::Int
    max_index = -1
    if !isdir(dir_path)
        return max_index
    end

    for dir in readdir(dir_path)
        if isdir(joinpath(dir_path, dir))
            m = match(r"#(\d)$", dir)
            if m === nothing
                continue
            end

            index = parse(Int, m.captures[end])
            max_index = max(max_index, index)
        end
    end

    return max_index
end

end
