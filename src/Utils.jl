module Utils

export ensure_path

function ensure_path(path::AbstractString)
    if !isdir(path)
        mkpath(path)
    end
end

end