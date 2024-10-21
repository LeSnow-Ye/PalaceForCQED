module PalaceRunner

export palace_run, load_config, save_config

using JSON

"""
Run Palace. Add "--use-hwthread-cpus" to `launcher_args` for multi-cpu machines.
See https://www.open-mpi.org/doc/v4.0/man1/mpirun.1.php for more details.
"""
function palace_run(config_path::AbstractString, np::Integer, launcher_args::AbstractString="")
    if launcher_args == ""
        run(`palace -np $(np) $(config_path)`)
    else
        run(`palace -np $(np) $(config_path) -launcher-args $(launcher_args)`)
    end
end

function load_config(config_path::AbstractString)
    open(config_path, "r") do f
        return JSON.parse(join(getindex.(split.(eachline(f), "//"), 1), "\n"))
    end
end

function save_config(config::Any, config_path::AbstractString)
    open(config_path, "w") do f
        return JSON.print(f, config, 4)
    end
end

end