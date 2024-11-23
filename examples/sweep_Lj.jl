include("run.jl")

"""
    sweep_Lj(inds::Vector{Float64})

Sweeps the Josephson inductance Lj for the SC qubit.

# Arguments
- `inds::Vector{Float64}`: Vector of inductances to sweep over.
"""
function sweep_Lj(inds::Vector{Float64})
    # List inds and ask user to confirm
    println("Running simulations for inductances ($(length(inds))):")
    for ind in inds
        println("  $(ind) nH")
    end
    print("Confirm? (y/n) ")
    if lowercase(strip(readline())) != "y"
        return
    end

    test_id = max_index_in_dir(joinpath(OUTPUT_DIR, "Lj")) + 1
    for ind in inds
        output_path = joinpath(
            OUTPUT_DIR,
            "Lj",
            "#$(test_id)",
            "eigen_L$(ind)nH_$(Dates.format(now(), "yyyy-mm-ddTHHMMSS"))",
        )
        eigen_qubit(ind, 1, 0, 3.0; output_path = output_path)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    inds = collect(10.0:3.0:25.0)
    sweep_Lj(inds)
end
