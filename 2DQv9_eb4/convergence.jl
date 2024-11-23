include("run.jl")

function eigen_qubit_conv(
    refine::Real,
    order::Int,
    test_id::Int = 0,
    num_eigens::Int = 2,
    target_freq::Real = 3.0,
    with_resonator::Bool = false,
)
    output_path = joinpath(
        OUTPUT_DIR,
        with_resonator ? "RQ" : "qubit",
        "convergence_test_#$(test_id)",
        "eigen_r$(refine)o$(order)_$(Dates.format(now(), "yyyy-mm-ddTHHMMSS"))",
    )
    eigen_qubit(
        21.70551,
        num_eigens,
        0,
        target_freq;
        with_resonator = with_resonator,
        refine = refine,
        order = order,
        output_path = output_path,
    )
end

function eigen_resonator_conv(
    refine::Real,
    order::Int,
    test_id::Int = 0,
    num_eigens::Int = 1,
    target_freq::Real = 5.0,
)
    output_path = joinpath(
        OUTPUT_DIR,
        "resonator",
        "convergence_test_#$(test_id)",
        "eigen_r$(refine)o$(order)_$(Dates.format(now(), "yyyy-mm-ddTHHMMSS"))",
    )
    eigen_resonator(
        num_eigens,
        0,
        target_freq;
        refine = refine,
        order = order,
        output_path = output_path,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    test_id = max_index_in_dir(joinpath(OUTPUT_DIR, "RQ")) + 1
    # test_id = max_index_in_dir(joinpath(OUTPUT_DIR, "resonator")) + 1
    # test_id = 2


    for order in range(3, 6)
        for refine in range(1, 9 - order)
            eigen_qubit_conv(refine / 2 - 0.5, order, test_id, 2, 3.0, true)
            # eigen_resonator_conv(refine / 2 - 0.5, order, test_id)
        end
    end
end
