include("def_2DQv8_eb4.jl")

function max_index_in_dir(dir_path::AbstractString)
    max_index = -1
    if !isdir(dir_path)
        return max_index
    end

    for dir in readdir(dir_path)
        if isdir(joinpath(dir_path, dir))
            index = parse(Int, split(dir, "#")[end])
            max_index = max(max_index, index)
        end
    end

    return max_index
end

function eigen_qubit(refine::Real, order::Int, test_id::Int=0, num_eigens::Int=2, target_freq::Real=3.0, with_resonator::Bool=false)
    output_path = joinpath(OUTPUT_DIR, "RQ", "convergence_test_#$(test_index)", "eigen_r$(refine)o$(order)_$(Dates.format(now(), "yyyy-mm-ddTHHMMSS"))")
    ensure_path(output_path)

    # Generate mesh.
    mesh_config = basic_config(GEO_PATH, output_path, with_resonator ? RQ_RECT : QUBIT_RECT, 300.0; jjs=[JUNCTIONS[2]], refinement_level=refine)
    mesh_path = generate_mesh(mesh_config)

    # Update parameters.
    params = load_config(joinpath(CONFIG_DIR, "2DQv9_eb4_RQ_eigen.json"))
    params["Model"]["Mesh"] = mesh_path
    params["Problem"]["Output"] = output_path
    params["Solver"]["Order"] = order
    params["Solver"]["Eigenmode"]["N"] = num_eigens
    params["Solver"]["Eigenmode"]["Save"] = 0
    params["Solver"]["Eigenmode"]["Target"] = target_freq
    params["Boundaries"]["LumpedPort"][1]["L"] = 21.70551 * 1e-9
    save_config(params, joinpath(output_path, "palace-config.json"))

    # Run Palace.
    palace_run(joinpath(output_path, "palace-config.json"), 64, "--use-hwthread-cpus")
end

function eigen_resonator(refine::Real, order::Int, test_id::Int=0, num_eigens::Int=1, target_freq::Real=5.0)
    output_path = joinpath(OUTPUT_DIR, "resonator", "convergence_test_#$(test_id)", "eigen_r$(refine)o$(order)_$(Dates.format(now(), "yyyy-mm-ddTHHMMSS"))")
    ensure_path(output_path)

    # Generate mesh.
    mesh_config = basic_config(GEO_PATH, output_path, RESONATOR_RECT, 300.0; refinement_level=refine)
    mesh_path = generate_mesh(mesh_config)

    # Update parameters.
    params = load_config(joinpath(CONFIG_DIR, "2DQv9_eb4_resonator_eigen.json"))
    params["Model"]["Mesh"] = mesh_path
    params["Problem"]["Output"] = output_path
    params["Solver"]["Order"] = order
    params["Solver"]["Eigenmode"]["N"] = num_eigens
    params["Solver"]["Eigenmode"]["Save"] = 0
    params["Solver"]["Eigenmode"]["Target"] = target_freq
    save_config(params, joinpath(output_path, "palace-config.json"))

    # Run Palace.
    palace_run(joinpath(output_path, "palace-config.json"), 64, "--use-hwthread-cpus")
end

# test_id = max_index_in_dir(joinpath(OUTPUT_DIR, "RQ")) + 1
test_id = max_index_in_dir(joinpath(OUTPUT_DIR, "resonator")) + 1
# test_id = 1

for order in range(3, 6)
    for refine in range(1, 8 - order)
        # eigen_qubit(refine / 2 - 0.5, order, 2, 3.0, true)
        eigen_resonator(refine / 2 - 0.5, order, test_id)
    end
end
