include("../src/load_modules.jl")

include("def_2DQv8_eb4.jl")
# include("def_2DQv9_eb4.jl")

function driven_lumped(MinFreq::Float64, MaxFreq::Float64, FreqStep::Float64=0.001, SaveStep::Integer=0, lj1::Real=-1, lj2::Real=-1)
    output_path = joinpath(OUTPUT_DIR, "driven/lumped_$(MinFreq)-$(MaxFreq)_Step$(FreqStep)_$(Dates.format(now(), "yyyy-mm-ddTHHMMSS"))")
    ensure_path(output_path)

    # Generate mesh.
    mesh_config = basic_config(GEO_PATH, output_path, Rectangle(), 200.0;
        excitation_type=MeshGenerator.LumpedPort,
        lumped_ports=Rectangle[EXCITAION_LUMPED_PORT],
        jjs=JUNCTIONS)
    mesh_path = generate_mesh(mesh_config)

    # Update parameters.
    params = load_config(joinpath(CONFIG_DIR, "2DQv9_eb4_driven_lumped.json"))
    params["Model"]["Mesh"] = mesh_path
    params["Problem"]["Output"] = output_path
    params["Solver"]["Driven"]["MinFreq"] = MinFreq
    params["Solver"]["Driven"]["MaxFreq"] = MaxFreq
    params["Solver"]["Driven"]["FreqStep"] = FreqStep
    params["Solver"]["Driven"]["SaveStep"] = SaveStep
    if lj1 > 0 && params["Boundaries"]["LumpedPort"][2]["Attributes"][1] == 2001
        params["Boundaries"]["LumpedPort"][2]["L"] = lj1 * 1e-9
    end

    if lj2 > 0 && params["Boundaries"]["LumpedPort"][3]["Attributes"][1] == 2002
        params["Boundaries"]["LumpedPort"][3]["L"] = lj2 * 1e-9
    end

    save_config(params, joinpath(output_path, "palace-config.json"))

    # Run Palace.
    palace_run(joinpath(output_path, "palace-config.json"), 64, "--use-hwthread-cpus")
end

function electrostatic_RQ()
    output_path = joinpath(OUTPUT_DIR, "RQ/electrostatic_$(Dates.format(now(), "yyyy-mm-ddTHHMMSS"))")
    ensure_path(output_path)

    # Generate mesh.
    mesh_config = basic_config(GEO_PATH, output_path, RQ_RECT, 300.0; split_metal_physical_group=true)
    mesh_path = generate_mesh(mesh_config)

    # Update parameters.
    params = load_config(joinpath(CONFIG_DIR, "2DQv9_eb4_RQ_electrostatic.json"))
    params["Model"]["Mesh"] = mesh_path
    params["Problem"]["Output"] = output_path
    save_config(params, joinpath(output_path, "palace-config.json"))

    # Run Palace.
    palace_run(joinpath(output_path, "palace-config.json"), 64, "--use-hwthread-cpus")
end

"""
    eigen_qubit(jj_inductance::Real, num_eigens::Int=2, save_eigens::Int=2, target_freq::Real=3.0)

Compute the eigenmodes of the qubit system with a given Josephson junction inductance.

# Arguments
- `jj_inductance::Real`: the Josephson junction inductance ``L_{j}`` (nH).
- `num_eigens::Int=2`: the number of eigenvalues to compute.
- `save_eigens::Int=2`: the number of eigenvalues to save.
- `target_freq::Real=3.0`: the target frequency of the eigenmodes to compute.
"""
function eigen_qubit(jj_inductance::Real, num_eigens::Int=2, save_eigens::Int=2, target_freq::Real=3.0, with_resonator::Bool=false)
    @assert num_eigens >= save_eigens "Number of eigenvalues to save must be less than or equal to the number of eigenvalues to compute."

    output_path = joinpath(OUTPUT_DIR, with_resonator ? "RQ" : "qubit", "eigen_L$(jj_inductance)nH_$(Dates.format(now(), "yyyy-mm-ddTHHMMSS"))")

    ensure_path(output_path)

    # Generate mesh.
    mesh_config = basic_config(GEO_PATH, output_path, with_resonator ? RQ_RECT : QUBIT_RECT, 300.0; jjs=[JUNCTIONS[2]])
    mesh_path = generate_mesh(mesh_config)

    # Update parameters.
    params = load_config(joinpath(CONFIG_DIR, "2DQv9_eb4_RQ_eigen.json"))
    params["Model"]["Mesh"] = mesh_path
    params["Problem"]["Output"] = output_path
    params["Solver"]["Eigenmode"]["N"] = num_eigens
    params["Solver"]["Eigenmode"]["Save"] = save_eigens
    params["Solver"]["Eigenmode"]["Target"] = target_freq
    params["Boundaries"]["LumpedPort"][1]["L"] = jj_inductance * 1e-9
    save_config(params, joinpath(output_path, "palace-config.json"))

    # Run Palace.
    # palace_run(joinpath(output_path, "palace-config.json"), 64, "--use-hwthread-cpus")
end


# eigen_qubit(16.710, 1, 1, 3.0)
eigen_qubit(21.70551, 3, 3, 3.0, true)
# driven_lumped(3.5, 4.8, 0.0005)
# driven_lumped(4.35, 4.45, 0.0001,0, 16.78543, 21.70551)
# driven_lumped(3.83, 3.93, 0.0001,0, 16.78543, 21.70551)
# driven_lumped(6.7, 7.7, 0.001, 0, 16.78543, 21.70551)