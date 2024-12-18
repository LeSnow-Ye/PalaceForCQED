using PalaceForCQED
using Dates

include("def_2DQv8_eb4.jl")
# include("def_2DQv9_eb4.jl")

function driven_lumped(
    MinFreq::Float64,
    MaxFreq::Float64,
    FreqStep::Float64 = 0.001,
    SaveStep::Integer = 0,
    lj1::Real = -1,
    lj2::Real = -1;
    refine::Real = DEFAULT_REFINEMENT,
    order::Int = DEFAULT_ORDER,
    output_path::AbstractString = "",
)
    default_path = joinpath(
        OUTPUT_DIR,
        "driven/lumped_$(MinFreq)-$(MaxFreq)_Step$(FreqStep)_$(Dates.format(now(), "yyyy-mm-ddTHHMMSS"))",
    )
    output_path = output_path == "" ? default_path : output_path
    ensure_path(output_path)

    # Generate mesh.
    mesh_config = basic_config(
        GEO_PATH,
        output_path,
        Rectangle(),
        200.0;
        excitation_type = LumpedPort,
        lumped_ports = Rectangle[EXCITAION_LUMPED_PORT],
        jjs = JUNCTIONS,
        refinement_level = refine,
    )
    mesh_path = generate_mesh(mesh_config)

    # Update parameters.
    params = load_config(joinpath(CONFIG_DIR, "2DQv9_eb4_driven_lumped.json"))
    params["Model"]["Mesh"] = mesh_path
    params["Problem"]["Output"] = output_path
    params["Solver"]["Order"] = order
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
    palace_run(
        joinpath(output_path, "palace-config.json"),
        NUM_THREADS,
        "--use-hwthread-cpus",
    )
end

function electrostatic_RQ()
    output_path =
        joinpath(OUTPUT_DIR, "RQ/electrostatic_$(Dates.format(now(), "yyyy-mm-ddTHHMMSS"))")
    ensure_path(output_path)

    # Generate mesh.
    mesh_config = basic_config(
        GEO_PATH,
        output_path,
        RQ_RECT,
        300.0;
        split_metal_physical_group = true,
    )
    mesh_path = generate_mesh(mesh_config)

    # Update parameters.
    params = load_config(joinpath(CONFIG_DIR, "2DQv9_eb4_RQ_electrostatic.json"))
    params["Model"]["Mesh"] = mesh_path
    params["Problem"]["Output"] = output_path
    save_config(params, joinpath(output_path, "palace-config.json"))

    # Run Palace.
    palace_run(
        joinpath(output_path, "palace-config.json"),
        NUM_THREADS,
        "--use-hwthread-cpus",
    )
end

"""
	function eigen_qubit(
		jj_inductance::Real,
		num_eigens::Int=2,
		save_eigens::Int=2,
		target_freq::Real=3.0;
		with_resonator::Bool=false,
		refine::Real=DEFAULT_REFINEMENT,
		order::Int=DEFAULT_ORDER,
		output_path::AbstractString=""
	)

Compute the eigenmodes of the qubit system with a given Josephson junction inductance.

Note that simulation time can be reduced a lot by unsetting the Absorbing Boundary, 
while not affecting the accuracy of the eigenmodes too much.

# Arguments
- `jj_inductance::Real`: the Josephson junction inductance ``L_{j}`` (nH).
- `num_eigens::Int=2`: the number of eigenvalues to compute.
- `save_eigens::Int=2`: the number of eigenvalues to save.
- `target_freq::Real=3.0`: the target frequency of the eigenmodes to compute.
- `with_resonator::Bool=false`: whether to include the readout resonator.
- `refinement::Int=DEFAULT_REFINEMENT`: the refinement level of the mesh.
- `order::Int=DEFAULT_ORDER`: the order of the finite element method.
- `out_name::AbstractString=""`: the name of the output directory.
"""
function eigen_qubit(
    jj_inductance::Real,
    num_eigens::Int = 2,
    save_eigens::Int = 2,
    target_freq::Real = 3.0;
    with_resonator::Bool = false,
    refine::Real = DEFAULT_REFINEMENT,
    order::Int = DEFAULT_ORDER,
    output_path::AbstractString = "",
)
    @assert num_eigens >= save_eigens "Number of eigenvalues to save must be less than or equal to the number of eigenvalues to compute."

    default_path = joinpath(
        OUTPUT_DIR,
        with_resonator ? "RQ" : "qubit",
        "eigen_L$(jj_inductance)nH_$(Dates.format(now(), "yyyy-mm-ddTHHMMSS"))",
    )
    output_path = output_path == "" ? default_path : output_path
    ensure_path(output_path)

    # Generate mesh.
    mesh_config = basic_config(
        GEO_PATH,
        output_path,
        with_resonator ? RQ_RECT : QUBIT_RECT,
        300.0;
        jjs = [JUNCTIONS[2]],
        refinement_level = refine,
    )
    mesh_path = generate_mesh(mesh_config)

    # Update parameters.
    params = load_config(joinpath(CONFIG_DIR, "2DQv9_eb4_RQ_eigen.json"))
    params["Model"]["Mesh"] = mesh_path
    params["Problem"]["Output"] = output_path
    params["Solver"]["Order"] = order
    params["Solver"]["Eigenmode"]["N"] = num_eigens
    params["Solver"]["Eigenmode"]["Save"] = save_eigens
    params["Solver"]["Eigenmode"]["Target"] = target_freq
    params["Boundaries"]["LumpedPort"][1]["L"] = jj_inductance * 1e-9
    save_config(params, joinpath(output_path, "palace-config.json"))

    # Run Palace.
    palace_run(
        joinpath(output_path, "palace-config.json"),
        NUM_THREADS,
        "--use-hwthread-cpus",
    )
end

function eigen_resonator(
    num_eigens::Int = 1,
    save_eigens::Int = 1,
    target_freq::Real = 3.0;
    refine::Real = DEFAULT_REFINEMENT,
    order::Int = DEFAULT_ORDER,
    output_path::AbstractString = "",
)
    @assert num_eigens >= save_eigens "Number of eigenvalues to save must be less than or equal to the number of eigenvalues to compute."

    default_path = joinpath(
        OUTPUT_DIR,
        "resonator",
        "eigen_$(Dates.format(now(), "yyyy-mm-ddTHHMMSS"))",
    )
    output_path = output_path == "" ? default_path : output_path
    ensure_path(output_path)

    # Generate mesh.
    mesh_config = basic_config(
        GEO_PATH,
        output_path,
        RESONATOR_RECT,
        300.0;
        refinement_level = refine,
    )
    mesh_path = generate_mesh(mesh_config)

    # Update parameters.
    params = load_config(joinpath(CONFIG_DIR, "2DQv9_eb4_resonator_eigen.json"))
    params["Model"]["Mesh"] = mesh_path
    params["Problem"]["Output"] = output_path
    params["Solver"]["Order"] = order
    params["Solver"]["Eigenmode"]["N"] = num_eigens
    params["Solver"]["Eigenmode"]["Save"] = save_eigens
    params["Solver"]["Eigenmode"]["Target"] = target_freq
    save_config(params, joinpath(output_path, "palace-config.json"))

    # Run Palace.
    palace_run(
        joinpath(output_path, "palace-config.json"),
        NUM_THREADS,
        "--use-hwthread-cpus",
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    # EXAMPLE USAGE
    eigen_resonator(1, 1, 5.0)
    eigen_qubit(21.70551, 2, 2, 3.0; with_resonator = true)

    driven_lumped(7.0, 7.8, 0.001, 0, 16.78543, 21.70551)
    driven_lumped(4.35, 4.45, 0.0001, 0, 16.78543, 21.70551)
    driven_lumped(3.83, 3.93, 0.0001, 0, 16.78543, 21.70551)
end
