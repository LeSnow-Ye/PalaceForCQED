include("../src/load_modules.jl")

# Constants
const CONFIG_DIR = joinpath(@__DIR__, "defaul_configs")
const OUTPUT_DIR = "/data/lesnow/2DQv9_eb4_data"
const GEO_PATH = joinpath(@__DIR__, "geo_files/2DQv9_eb4_edit.geo")
const RQ_RECT = Rectangle(40, 1160, -705, 105)
const QUBIT_RECT = Rectangle(625, 1160, -705, 105)
const WAVE_RECT = Rectangle(-2400, 2400, -2293.5, 2400)
const JUNCTIONS = Rectangle[Rectangle(-955, -890, 650, 660), Rectangle(860, 925, -280, -285)]
const EXCITAION_LUMPED_PORT = Rectangle(-10, 10, -2350, -2293.5)

function driven_lumped(MinFreq::Float64, MaxFreq::Float64, FreqStep::Float64=0.001, SaveStep::Integer=0)
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

function eigen_qubit(jj_inductance::Real, num_eigens::Int=2, save_eigens::Int=2, target_freq::Real=3.0)
    @assert num_eigens >= save_eigens "Number of eigenvalues to save must be less than or equal to the number of eigenvalues to compute."

    output_path = joinpath(OUTPUT_DIR, "qubit/eigen_L$(jj_inductance)nH_$(Dates.format(now(), "yyyy-mm-ddTHHMMSS"))")
    ensure_path(output_path)

    # Generate mesh.
    mesh_config = basic_config(GEO_PATH, output_path, QUBIT_RECT, 300.0; jjs=[JUNCTIONS[2]])
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
    palace_run(joinpath(output_path, "palace-config.json"), 64, "--use-hwthread-cpus")
end

# eigen_qubit(15.65, 1, 1, 4.0)
# eigen_qubit(17.42, 1, 1, 4.0)
driven_lumped(4.0, 4.9, 0.002)