include("../src/load_modules.jl")

# Constants
const CONFIG_DIR = joinpath(@__DIR__, "defaul_configs")
const OUTPUT_DIR = "/data/lesnow/2DQv9_eb4_data"
const GEO_PATH = joinpath(@__DIR__, "geo_files/2DQv9_eb4_edit.geo")
const RQ_RECT = Rectangle(40, 1160, -705, 105)

function electrostatic_RQ()
    output_path = joinpath(OUTPUT_DIR, "RQ/electrostatic_$(Dates.format(now(), "yyyy-mm-ddTHHMMSS"))")
    ensure_path(output_path)

    # Generate mesh.
    mesh_config = basic_config(GEO_PATH, output_path, RQ_RECT, 300.0)
    to_electrostatic!(mesh_config)
    mesh_path = generate_mesh(mesh_config)

    # Update parameters.
    params = load_config(joinpath(CONFIG_DIR, "2DQv9_eb4_RQ_electrostatic.json"))
    params["Model"]["Mesh"] = mesh_path
    params["Problem"]["Output"] = output_path
    save_config(params, joinpath(output_path, "palace-config.json"))

    # Run Palace.
    palace_run(joinpath(output_path, "palace-config.json"), 64, "--use-hwthread-cpus")
end
