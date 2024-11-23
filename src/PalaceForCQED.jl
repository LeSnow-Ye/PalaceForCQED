module PalaceForCQED

include("MeshGenerator.jl")
include("Utils.jl")
include("PalaceRunner.jl")

using .MeshGenerator
export GeneratorConfig,
    Rectangle,
    generate_mesh,
    basic_config,
    to_electrostatic!,
    LumpedPort,
    WavePort,
    CoaxWavePort,
    NoExcitation

using .PalaceRunner
export palace_run, load_config, save_config

using .Utils
export ensure_path, find_valleys, max_index_in_dir, get_closest_value


end
