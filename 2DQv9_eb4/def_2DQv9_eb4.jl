
# Constants
const CONFIG_DIR = joinpath(@__DIR__, "defaul_configs")
const OUTPUT_DIR = "/data/lesnow/2DQv9_eb4_data"
const GEO_PATH = joinpath(@__DIR__, "geo_files/2DQv9_eb4_edit.geo")
const RQ_RECT = Rectangle(40, 1160, -705, 105)
const QUBIT_RECT = Rectangle(625, 1160, -705, 105)
const WAVE_RECT = Rectangle(-2400, 2400, -2293.5, 2400)
const JUNCTIONS = Rectangle[Rectangle(-955, -890, 650, 660), Rectangle(860, 925, -280, -285)]
const EXCITAION_LUMPED_PORT = Rectangle(-10, 10, -2350, -2293.5)

