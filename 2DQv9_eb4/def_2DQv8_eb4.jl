# Constants
const CONFIG_DIR = joinpath(@__DIR__, "defaul_configs")
const OUTPUT_DIR = "/data/lesnow/2DQv8_eb4_data"
const GEO_PATH = joinpath(@__DIR__, "geo_files/2DQv8_eb4_edit.geo")
const RQ_RECT = Rectangle(40, 1165, -705, 105)
const QUBIT_RECT = Rectangle(630, 1165, -705, 105)
const WAVE_RECT = Rectangle(-2400, 2400, -2380, 2400)
const JUNCTIONS = Rectangle[Rectangle(-965, -900, 650, 660), Rectangle(865, 930, -280, -285)]
const EXCITAION_LUMPED_PORT = Rectangle(-10, 10, -2295.25, -2380)
