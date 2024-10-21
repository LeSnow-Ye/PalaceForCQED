using Test
using PalaceForCQED.Utils

@testset "Utils.jl" begin
    # Happy Path
    begin
        dir_path = "tmp/test_dir"
        mkpath(dir_path)
        mkpath(joinpath(dir_path, "xxx_#0"))
        mkpath(joinpath(dir_path, "xxx_#2"))
        mkpath(joinpath(dir_path, "xxx_#3"))
        @test max_index_in_dir(dir_path) == 3
    end

    # Edge Case 1: No directories present
    begin
        dir_path = "tmp/empty_dir"
        mkpath(dir_path)
        @test max_index_in_dir(dir_path) == -1
    end

    # Edge Case 2: Non-existing directory
    @test max_index_in_dir("tmp/non_existing_dir") == -1

    # Edge Case 3: Minimum input with single directory
    begin
        dir_path = "tmp/single_dir"
        mkpath(dir_path)
        mkpath(joinpath(dir_path, "xxx_#5"))
        @test max_index_in_dir(dir_path) == 5
    end

    # Edge Case 4: Invalid directory name format
    begin
        dir_path = "tmp/invalid_dir"
        mkpath(dir_path)
        mkpath(joinpath(dir_path, "xxx_#1"))
        mkpath(joinpath(dir_path, "directory1"))
        mkpath(joinpath(dir_path, "directory2"))
        mkpath(joinpath(dir_path, "xxx_no_number"))
        @test max_index_in_dir(dir_path) == 1
    end

    rm("tmp", force=true, recursive=true)
end
