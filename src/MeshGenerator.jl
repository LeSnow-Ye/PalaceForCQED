module MeshGenerator

include("GmshUtils.jl")
using .GmshUtils

using Gmsh: gmsh

export GeneratorConfig, Rectangle, ExcitationType, generate_mesh, basic_config, to_electrostatic!

struct Rectangle
    x_min::Float64
    x_max::Float64
    y_min::Float64
    y_max::Float64

    Rectangle(x_min=0.0, x_max=0.0, y_min=0.0, y_max=0.0) = new(x_min, x_max, y_min, y_max)
end

function included_in_rectangle(obj::Rectangle, rect::Rectangle)
    return (obj.x_min >= rect.x_min && obj.x_max <= rect.x_max && obj.y_min >= rect.y_min && obj.y_max <= rect.y_max)
end

@enum ExcitationType begin
    LumpedPort
    WavePort
    CoaxWavePort
    NoExcitation
end

@kwdef mutable struct GeneratorConfig
    # General
    geo_file_path::AbstractString
    output_dir::AbstractString
    mesh_name::AbstractString = "mesh"
    refinement_level::Real = 1.5
    mesh_order::Integer = 1
    gui::Bool = false
    gmsh_threads::Integer = 1
    verbose::Integer = 5

    # Cutting
    gaps_area::Rectangle = Rectangle() # Set to Rectangle() to automatically detect gaps.
    area_expanded_from_gaps::Real = 0.0

    # Ports
    excitation_type::ExcitationType
    lumped_ports::Vector{Rectangle} = []
    jjs::Vector{Rectangle} = [] # They are also lumped ports. Renaming for convenience of readability.

    # Geometry 
    air_domain_height_μm::Real = 1500.0 # Above metal layer
    trace_width_μm::Real = 20.0
    substrate_height_μm::Real = 525.0
    metal_height_μm::Real = 0.0
    remove_metal_vol::Bool = false

    # Others
    split_metal_physical_group::Bool = false

end

function basic_config(
    geo_file_path::AbstractString,
    output_dir::AbstractString,
    gaps_area::Rectangle,
    area_expanded_from_gaps::Real;
    kwargs...
)
    return GeneratorConfig(
        geo_file_path=geo_file_path,
        output_dir=output_dir,
        gaps_area=gaps_area,
        area_expanded_from_gaps=area_expanded_from_gaps,
        excitation_type=NoExcitation;
        kwargs...
    )
end

function check_config(config::GeneratorConfig)
    @assert config.air_domain_height_μm > config.substrate_height_μm
    @assert config.gmsh_threads == 1 # IDK why, but when `num_threads` is greater than 0, there is a segmentation fault.

    @assert config.trace_width_μm > 0.0
    @assert config.substrate_height_μm > 0.0
    @assert config.metal_height_μm >= 0.0
end

function gmsh_add_rectangle(rect::Rectangle)
    return gmsh.model.occ.add_rectangle(rect.x_min, rect.y_min, 0.0, rect.x_max - rect.x_min, rect.y_max - rect.y_min)
end

function preprocess_geo_file!(config::GeneratorConfig)
    new_geo_file_path = joinpath(config.output_dir, "uniformized.geo")
    uniformize(config.geo_file_path, new_geo_file_path; min_length=2 * 20.0 * (2.0^-config.refinement_level))
    config.geo_file_path = new_geo_file_path
end

function generate_mesh(config::GeneratorConfig)
    check_config(config)
    preprocess_geo_file!(config)

    # Load .geo
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", config.verbose)
    gmsh.option.setNumber("General.NumThreads", config.gmsh_threads)
    gmsh.clear()
    gmsh.parser.parse(config.geo_file_path)
    gmsh.model.occ.synchronize()  # Remember to add `SetFactory("OpenCASCADE");` to the .geo file

    # Set metal area
    x_min, y_min, x_max, y_max = 0.0, 0.0, 0.0, 0.0
    gaps_raw_dimtags = gmsh.model.occ.get_entities(2)
    auto_detect = (config.gaps_area === Rectangle())
    if auto_detect
        for dim_tag in gaps_raw_dimtags
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.get_bounding_box(dim_tag[1], dim_tag[2])
            x_min = round(min(x_min, xmin), digits=3)
            y_min = round(min(y_min, ymin), digits=3)
            x_max = round(max(x_max, xmax), digits=3)
            y_max = round(max(y_max, ymax), digits=3)
            @assert round(zmin, digits=3) == round(zmax, digits=3) == 0.0
        end
    else
        x_min, y_min, x_max, y_max = config.gaps_area.x_min, config.gaps_area.y_min, config.gaps_area.x_max, config.gaps_area.y_max
    end

    dx = x_max - x_min
    dy = y_max - y_min

    sep_dz = config.air_domain_height_μm
    sep_dx = 0.5 * sep_dz
    sep_dy = 0.5 * sep_dz

    # Mesh parameters
    l_trace = 1.5 * config.trace_width_μm * (2.0^-config.refinement_level)
    l_farfield = 3 * config.substrate_height_μm * (2.0^-(config.refinement_level * 0.25))

    # Cut
    if !auto_detect
        gap_area = gmsh.model.occ.add_rectangle(x_min, y_min, 0.0, dx, dy)
        gmsh.model.occ.synchronize()
        gaps_raw_dimtags, _ = gmsh.model.occ.intersect(gaps_raw_dimtags, (2, gap_area))
    end

    x_min -= config.area_expanded_from_gaps
    y_min -= config.area_expanded_from_gaps
    x_max += config.area_expanded_from_gaps
    y_max += config.area_expanded_from_gaps
    metal_rect = Rectangle(x_min, x_max, y_min, y_max)

    println("Meshing area: ($x_min, $y_min) -> ($x_max, $y_max)")

    dx = x_max - x_min
    dy = y_max - y_min

    metal_raw_id = gmsh.model.occ.add_rectangle(x_min, y_min, 0.0, dx, dy)
    gmsh.model.occ.synchronize()
    metal_dimtags, _ =
        gmsh.model.occ.cut((2, metal_raw_id), gaps_raw_dimtags, -1, false, true)
    gaps_dimtags, _ = gmsh.model.occ.cut((2, metal_raw_id), metal_dimtags, -1, true, false)

    # Metal thickness
    metal_boundary = [x[2] for x in metal_dimtags if x[1] == 2]
    metal_boundary_top = typeof(metal_boundary)(undef, 0)
    metal = typeof(metal_boundary)(undef, 0)
    if config.metal_height_μm > 0
        metal_dimtags =
            gmsh.model.occ.extrude([(2, x) for x in metal_boundary], 0.0, 0.0, config.metal_height_μm)
        metal = [x[2] for x in filter(x -> x[1] == 3, metal_dimtags)]
        for domain in metal
            _, boundary = gmsh.model.occ.getSurfaceLoops(domain)
            @assert length(boundary) == 1
            append!(metal_boundary_top, first(boundary))
        end
        filter!(x -> !(x in metal_boundary), metal_boundary_top)
    end

    # Substrate
    substrate = gmsh.model.occ.addBox(x_min, y_min, -config.substrate_height_μm, dx, dy, config.substrate_height_μm)

    # Exterior box
    add_wave_port = config.excitation_type == WavePort || config.excitation_type == CoaxWavePort
    if add_wave_port
        # TODO: now still need to add the port manually
        domain = gmsh.model.occ.addBox(
            x_min - sep_dx,
            y_min,
            -sep_dz,
            dx + 2.0 * sep_dx,
            dy + sep_dy,
            2.0 * sep_dz
        )
    else
        domain = gmsh.model.occ.addBox(
            x_min - sep_dx,
            y_min - sep_dy,
            -sep_dz,
            dx + 2.0 * sep_dx,
            dy + 2.0 * sep_dy,
            2.0 * sep_dz
        )
    end

    _, domain_boundary = gmsh.model.occ.getSurfaceLoops(domain)
    @assert length(domain_boundary) == 1
    domain_boundary = first(domain_boundary)

    # Junctions
    junctions_ids = []
    for j in config.jjs
        if included_in_rectangle(j, metal_rect)
            push!(junctions_ids, gmsh_add_rectangle(j))
        end
    end

    # Lumped ports
    lumped_ports_ids = []
    for l in config.lumped_ports
        if included_in_rectangle(l, metal_rect)
            push!(lumped_ports_ids, gmsh_add_rectangle(l))
        end
    end

    # Ports
    if config.excitation_type == CoaxWavePort
        ra = 125 * 0.5
        rb = 125 * 0.5 + 56.5
        let da, db
            da = gmsh.model.occ.addDisk(0.0, y_min, 0.0, ra, ra, -1, [0, 1, 0], [])
            db = gmsh.model.occ.addDisk(0.0, y_min, 0.0, rb, rb, -1, [0, 1, 0], [])
            global p1 = last(first(first(gmsh.model.occ.cut((2, db), (2, da)))))
        end
    end
    if config.excitation_type == WavePort
        dxp1 = 160
        dxp2 = dxp1
        dzp1 = 0.5 * (sep_dz + config.substrate_height_μm)
        dzp2 = config.substrate_height_μm
        let pa, pb, l
            pa = gmsh.model.occ.addPoint(-dxp2, y_min, -dzp1)
            pb = gmsh.model.occ.addPoint(dxp1, y_min, -dzp1)
            l = gmsh.model.occ.addLine(pa, pb)
            global p1 = first(
                filter(x -> x[1] == 2, gmsh.model.occ.extrude([1, l], 0.0, 0.0, dzp1 + dzp2))
            )[2]
        end
    end

    if add_wave_port
        let pa, pb, l
            pa = gmsh.model.occ.addPoint(x_min - sep_dx, y_min, -sep_dz)
            pb = gmsh.model.occ.addPoint(x_max + sep_dx, y_min, -sep_dz)
            l = gmsh.model.occ.addLine(pa, pb)
            global p2 =
                first(filter(x -> x[1] == 2, gmsh.model.occ.extrude([1, l], 0.0, 0.0, 2.0 * sep_dz)))[2]
        end
    end

    # Embedding
    geom_dimtags = filter(x -> x[1] == 2 || x[1] == 3, gmsh.model.occ.getEntities())
    _, geom_map = gmsh.model.occ.fragment(geom_dimtags, [])
    gmsh.model.occ.synchronize()

    # Add physical groups
    basic_group_tag = 1000
    additional_group_tag = 2000
    port_group_tag = 3000

    metal_domains =
        last.(
            collect(
                Iterators.flatten(
                    geom_map[findall(x -> x[1] == 3 && x[2] in metal, geom_dimtags)]
                )
            )
        )

    si_domain = last.(geom_map[findfirst(x -> x == (3, substrate), geom_dimtags)])
    @assert length(si_domain) == 1
    si_domain = first(si_domain)

    air_domain = last.(geom_map[findfirst(x -> x == (3, domain), geom_dimtags)])
    filter!(x -> !(x == si_domain || x in metal_domains), air_domain)
    @assert length(air_domain) == 1
    air_domain = first(air_domain)

    if length(metal_domains) > 0 && config.remove_metal_vol
        remove_dimtags = [(3, x) for x in metal_domains]
        for tag in
            last.(
            filter(
                x -> x[1] == 2,
                gmsh.model.getBoundary(
                    [(3, z) for z in metal_domains],
                    false,
                    false,
                    false
                )
            )
        )
            normal = gmsh.model.getNormal(tag, [0, 0])
            if abs(normal[1]) == 1.0
                push!(remove_dimtags, (2, tag))
            end
        end
        gmsh.model.occ.remove(remove_dimtags)
        gmsh.model.occ.synchronize()
        filter!.(x -> !(x in remove_dimtags), geom_map)
        empty!(metal_domains)
    end

    air_domain_group = gmsh.model.addPhysicalGroup(3, [air_domain], (basic_group_tag += 1), "air")
    si_domain_group = gmsh.model.addPhysicalGroup(3, [si_domain], (basic_group_tag += 1), "si")
    metal_domain_group = gmsh.model.addPhysicalGroup(3, metal_domains, (basic_group_tag += 1), "metal")

    if add_wave_port
        port1 = last.(geom_map[findfirst(x -> x == (2, p1), geom_dimtags)])
        port1_group = gmsh.model.addPhysicalGroup(2, port1, (port_group_tag += 1), "port1")

        end1 = last.(geom_map[findfirst(x -> x == (2, p2), geom_dimtags)])
        filter!(x -> !(x in port1), end1)

        end1_group = gmsh.model.addPhysicalGroup(2, end1, (port_group_tag += 1), "end1")
    end

    farfield =
        last.(
            collect(
                Iterators.flatten(
                    geom_map[findall(
                        x -> x[1] == 2 && x[2] in domain_boundary,
                        geom_dimtags
                    )]
                )
            )
        )

    if add_wave_port
        filter!(x -> !(x in port1 || x in end1), farfield)
    end

    farfield_group = gmsh.model.addPhysicalGroup(2, farfield, (basic_group_tag += 1), "farfield")

    junction_dimtags = []
    for (i, junction_id) in enumerate(junctions_ids)
        j = last.(geom_map[findfirst(x -> x == (2, junction_id), geom_dimtags)])
        junction_dimtags = vcat(junction_dimtags, j)
        gmsh.model.addPhysicalGroup(2, j, (additional_group_tag += 1), "j$i")
    end

    lumped_port_dimtags = []
    for (i, lumped_port_id) in enumerate(lumped_ports_ids)
        l = last.(geom_map[findfirst(x -> x == (2, lumped_port_id), geom_dimtags)])
        lumped_port_dimtags = vcat(lumped_port_dimtags, l)
        gmsh.model.addPhysicalGroup(2, l, (port_group_tag += 1), "lp$i")
    end

    trace =
        last.(
            collect(
                Iterators.flatten(
                    geom_map[findall(
                        x -> x[1] == 2 && x[2] in metal_boundary,
                        geom_dimtags
                    )]
                )
            )
        )
    gap =
        last.(
            collect(
                Iterators.flatten(
                    geom_map[findall(x -> x in gaps_dimtags, geom_dimtags)]
                )
            )
        )

    setdiff!(gap, junction_dimtags, lumped_port_dimtags)

    if !config.split_metal_physical_group
        trace_group = gmsh.model.addPhysicalGroup(2, trace, (basic_group_tag += 1), "metal_layer")
    else
        for (i, metal_layer) in enumerate(trace)
            group_tag = (basic_group_tag += 1)
            group_name = "metal_layer_$i"
            gmsh.model.addPhysicalGroup(2, [metal_layer], group_tag, group_name)
        end
    end

    gap_group = gmsh.model.addPhysicalGroup(2, gap, (basic_group_tag += 1), "gap")

    trace_top =
        last.(
            collect(
                Iterators.flatten(
                    geom_map[findall(
                        x -> x[1] == 2 && x[2] in metal_boundary_top,
                        geom_dimtags
                    )]
                )
            )
        )

    trace_top_group = gmsh.model.addPhysicalGroup(2, trace_top, (basic_group_tag += 1), "trace_top")

    # Generate mesh
    gmsh.option.setNumber("Mesh.MeshSizeMin", l_trace)
    gmsh.option.setNumber("Mesh.MeshSizeMax", l_farfield)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    gap_points =
        last.(
            filter(
                x -> x[1] == 0,
                gmsh.model.getBoundary([(2, z) for z in gap], false, true, true)
            )
        )
    gap_curves =
        last.(
            filter(
                x -> x[1] == 1,
                gmsh.model.getBoundary([(2, z) for z in gap], false, false, false)
            )
        )

    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "PointsList", gap_points)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", gap_curves)
    gmsh.model.mesh.field.setNumbers(1, "SurfacesList", gap)
    gmsh.model.mesh.field.setNumber(1, "Sampling", ceil(max(dx, dy) / l_trace))

    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", l_trace)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", l_farfield)
    gmsh.model.mesh.field.setNumber(2, "DistMin", config.trace_width_μm)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 0.9 * sep_dz)

    gmsh.model.mesh.field.add("Min", 101)
    gmsh.model.mesh.field.setNumbers(101, "FieldsList", [2])
    gmsh.model.mesh.field.setAsBackgroundMesh(101)

    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)
    for tag in Iterators.flatten((gap, trace, trace_top))
        gmsh.model.mesh.setAlgorithm(2, tag, 8)
    end

    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(config.mesh_order)

    # Save mesh
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Binary", 0)

    full_path = ""
    if config.output_dir == ""
        full_path = joinpath(dirname(@__DIR__), "tmp", "$(config.mesh_name).msh")
    else
        full_path = joinpath(config.output_dir, "$(config.mesh_name).msh")
    end

    gmsh.write(full_path)

    # Optionally launch GUI
    if config.gui
        gmsh.fltk.run()
    end

    gmsh.finalize()
    return full_path
end

end