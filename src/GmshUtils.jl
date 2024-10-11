#! WARNING: Current version of GmshUtils.jl is not GENERALLY functional and POORLY impledented.

module GmshUtils

export uniformize

struct Point
    id::Int
    x::Float64
    y::Float64
    z::Float64
end

mutable struct Line
    id::Int
    point_ids::Vector{Int}
end

mutable struct LineLoop
    id::Int
    line_ids::Vector{Int}
end

function length_between_points(p1::Point, p2::Point)
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    return sqrt(dx^2 + dy^2)
end

function geo_string(p::Point)
    return "Point($(p.id))={$(p.x),$(p.y),$(p.z)};\n"
end

function geo_string(l::Line)
    return "Line($(l.id))={$(join(l.point_ids, ","))};\n"
end

function geo_string(ll::LineLoop)
    return "Line Loop($(ll.id))={$(join(ll.line_ids, ","))};\n"
end

function vector(p1::Point, p2::Point)
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    return (dx, dy)
end

function line_length(l::Line)
    p1 = points[findfirst(p -> p.id == l.point_ids[1], points)]
    p2 = points[findfirst(p -> p.id == l.point_ids[2], points)]
    dx, dy = vector(p1, p2)
    return sqrt(dx^2 + dy^2)
end

function angle(vector)
    atan(vector[2], vector[1])
end

function is_vertical_or_horizontal(line::Line, points::Vector{Point})
    p1 = points[findfirst(p -> p.id == line.point_ids[1], points)]
    p2 = points[findfirst(p -> p.id == line.point_ids[2], points)]
    return (p1.x == p2.x) || (p1.y == p2.y)
end

function angle_difference(line1::Line, line2::Line, points::Vector{Point})
    p1_start = points[findfirst(p -> p.id == line1.point_ids[1], points)]
    p1_end = points[findfirst(p -> p.id == line1.point_ids[2], points)]
    p2_start = points[findfirst(p -> p.id == line2.point_ids[1], points)]
    p2_end = points[findfirst(p -> p.id == line2.point_ids[2], points)]

    vector1 = vector(p1_start, p1_end)
    vector2 = vector(p2_start, p2_end)
    angle_diff = angle(vector1) - angle(vector2)
    angle_diff = mod(angle_diff + pi, 2 * pi) - pi

    # println("angle1 = $(rad2deg(angle1)), angle2 = $(rad2deg(angle2)), angle_diff = $(rad2deg(angle_diff))")
    return abs(angle_diff)
end

function distance_between_lines(line1::Line, line2::Line, points::Vector{Point})
    p1_start = points[findfirst(p -> p.id == line1.point_ids[1], points)]
    p1_end = points[findfirst(p -> p.id == line1.point_ids[2], points)]
    p1 = (p1_start.x + p1_end.x) / 2, (p1_start.y + p1_end.y) / 2

    p2_start = points[findfirst(p -> p.id == line2.point_ids[1], points)]
    p2_end = points[findfirst(p -> p.id == line2.point_ids[2], points)]
    p2 = (p2_start.x + p2_end.x) / 2, (p2_start.y + p2_end.y) / 2

    distance = sqrt((p1[1] - p2[1])^2 + (p1[2] - p2[2])^2)
    return distance
end

function trim(str::String)
    return replace(str, " " => "")
end

function parse_point(str::String)
    m = match(r"Point\((\d+)\)\=\{([+-]?\d+\.\d+e[+-]?\d+),([+-]?\d+\.\d+e[+-]?\d+),([+-]?\d+\.\d+e[+-]?\d+)\}", trim(str))
    if m === nothing
        error("Invalid string format: $str")
    end

    id = parse(Int, m.captures[1])
    x = parse(Float64, m.captures[2])
    y = parse(Float64, m.captures[3])
    z = parse(Float64, m.captures[4])

    return Point(id, x, y, z)
end

function parse_line(str::String)
    m = match(r"Line\((\d+)\)=\{([\d,]+)\};", trim(str))
    if m === nothing
        error("Invalid string format: $str")
    end

    line_id_str, point_ids_str = m.captures

    line_id = parse(Int, line_id_str)
    point_ids = parse.(Int, split(point_ids_str, ","))

    return Line(line_id, point_ids)
end

function parse_line_loop(str::String)
    m = match(r"LineLoop\((\d+)\)=\{([\d,]+)\};", trim(str))
    if m === nothing
        error("Invalid string format: $str")
    end

    line_loop_id_str, line_ids_str = m.captures

    line_loop_id = parse(Int, line_loop_id_str)
    line_ids = parse.(Int, split(line_ids_str, ","))

    return LineLoop(line_loop_id, line_ids)
end

function remove_lines_until_threshold!(line_loop::LineLoop, lines::Vector{Line}, points::Vector{Point}, threshold_angle::Float64, min_length::Float64)
    need_to_iterate = true
    i = 1
    new_line_ids = Int[]
    push!(new_line_ids, line_loop.line_ids[i])

    while need_to_iterate
        line1 = lines[findfirst(l -> l.id == line_loop.line_ids[i], lines)]
        for j in i+1:length(line_loop.line_ids)
            line2 = lines[findfirst(l -> l.id == line_loop.line_ids[j], lines)]

            if !is_vertical_or_horizontal(line2, points) && (angle_difference(line1, line2, points) < threshold_angle && distance_between_lines(line1, line2, points) < min_length)
                if j == length(line_loop.line_ids) # Last line in the loop
                    push!(new_line_ids, line_loop.line_ids[j])
                    need_to_iterate = false
                end
                continue # Skip this line
            end

            push!(new_line_ids, line_loop.line_ids[j])
            i = j
            if i == length(line_loop.line_ids)
                need_to_iterate = false
            end
            break
        end
    end

    line_loop.line_ids = new_line_ids
    # println(line_loop.line_ids)
end

function enclose_line_loop!(line_loop::LineLoop, lines::Vector{Line})
    for i in 1:length(line_loop.line_ids)
        line1 = lines[findfirst(l -> l.id == line_loop.line_ids[i], lines)]
        j = i == length(line_loop.line_ids) ? 1 : i + 1
        line2 = lines[findfirst(l -> l.id == line_loop.line_ids[j], lines)]
        line1.point_ids = [line1.point_ids[1], line2.point_ids[1]]
    end
end

function get_unique_points_from_line_loop(line_loop::LineLoop, lines::Vector{Line}, points::Vector{Point})
    point_id_set = Set{Int}()
    for line_id in line_loop.line_ids
        line = lines[findfirst(l -> l.id == line_id, lines)]
        for point_id in line.point_ids
            push!(point_id_set, point_id)
        end
    end

    loop_points::Vector{Point} = []
    for point_id in point_id_set
        push!(loop_points, points[findfirst(p -> p.id == point_id, points)])
    end


    return loop_points
end

function uniformized_line_points(buffer_line_points::Vector{Point}, target_segment_length::Float64)
    total_length::Float64 = 0.0
    for i in 1:length(buffer_line_points)-1
        total_length += length_between_points(buffer_line_points[i], buffer_line_points[i+1])
    end

    l = total_length / ceil(total_length / target_segment_length)

    new_line_points::Vector{Point} = []
    push!(new_line_points, buffer_line_points[1])

    buf = l
    for i in 2:length(buffer_line_points)
        p1 = buffer_line_points[i-1]
        p2 = buffer_line_points[i]
        segment_length = length_between_points(p1, p2)

        while buf < segment_length && !(i == length(buffer_line_points) && ((segment_length - buf) < 0.2 * l))
            x = p1.x + (p2.x - p1.x) * buf / segment_length
            y = p1.y + (p2.y - p1.y) * buf / segment_length
            z = p1.z + (p2.z - p1.z) * buf / segment_length
            push!(new_line_points, Point(new_line_points[end].id + 1, x, y, z))
            buf += l
        end

        buf -= segment_length
    end

    return new_line_points
end

function uniformized_line_loop(line_loop::LineLoop, lines::Vector{Line}, points::Vector{Point}, min_length::Float64)
    new_line_loop_points::Vector{Point} = []
    buffer_line_points::Vector{Point} = []
    for line_id = line_loop.line_ids
        line = lines[findfirst(l -> l.id == line_id, lines)]

        @assert length(line.point_ids) == 2 # Only support lines with two points

        if is_vertical_or_horizontal(line, points)
            if length(buffer_line_points) > 0
                new_line_loop_points = [new_line_loop_points; uniformized_line_points(buffer_line_points, min_length)]
                buffer_line_points = []
            end

            push!(new_line_loop_points, points[findfirst(p -> p.id == line.point_ids[1], points)])
        else
            push!(buffer_line_points, points[findfirst(p -> p.id == line.point_ids[1], points)])
        end
    end

    if length(buffer_line_points) > 0
        new_line_loop_points = [new_line_loop_points; uniformized_line_points(buffer_line_points, min_length)]
    end

    last_point_id = lines[findfirst(l -> l.id == line_loop.line_ids[end], lines)].point_ids[2]
    new_line_loop_points = [new_line_loop_points; points[findfirst(p -> p.id == last_point_id, points)]]
    @assert length(new_line_loop_points) - 1 <= length(line_loop.line_ids)

    new_lines::Vector{Line} = []
    for i in 1:length(new_line_loop_points)-1
        line = Line(line_loop.line_ids[i], [new_line_loop_points[i].id, new_line_loop_points[i+1].id])
        push!(new_lines, line)
    end

    new_line_loop = LineLoop(line_loop.id, [line.id for line in new_lines])

    return new_line_loop, new_lines, new_line_loop_points[1:end-1]
end

function uniformized_surface(raw_surface_str::String, min_length::Float64)
    points::Vector{Point} = []
    lines::Vector{Line} = []
    line_loops::Vector{LineLoop} = []

    buffer::String = ""

    for line in eachline(IOBuffer(raw_surface_str))
        if startswith(line, "Point")
            push!(points, parse_point(line))
        elseif startswith(line, "Line Loop")
            push!(line_loops, parse_line_loop(line))
        elseif startswith(line, "Line")
            push!(lines, parse_line(line))
        else
            buffer *= line
            buffer *= "\n"
        end
    end

    @assert length(line_loops) == 1

    new_line_loop, new_lines, new_line_loop_points = uniformized_line_loop(line_loops[1], lines, points, min_length)

    for point in new_line_loop_points
        buffer *= geo_string(point)
    end

    for line in new_lines
        buffer *= geo_string(line)
    end

    buffer *= geo_string(new_line_loop)
    return buffer
end

"""
Uniformize the geo file by removing lines with small angle difference and small distance between lines.

# Arguments
- `geo_file_path::String`: the path of the input geo file.
- `output_path::String`: the path of the output geo file.
- `section_length_threshold::Int=40`: the threshold of the length of the section to be uniformized.
- `threshold_angle::Float64=18 * pi / 180`: the threshold of the angle difference between two lines.
- `min_length::Float64=5.0`: the minimum length of the line.

"""
function uniformize(
    geo_file_path::String,
    output_path::String;
    min_length::Float64=2 * 20.0 * (2.0^-1.5))

    if !isdir(dirname(output_path))
        mkpath(dirname(output_path))
    end

    open(geo_file_path, "r") do input
        open(output_path, "w") do output
            buffer = ""
            section_length = 0
            for line in eachline(input)
                if startswith(line, "Plane Surface")
                    write(output, uniformized_surface(buffer, min_length))
                    buffer = ""
                    section_length = 0
                end

                buffer *= line
                buffer *= "\n"
                section_length += 1
            end

            write(output, buffer)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    str = "Point(298)={1.350000e+02,-6.838350e+02,0.000000e+00};"
    point = parse_point(str)
    println(point)
    println(geo_string(point))

    str = "Line(2)={298, 299, 300};"
    line = parse_line(str)
    println(line)
    println(geo_string(line))

    str = "Line Loop(20)={124,125,126,127,128,129,130,266,267,268,269,297};"
    line_loop = parse_line_loop(str)
    println(line_loop)
    println(geo_string(line_loop))
end

end