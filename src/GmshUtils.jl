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

function length_of_line(l::Line, points::Vector{Point})
    p_start = points[findfirst(p -> p.id == l.point_ids[1], points)]
    p_end = points[findfirst(p -> p.id == l.point_ids[2], points)]
    return sqrt((p_end.x - p_start.x)^2 + (p_end.y - p_start.y)^2 + (p_end.z - p_start.z)^2)
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

function angle(vector)
    atan(vector[2], vector[1])
end

function angle_difference(line1::Line, line2::Line, points::Vector{Point})
    # 获取两条线的起点和终点
    p1_start = points[findfirst(p -> p.id == line1.point_ids[1], points)]
    p1_end = points[findfirst(p -> p.id == line1.point_ids[2], points)]
    p2_start = points[findfirst(p -> p.id == line2.point_ids[1], points)]
    p2_end = points[findfirst(p -> p.id == line2.point_ids[2], points)]

    # 计算两条线的向量
    vector1 = vector(p1_start, p1_end)
    vector2 = vector(p2_start, p2_end)
    angle_diff = angle(vector1) - angle(vector2)
    angle_diff = mod(angle_diff + pi, 2 * pi) - pi

    # println("angle1 = $(rad2deg(angle1)), angle2 = $(rad2deg(angle2)), angle_diff = $(rad2deg(angle_diff))")
    return abs(angle_diff) # 返回角度差的绝对值
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

            if (angle_difference(line1, line2, points) < threshold_angle) && (distance_between_lines(line1, line2, points) < min_length)
                if j == length(line_loop.line_ids)
                    push!(new_line_ids, line_loop.line_ids[j])
                    need_to_iterate = false
                end
                continue
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
        push!(point_id_set, line.point_ids[1])
        push!(point_id_set, line.point_ids[2])
    end

    loop_points::Vector{Point} = []
    for point_id in point_id_set
        push!(loop_points, points[findfirst(p -> p.id == point_id, points)])
    end

    return loop_points
end

function uniformize_surface(str::String, threshold_angle::Float64, min_length::Float64)
    points::Vector{Point} = []
    lines::Vector{Line} = []
    line_loops::Vector{LineLoop} = []

    buffer::String = ""

    for line in eachline(IOBuffer(str))
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

    remove_lines_until_threshold!(line_loops[1], lines, points, threshold_angle, min_length)
    enclose_line_loop!(line_loops[1], lines)

    for point in get_unique_points_from_line_loop(line_loops[1], lines, points)
        buffer *= geo_string(point)
    end

    for line_id in line_loops[1].line_ids
        buffer *= geo_string(lines[findfirst(l -> l.id == line_id, lines)])
    end

    buffer *= geo_string(line_loops[1])
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
    section_length_threshold::Int=40,
    threshold_angle::Float64=18 * pi / 180,
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
                    if section_length >= section_length_threshold
                        write(output, uniformize_surface(buffer, threshold_angle, min_length))
                    else
                        write(output, buffer)
                    end
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