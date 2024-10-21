using PalaceForCQED.Utils

using CairoMakie
using CSV, DataFrames

function driven_plot(path::AbstractString;
    xlim::Tuple{Real,Real}=(0, 0),
    ylim_s::Tuple{Real,Real}=(0, 0),
    show_arg::Bool=true,
    ylim_arg::Tuple{Real,Real}=(0, 0),
    ref_data::Vector{<:Real}=Float64[],
    tol::Real=0.0,
    figure_size::Tuple{Real,Real}=(600, 450),
    save_filename::AbstractString=""
)
    # Plot Setup
    f = Figure(size=figure_size)

    kwargs = (; xgridvisible=false, ygridvisible=false)
    ax = Axis(f[1, 1];
        title="Port Response of 2DQv8_eb4",
        xlabel="Frequency (GHz)",
        ylabel="|S[1][1]| (dB)",
        yautolimitmargin=(0.15f0, 0.05f0), kwargs...
    )

    if show_arg
        ax_arg = Axis(f[1, 1];
            ylabel="arg(S[1][1]) (deg.)",
            yaxisposition=:right,
            yticks=[-180, 0, 180], kwargs...
        )
    end

    if ylim_s != (0, 0)
        ylims!(ax, ylim_s)
    end

    if show_arg
        if ylim_arg != (0, 0)
            ylims!(ax_arg, ylim_arg)
        end
    end

    if xlim != (0, 0)
        xlims!(ax, xlim)
        if show_arg
            xlims!(ax_arg, xlim)
        end
    end

    legend_icon = []
    legend_label = []

    # Data
    df = CSV.read(joinpath(path, "port-S.csv"), DataFrame; normalizenames=true)

    freq = df.f_GHz_
    s11 = df._S_1_1_dB_
    arg11 = df.arg_S_1_1_deg_

    l_s = lines!(ax, freq, s11; label="S[1][1]")
    push!(legend_icon, l_s)
    push!(legend_label, "S[1][1]")

    if show_arg
        l_arg = lines!(ax_arg, freq, arg11; label="arg(S[1][1])", color=:orange, linewidth=1)
        push!(legend_icon, l_arg)
        push!(legend_label, "arg(S[1][1])")
    end

    # Valleys
    valley_indices = find_valleys(s11, 2)
    println("Valley Indices: $valley_indices")
    scatter!(ax, freq[valley_indices], s11[valley_indices]; color=:red)
    text!(ax, freq[valley_indices], s11[valley_indices];
        text=["$(round(f, digits=4))" for f in freq[valley_indices]],
        align=(:center, :top),
        offset=(0, -5)
    )

    # Tol
    if tol != 0.0
        valley_round = vcat(freq[valley_indices] .* (1 + tol), freq[valley_indices] .* (1 - tol))
        vl_tol = vlines!(ax, valley_round; color=:black, linestyle=(:dashdot))
        push!(legend_icon, vl_tol)
        push!(legend_label, "Valley ± $(tol * 100)%")
    end

    # Reference Data
    if length(ref_data) > 0
        vl = vlines!(ax, ref_data; color=:gray, linestyle=:dash, linewidth=1, label="Reference Data")
        push!(legend_icon, vl)
        push!(legend_label, "Reference Data")
    end

    axislegend(ax, legend_icon, legend_label, position=:rb)

    # Save
    if save_filename != ""
        save(joinpath(path, save_filename), f)
    end

    display(f)
end

output_path = joinpath("/data/lesnow/2DQv8_eb4_data", "driven", "lumped_3.83-3.93_Step0.0001_2024-10-11T151550")
driven_plot(output_path; show_arg=false, ref_data=[3.88017], tol=0.003, save_filename="plot.svg")

output_path = joinpath("/data/lesnow/2DQv8_eb4_data", "driven", "lumped_4.35-4.45_Step0.0001_2024-10-11T132010")
driven_plot(output_path; xlim=(4.37, 4.45), show_arg=false, ref_data=[4.38893], tol=0.003, save_filename="plot.svg")

output_path = joinpath("/data/lesnow/2DQv8_eb4_data", "driven", "lumped_6.9-7.8_Step0.001_2024-10-11T103611")
refs = [7.178727, 7.2684245, 7.5484463, 7.644717]
driven_plot(output_path; xlim=(7.0, 7.8), ylim_arg=(-120, 190), ref_data=refs, figure_size=(800, 550), save_filename="plot.svg")