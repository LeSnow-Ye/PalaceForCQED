using CairoMakie
using CSV, DataFrames
using JSON
using LsqFit

function sweep_Lj_plot(
    path::AbstractString;
    eig_index::Int = 1,
    ylim::Tuple{Real,Real} = (0, 0),
    save_filename::AbstractString = "",
)
    # Plot Setup
    f = Figure(fontsize = 20)
    ax = Axis(
        f[1, 1];
        title = "Josephson Inductance Sweep",
        xlabel = "Lj (nH)",
        ylabel = "Eigenmode Freq. (GHz)",
    )

    if ylim != (0, 0)
        ylims!(ax, ylim)
    end

    # Data
    lj = []
    eig = []
    for dir in readdir(path)
        if isdir(joinpath(path, dir))
            try
                df = CSV.read(
                    joinpath(path, dir, "eig.csv"),
                    DataFrame;
                    normalizenames = true,
                )
                push!(eig, df.Re_f_GHz_[eig_index])
                m = match(r"L(\d+\.\d+)nH", dir)
                push!(lj, parse(Float64, m.captures[1]))
            catch
            end
        end
    end

    # Curve Fitting
    model(x, p) = p[1] ./ x .^ 0.5
    p0 = [1.0]
    fit = curve_fit(model, lj, eig, p0)

    min_lj = minimum(lj) - 2
    max_lj = maximum(lj) + 4
    x_fit = min_lj:0.1:max_lj
    y_fit = model(x_fit, fit.param)

    # Plotting
    scatter!(ax, lj, eig; label = "Data")
    lines!(ax, x_fit, y_fit; label = "Fit")
    p = round.(fit.param[1]; digits = 4)
    text!(
        ax,
        0.55,
        0.35,
        text = L"f=\frac{2 \pi}{\sqrt{L_{j}  C}} =\frac{%$(p)}{\sqrt{L_{j}}}",
        space = :relative,
    )

    # Legend
    axislegend(ax, position = :rt)

    # Save
    save_path = joinpath(@__DIR__, "imgs")
    if !isdir(dirname(save_path))
        mkpath(dirname(save_path))
    end

    if save_filename == ""
        # save(joinpath(path, save_filename), f)
        print("Saving to $(save_path)")
        save(jointpath(save_path, splitpath(path)[end] * ".svg"), f)
    else
        save(joinpath(save_path, save_filename), f)
    end

    # Display
    display(f)
end


output_path = joinpath("/data/lesnow/2DQv8_eb4_data", "Lj", "#1")
sweep_Lj_plot(output_path, save_filename = "sweep_Lj_plot.svg")
