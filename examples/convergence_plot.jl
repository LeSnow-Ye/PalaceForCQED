using CairoMakie
using CSV, DataFrames
using JSON
using LsqFit

function convergence_plot(
    path::AbstractString;
    eig_index::Int=1,
    ylim::Tuple{Real,Real}=(0, 0),
    xlim::Tuple{Real,Real}=(0, 0),
    tol::Real=0.003,
    tol_bracket_xloc::Real=0.4,
    tol_bracket_flip::Bool=false,
    use_dofs::Bool=false,
    fitting::Bool=false,
    title::AbstractString="",
    save_filename::AbstractString="",
    legend_position::Symbol=:rb,
    display::Bool=false,
)
    dirs = readdir(path)
    eig_r15o4 = 0.0

    f = Figure(fontsize=18)
    ax = Axis(
        f[1, 1];
        # title = title == "" ? "Convergence Test" : title,
        xlabel=use_dofs ? "Degrees of Freedom" : "Refinement Level",
        ylabel="Eigenmode Frequency (GHz)",
        xticks=use_dofs ? Makie.automatic : (0:0.5:3),
    )

    if ylim != (0, 0)
        ylims!(ax, ylim)
    end

    if xlim != (0, 0)
        xlims!(ax, xlim)
    end

    legend_icon = []
    legend_label = []

    for order in range(3, 6)
        ref = Float64[]
        dof = Int64[]
        eig = Float64[]
        time = Float64[]
        for refine in range(1, 9 - order)
            r = refine / 2 - 0.5
            try
                dir = dirs[findfirst(x -> occursin("r$(r)o$(order)", x), dirs)]
                df = CSV.read(
                    joinpath(path, dir, "eig.csv"),
                    DataFrame;
                    normalizenames=true,
                )
                js = JSON.parsefile(joinpath(path, dir, "palace.json"))

                push!(eig, df.Re_f_GHz_[eig_index])
                push!(ref, r)
                push!(time, js["ElapsedTime"]["Durations"]["Total"])
                push!(dof, js["Problem"]["DegreesOfFreedom"])

                if order == 4 && r == 1.5
                    eig_r15o4 = eig[end]
                end
            catch
            end
        end

        x = use_dofs ? dof : ref

        if fitting
            # Curve Fit
            # a - b * (c ^ (d - x))
            model(x, p) = p[1] .- p[2] .* (p[3] .^ (p[4] .- x))
            p0 = [3.9, 1.0, 2.0, 0.0]

            if order < 5
                fit = curve_fit(model, x[2:end], eig[2:end], p0)
            else
                fit = curve_fit(model, x, eig, p0)
            end

            x1 = 0:0.1:3.0
            y1 = model(x1, fit.param)
            li = lines!(ax, x1, y1; alpha=0.4, label="Order $(order) Fit")
        else
            li = lines!(ax, x, eig; alpha=0.6)
        end

        sc = scatter!(ax, x, eig, markersize=[0.4 * sqrt(t) for t in time]; alpha=0.8)

        push!(legend_icon, [li, sc])
        push!(legend_label, "Order $(order)")
    end

    push!(legend_icon, MarkerElement(color=:blue, marker=:circle))
    push!(legend_label, "Elapsed Time (Size)")

    # tol
    hl = hlines!(ax, [eig_r15o4]; color=:black, linestyle=(:dash), alpha=0.5)
    push!(legend_icon, hl)
    push!(legend_label, "Mode of r1.5o4")

    plus = eig_r15o4 * (1 + tol)
    minus = eig_r15o4 * (1 - tol)
    hl_round = hlines!(ax, [plus, minus]; color=(:red, 0.8), linestyle=(:dashdot))
    # push!(legend_icon, hl_round)
    # push!(legend_label, "r1.5o4 ± $(tol * 100)%")

    bracket_args = (;
        orientation=tol_bracket_flip ? :down : :up,
        rotation=0,
        textoffset=28,
        color=(:red, 0.8),
        textcolor=(:red, 0.8),
    )

    bracket!(
        tol_bracket_xloc,
        eig_r15o4,
        tol_bracket_xloc,
        plus;
        text="+$(tol * 100)%",
        bracket_args...,
    )
    bracket!(
        tol_bracket_xloc,
        minus,
        tol_bracket_xloc,
        eig_r15o4;
        text="-$(tol * 100)%",
        bracket_args...,
    )


    # Legend
    axislegend(ax, legend_icon, legend_label, position=legend_position)
    # Legend(f[1, 2], legend_icon, legend_label, position = :rb)

    # Save
    save_path = joinpath(@__DIR__, "imgs-notitle")
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
    if display
        display(f)
    end
end

#WARNING: function is not generalized.

result_path = joinpath("/data/lesnow/2DQv8_eb4_data", "RQ", "convergence_test_#3")
convergence_plot(
    result_path;
    title="Convergence Test for Qubit",
    save_filename="convergence_test_qubit.svg",
    xlim=(-0.1, 3.1),
    tol_bracket_flip=true,
    tol_bracket_xloc=2.65,
    eig_index=1,
    fitting=false,
)

convergence_plot(
    result_path;
    title="Convergence Test for Resonator",
    save_filename="convergence_test_resonator.svg",
    eig_index=2,
    fitting=false,
    legend_position=:rt,
)

convergence_plot(
    result_path;
    title="Convergence Test for Qubit",
    save_filename="convergence_test_qubit_dof.svg",
    xlim=(-0.5e6, 1.8e7),
    tol_bracket_flip=true,
    tol_bracket_xloc=1.53e7,
    eig_index=1,
    fitting=false,
    use_dofs=true,
)

convergence_plot(
    result_path;
    title="Convergence Test for Resonator",
    save_filename="convergence_test_resonator_dof.svg",
    xlim=(-0.5e6, 1.8e7),
    tol_bracket_flip=true,
    tol_bracket_xloc=1.53e7,
    fitting=false,
    eig_index=2,
    legend_position=:rt,
    use_dofs=true,
)
