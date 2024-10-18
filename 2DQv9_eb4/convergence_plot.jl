using CairoMakie
using CSV, DataFrames
using JSON
using LsqFit

function convergence_plot(path::AbstractString; eig_index::Int=1, ylim::Tuple{Real,Real}=(0, 0), fitting::Bool=false, save_filename::AbstractString="")
    dirs = readdir(path)
    eig_r15o4 = 0.0

    f = Figure()
    ax = Axis(f[1, 1];
        title="Convergence Test for Order and Refinement Level",
        xlabel="Refinement Level",
        ylabel="Eigenmode Freq. (GHz)",
        xticks=0:0.5:3.0
    )

    if ylim != (0, 0)
        ylims!(ax, ylim)
    end

    legend_icon = []
    legend_label = []

    for order in range(3, 6)
        ref = Real[]
        eig = Real[]
        time = Real[]
        for refine in range(1, 9 - order)
            r = refine / 2 - 0.5
            try
                dir = dirs[findfirst(x -> occursin("r$(r)o$(order)", x), dirs)]
                df = CSV.read(joinpath(output_path, dir, "eig.csv"), DataFrame; normalizenames=true)
                js = JSON.parsefile(joinpath(output_path, dir, "palace.json"))

                push!(eig, df.Re_f_GHz_[eig_index])
                push!(ref, r)
                push!(time, js["ElapsedTime"]["Durations"]["Total"])

                if order == 4 && r == 1.5
                    eig_r15o4 = eig[end]
                end
            catch
            end
        end

        if fitting
            # Curve Fit
            # a - b * (c ^ (d - x))
            model(x, p) = p[1] .- p[2] .* (p[3] .^ (p[4] .- x))
            p0 = [3.9, 1.0, 2.0, 0.0]

            if order < 5
                fit = curve_fit(model, ref[2:end], eig[2:end], p0)
            else
                fit = curve_fit(model, ref, eig, p0)
            end

            x1 = 0:0.1:3.0
            y1 = model(x1, fit.param)
            li = lines!(ax, x1, y1; alpha=0.4, label="Order $(order) Fit")
        else
            li = lines!(ax, ref, eig; alpha=0.6)
        end

        sc = scatter!(ax, ref, eig, markersize=[0.4 * sqrt(t) for t in time]; alpha=0.8)

        push!(legend_icon, [li, sc])
        push!(legend_label, "Order $(order)")
    end

    push!(legend_icon, MarkerElement(color=:blue, marker=:circle))
    push!(legend_label, "Elapsed Time (Size)")

    tol = 0.003
    hl = hlines!(ax, [eig_r15o4];
        color=:black, linestyle=(:dash), alpha=0.5)
    push!(legend_icon, hl)
    push!(legend_label, "Freq. of r1.5o4")

    hl_round = hlines!(ax, [eig_r15o4 * (1 + tol), eig_r15o4 * (1 - tol)];
        color=:black, linestyle=(:dashdot))
    push!(legend_icon, hl_round)
    push!(legend_label, "r1.5o4 ± $(tol * 100)%")

    axislegend(ax, legend_icon, legend_label, position=:rb)

    if save_filename != ""
        save(joinpath(path, save_filename), f)
    end

    display(f)
end

output_path = joinpath("/data/lesnow/2DQv8_eb4_data", "RQ", "convergence_test")
convergence_plot(output_path; save_filename="eig_qubit.svg", eig_index=1, ylim=(3.78, 3.9), fitting=true)
convergence_plot(output_path; save_filename="eig_resonator.svg", eig_index=2, fitting=false)


output_path = joinpath("/data/lesnow/2DQv8_eb4_data", "resonator", "convergence_test")
convergence_plot(output_path; save_filename="eig.svg", eig_index=1, fitting=false)