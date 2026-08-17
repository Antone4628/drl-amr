# conv_plot_multi.jl — multi-nop order table + combined log-log convergence figure.
# Usage: julia --project=. conv_plot_multi.jl <out.png> <results.txt> <nop> [<results.txt> <nop> ...]
# Each results file holds vtu_error.jl lines: "nex ney h L2 nodeRMS maxabs [dt cfl]".
# Per series: prints the segment-order table, fits ONE least-squares line over h <= 1
# (h = 2 is pre-asymptotic for this Gaussian at every nop tested — state the exclusion
# wherever the fitted orders are quoted), and draws data points + the fitted dashed line
# with only the fitted slope annotated (legend).
using Plots
using Printf
using Statistics

function read_series(path)
    rows = [parse.(Float64, split(l)) for l in eachline(path)
            if !isempty(strip(l)) && !startswith(strip(l), "#")]
    isempty(rows) && error("no data rows in $path")
    sort!(rows, by = r -> -r[3])   # coarse -> fine
    return [r[3] for r in rows], [r[4] for r in rows]
end

function main()
    (length(ARGS) >= 4 && (length(ARGS) - 1) % 3 == 0) ||
        error("usage: conv_plot_multi.jl <out.png> <results.txt> <nop> <hmax_fit> [...]")
    out = ARGS[1]
    series = [(ARGS[i], parse(Int, ARGS[i+1]), parse(Float64, ARGS[i+2])) for i in 2:3:length(ARGS)]

    plt = plot(xscale = :log10, yscale = :log10,
               xlabel = "element size h", ylabel = "L2 error at t = 4",
               title = "2D DG linear advection — L2 vs exact solution\nSSPRK33 (Δt floor-sized per mesh), upwind, collocated LGL; fits over h ≤ 1",
               titlefontsize = 11,
               legend = :topleft,
               minorgrid = true, gridalpha = 0.3,
               size = (760, 540), dpi = 300,
               framestyle = :box,
               top_margin = 3Plots.mm, left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)

    markers = [:circle, :rect, :diamond, :utriangle]
    allh = Float64[]
    for (k, (path, nop, hmax)) in enumerate(series)
        h, L2 = read_series(path)
        append!(allh, h)
        println("== nop = ", nop, "  (", path, ")")
        println("       h            L2         order")
        @printf("%10.4f   %.6e       —\n", h[1], L2[1])
        for i in 2:length(h)
            o = log(L2[i-1] / L2[i]) / log(h[i-1] / h[i])
            @printf("%10.4f   %.6e   %6.3f\n", h[i], L2[i], o)
        end
        fit_idx = [i for i in eachindex(h) if h[i] <= hmax + 1e-12]
        length(fit_idx) >= 2 || error("series $path: fewer than 2 points with h <= 1")
        lx = log.(h[fit_idx])
        ly = log.(L2[fit_idx])
        p = sum((lx .- mean(lx)) .* (ly .- mean(ly))) / sum((lx .- mean(lx)) .^ 2)
        a = mean(ly) - p * mean(lx)
        @printf("least-squares order over h <= 1 : %.3f   (expected %d)\n\n", p, nop + 1)
        c = palette(:default)[k]
        scatter!(plt, h, L2;
                 marker = markers[mod1(k, length(markers))], markersize = 6, color = c,
                 label = @sprintf("nop = %d   (fit: order %.2f, h ≤ %g)", nop, p, hmax))
        href = [minimum(h[fit_idx]), maximum(h[fit_idx])]
        plot!(plt, href, exp.(a .+ p .* log.(href));
              ls = :dash, color = c, linewidth = 1.5, label = "")
    end
    hticks = sort(unique(allh))
    plot!(plt, xticks = (hticks, [@sprintf("%g", v) for v in hticks]))
    savefig(plt, out)
    println("wrote ", out)
end

main()