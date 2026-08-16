# conv_plot.jl — order table + publication-quality log-log plot from vtu_error.jl lines.
# Usage: julia --project=. conv_plot.jl results.txt [out.png]
using Plots
using Printf
using Statistics

function main()
    isempty(ARGS) && error("usage: conv_plot.jl results.txt [out.png]")
    rows = [parse.(Float64, split(l)) for l in eachline(ARGS[1])
            if !isempty(strip(l)) && !startswith(strip(l), "#")]
    sort!(rows, by = r -> -r[3])   # coarse -> fine
    h  = [r[3] for r in rows]
    L2 = [r[4] for r in rows]

    # per-segment observed orders
    ords = [log(L2[i-1] / L2[i]) / log(h[i-1] / h[i]) for i in 2:length(h)]
    println("       h            L2         order")
    @printf("%10.4f   %.6e       —\n", h[1], L2[1])
    for i in 2:length(h)
        @printf("%10.4f   %.6e   %6.3f\n", h[i], L2[i], ords[i-1])
    end

    # least-squares slope over the asymptotic segment (drop the coarsest point:
    # h = 2 sigma is pre-asymptotic for this Gaussian)
    fit_idx = length(h) > 2 ? (2:length(h)) : (1:length(h))
    lx = log.(h[fit_idx]); ly = log.(L2[fit_idx])
    p_fit = sum((lx .- mean(lx)) .* (ly .- mean(ly))) / sum((lx .- mean(lx)) .^ 2)
    @printf("least-squares order over h <= %.4g : %.3f\n", h[fit_idx[1]], p_fit)

    out = length(ARGS) >= 2 ? ARGS[2] : "convergence_2d_dg.png"
    plt = plot(h, L2;
               xscale = :log10, yscale = :log10,
               marker = :circle, markersize = 6, linewidth = 2,
               label = @sprintf("L2 error  (fit: order %.2f)", p_fit),
               xlabel = "element size h", ylabel = "L2 error at t = 4",
               title = "2D DG linear advection, nop = 4 — L2 vs exact solution\nΔt = 0.005 (0.00125 at finest), SSPRK33, upwind, collocated LGL",
               titlefontsize = 12,
               top_margin = 3Plots.mm,
               legend = :topleft,
               xticks = (h, [@sprintf("%g", v) for v in h]),
               minorgrid = true, gridalpha = 0.3,
               size = (760, 540), dpi = 300,
               framestyle = :box,
               left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)

    # slope-5 reference anchored at the finest point
    href = [h[end], h[1]]
    eref = L2[end] .* (href ./ h[end]) .^ 5
    plot!(plt, href, eref; ls = :dash, color = :gray, linewidth = 1.5,
          label = "slope 5 (reference)")

    # annotate each segment with its observed order, offset above the midpoint
    for i in 2:length(h)
        xm = sqrt(h[i-1] * h[i])
        ym = sqrt(L2[i-1] * L2[i]) * 2.2
        #annotate!(plt, xm, ym, text(@sprintf("%.2f", ords[i-1]), 9, :gray30, :center))
        annotate!(plt, xm, ym, text(@sprintf("p = %.2f", ords[i-1]), 11, :black, :center))
    end

    savefig(plt, out)
    println("wrote ", out)
end

main()