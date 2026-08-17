# vtu_error.jl — LGL-weighted L2 error of a Jexpresso 2D DG .vtu against the exact
# translated Gaussian (AdvDiff advection2d_dg family, uniform mesh, t=4).
# Usage: julia --project=. vtu_error.jl <path/to/iter_2_1.vtu> <nelemx> <nelemy> <nop> [dt]
# Prints one line: nelemx nelemy h L2 nodeRMS maxabs [dt cfl]   (append to results_nopN.txt)
# cfl = dt*(|cx|+|cy|)/dx_min with dx_min = h*(min LGL gap)/2 — the node-scale CFL.
using ReadVTK
using Printf

const XMIN, XMAX = -5.0, 5.0
const YMIN, YMAX =  0.0, 20.0
const CX, CY = 0.5, 1.0   # advection velocity (user_flux.jl) — used only for the CFL column
# Exact solution at t = 4: IC Gaussian A=1, center (0,8), sx=sy=1 (initialize.jl),
# advected by c = (0.5, 1.0) -> center (2,12) — as the PERIODIC image sum, not the
# free-space Gaussian. The nearest x-image (center 2-10 = -8) contributes exp(-9)
# = 1.23e-4 at the x = -5 boundary and has L2 norm 3.99e-5 over the domain: omitting
# it floored the first sweep at exactly that value (measured 2026-08-16). y-images
# (12±20) contribute <= exp(-64): genuinely negligible. One image pair per direction
# suffices (next pair is exp(-(13)^2)-class).
const XC, YC = 2.0, 12.0
const SX, SY = 1.0, 1.0
const A = 1.0
const LX, LY = XMAX - XMIN, YMAX - YMIN
gx(x) = exp(-((x - XC) / SX)^2) + exp(-((x - XC - LX) / SX)^2) + exp(-((x - XC + LX) / SX)^2)
gy(y) = exp(-((y - YC) / SY)^2) + exp(-((y - YC - LY) / SY)^2) + exp(-((y - YC + LY) / SY)^2)
q_exact(x, y) = A * gx(x) * gy(y)

# LGL nodes/weights for arbitrary nop, by Newton iteration on the Legendre recurrence
# (standard lglnodes algorithm). Verified against the nop=4 closed form:
# nodes {±1, ±sqrt(3/7), 0}, weights {1/10, 49/90, 32/45}.
function lgl_nodes_weights(nop::Int)
    N = nop
    ngl = N + 1
    x = [-cos(pi * i / N) for i in 0:N]   # Chebyshev-Lobatto initial guess, ascending
    P = zeros(ngl)
    xold = fill(2.0, ngl)
    while maximum(abs.(x .- xold)) > 1e-15
        xold .= x
        for j in 1:ngl
            p0 = 1.0
            p1 = x[j]
            for k in 2:N
                p2 = ((2k - 1) * x[j] * p1 - (k - 1) * p0) / k
                p0 = p1
                p1 = p2
            end
            # p1 = P_N(x_j), p0 = P_{N-1}(x_j)
            x[j] = xold[j] - (x[j] * p1 - p0) / (ngl * p1)
            P[j] = p1
        end
    end
    w = 2.0 ./ (N * ngl .* P .^ 2)
    return x, w
end

function nearest_node(xi, XI)
    best = 1
    bd = abs(xi - XI[1])
    for k in 2:length(XI)
        d = abs(xi - XI[k])
        if d < bd
            bd = d
            best = k
        end
    end
    return best, bd
end

function main()
    length(ARGS) in (4, 5) || error("usage: vtu_error.jl <vtu> <nelemx> <nelemy> <nop> [dt]")
    path = ARGS[1]
    nex = parse(Int, ARGS[2])
    ney = parse(Int, ARGS[3])
    nop = parse(Int, ARGS[4])
    dt = length(ARGS) == 5 ? parse(Float64, ARGS[5]) : nothing
    ngl = nop + 1
    XI, W = lgl_nodes_weights(nop)
    dx = (XMAX - XMIN) / nex
    dy = (YMAX - YMIN) / ney
    abs(dx - dy) < 1e-12 || error("family assumes dx == dy; got dx=$dx dy=$dy")
    vtk = VTKFile(path)
    pts = get_points(vtk)
    pd  = get_point_data(vtk)
    name = first(keys(pd))
    q = vec(get_data(pd[name]))
    x = pts[1, :]
    y = pts[2, :]
    n = length(q)
    n == nex * ney * ngl^2 ||
        error("point count $n != nelem*ngl^2 = $(nex * ney * ngl^2) — wrong mesh/nop args, or not a DG piece file")
    Jq = (dx / 2) * (dy / 2)
    l2sq = 0.0
    ssq = 0.0
    mx = 0.0
    for p in 1:n
        sx_ = (x[p] - XMIN) / dx
        sy_ = (y[p] - YMIN) / dy
        xi  = 2 * (sx_ - floor(sx_)) - 1
        eta = 2 * (sy_ - floor(sy_)) - 1
        # A point that floats onto an element's right/top edge wraps to xi/eta = -1 of the
        # next element index; endpoint weights are equal (W[1] == W[ngl]), so no correction.
        ki, di = nearest_node(xi, XI)
        kj, dj = nearest_node(eta, XI)
        (di < 1e-8 && dj < 1e-8) ||
            error("point $p at ($(x[p]), $(y[p])) is $(max(di, dj)) off an LGL node — mesh/nop args wrong?")
        e = q[p] - q_exact(x[p], y[p])
        l2sq += W[ki] * W[kj] * Jq * e^2
        ssq  += e^2
        mx = max(mx, abs(e))
    end
    if dt === nothing
        @printf("%d %d %.6f %.12e %.12e %.12e\n", nex, ney, dx, sqrt(l2sq), sqrt(ssq / n), mx)
    else
        dxmin = dx * minimum(diff(XI)) / 2
        cfl = dt * (abs(CX) + abs(CY)) / dxmin
        @printf("%d %d %.6f %.12e %.12e %.12e %.6g %.4f\n",
                nex, ney, dx, sqrt(l2sq), sqrt(ssq / n), mx, dt, cfl)
    end
end

main()