# vtu_error.jl — LGL-weighted L2 error of a Jexpresso 2D DG .vtu against the exact
# translated Gaussian (AdvDiff advection2d_dg family, nop=4, uniform mesh, t=4).
# Usage: julia --project=. vtu_error.jl <path/to/iter_2_1.vtu> <nelemx> <nelemy>
# Prints one line: nelemx nelemy h L2 nodeRMS maxabs   (append to results.txt)
using ReadVTK
using Printf

const XMIN, XMAX = -5.0, 5.0
const YMIN, YMAX =  0.0, 20.0
# Exact solution at t = 4: IC Gaussian A=1, center (0,8), sx=sy=1 (initialize.jl),
# advected by c = (0.5, 1.0) (user_flux.jl) -> center (2,12) — as the PERIODIC image
# sum, not the free-space Gaussian. The nearest x-image (center 2-10 = -8) contributes
# exp(-9) = 1.23e-4 at the x = -5 boundary and has L2 norm 3.99e-5 over the domain:
# omitting it floored the first sweep at exactly that value (measured 2026-08-16).
# y-images (12±20) contribute <= exp(-64): genuinely negligible. One image pair per
# direction suffices (next pair is exp(-(13)^2)-class).
const XC, YC = 2.0, 12.0
const SX, SY = 1.0, 1.0
const A = 1.0
const LX, LY = XMAX - XMIN, YMAX - YMIN
gx(x) = exp(-((x - XC) / SX)^2) + exp(-((x - XC - LX) / SX)^2) + exp(-((x - XC + LX) / SX)^2)
gy(y) = exp(-((y - YC) / SY)^2) + exp(-((y - YC - LY) / SY)^2) + exp(-((y - YC + LY) / SY)^2)
q_exact(x, y) = A * gx(x) * gy(y)

# LGL nodes/weights for nop = 4 (5 nodes) — matches the deck's :nop => 4.
const XI = [-1.0, -sqrt(3 / 7), 0.0, sqrt(3 / 7), 1.0]
const W  = [1 / 10, 49 / 90, 32 / 45, 49 / 90, 1 / 10]

function nearest_node(xi)
    best = 1; bd = abs(xi - XI[1])
    for k in 2:5
        d = abs(xi - XI[k])
        if d < bd
            bd = d; best = k
        end
    end
    return best, bd
end

function main()
    length(ARGS) == 3 || error("usage: vtu_error.jl <vtu> <nelemx> <nelemy>")
    path = ARGS[1]; nex = parse(Int, ARGS[2]); ney = parse(Int, ARGS[3])
    dx = (XMAX - XMIN) / nex
    dy = (YMAX - YMIN) / ney
    abs(dx - dy) < 1e-12 || error("family assumes dx == dy; got dx=$dx dy=$dy")
    vtk = VTKFile(path)
    pts = get_points(vtk)
    pd  = get_point_data(vtk)
    name = first(keys(pd))
    q = vec(get_data(pd[name]))
    x = pts[1, :]; y = pts[2, :]
    n = length(q)
    n == nex * ney * 25 ||
        error("point count $n != nelem*ngl^2 = $(nex * ney * 25) — wrong mesh args, or not a DG piece file")
    Jq = (dx / 2) * (dy / 2)
    l2sq = 0.0; ssq = 0.0; mx = 0.0
    for p in 1:n
        sx_ = (x[p] - XMIN) / dx
        sy_ = (y[p] - YMIN) / dy
        xi  = 2 * (sx_ - floor(sx_)) - 1
        eta = 2 * (sy_ - floor(sy_)) - 1
        # A point that floats onto an element's right/top edge wraps to xi/eta = -1 of the
        # next element index; endpoint weights are equal (W[1] == W[5]), so no correction.
        ki, di = nearest_node(xi)
        kj, dj = nearest_node(eta)
        (di < 1e-8 && dj < 1e-8) ||
            error("point $p at ($(x[p]), $(y[p])) is $(max(di, dj)) off an LGL node — mesh args wrong?")
        e = q[p] - q_exact(x[p], y[p])
        l2sq += W[ki] * W[kj] * Jq * e^2
        ssq  += e^2
        mx = max(mx, abs(e))
    end
    @printf("%d %d %.6f %.12e %.12e %.12e\n", nex, ney, dx, sqrt(l2sq), sqrt(ssq / n), mx)
end

main()