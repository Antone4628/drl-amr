# vtu_diag.jl — error-structure diagnostic: decomposes the error field into a rigid
# displacement along c, an amplitude deficit, and a residual; locates max error.
# Usage: julia --project=. vtu_diag.jl <iter_2_1.vtu> <nelemx> <nelemy>
using ReadVTK
using Printf

const XMIN, XMAX = -5.0, 5.0
const YMIN, YMAX =  0.0, 20.0
const XC, YC = 2.0, 12.0
const CX, CY = 0.5, 1.0
q_exact(x, y) = exp(-(x - XC)^2) * exp(-(y - YC)^2)
dqdx(x, y) = -2 * (x - XC) * q_exact(x, y)
dqdy(x, y) = -2 * (y - YC) * q_exact(x, y)

const XI = [-1.0, -sqrt(3 / 7), 0.0, sqrt(3 / 7), 1.0]
const W  = [1 / 10, 49 / 90, 32 / 45, 49 / 90, 1 / 10]

function main()
    path = ARGS[1]; nex = parse(Int, ARGS[2]); ney = parse(Int, ARGS[3])
    dx = (XMAX - XMIN) / nex; dy = (YMIN < YMAX ? (YMAX - YMIN) / ney : 0.0)
    vtk = VTKFile(path)
    pts = get_points(vtk)
    q = vec(get_data(get_point_data(vtk)[first(keys(get_point_data(vtk)))]))
    x = pts[1, :]; y = pts[2, :]
    n = length(q)
    n == nex * ney * 25 || error("point count $n != $(nex * ney * 25)")
    cnorm = hypot(CX, CY)
    # weights as in vtu_error.jl
    wq = zeros(n)
    Jq = (dx / 2) * (dy / 2)
    for p in 1:n
        sx = (x[p] - XMIN) / dx; sy = (y[p] - YMIN) / dy
        xi = 2 * (sx - floor(sx)) - 1; eta = 2 * (sy - floor(sy)) - 1
        ki = argmin(abs.(xi .- XI)); kj = argmin(abs.(eta .- XI))
        wq[p] = W[ki] * W[kj] * Jq
    end
    e = [q[p] - q_exact(x[p], y[p]) for p in 1:n]
    # basis 1: unit displacement along c (e ≈ -d * (c/|c|)·∇q for a shift by d downstream)
    g1 = [-(CX * dqdx(x[p], y[p]) + CY * dqdy(x[p], y[p])) / cnorm for p in 1:n]
    # basis 2: amplitude deficit (e ≈ b * q_exact)
    g2 = [q_exact(x[p], y[p]) for p in 1:n]
    a11 = sum(wq .* g1 .* g1); a12 = sum(wq .* g1 .* g2); a22 = sum(wq .* g2 .* g2)
    b1  = sum(wq .* g1 .* e);  b2  = sum(wq .* g2 .* e)
    det = a11 * a22 - a12 * a12
    d = (b1 * a22 - b2 * a12) / det          # displacement along c (positive = downstream)
    b = (b2 * a11 - b1 * a12) / det          # relative amplitude deficit
    r = e .- d .* g1 .- b .* g2
    L2(v) = sqrt(sum(wq .* v .^ 2))
    ea, ia = findmax(abs.(e))
    # value at the node nearest the exact peak
    ip = argmin((x .- XC) .^ 2 .+ (y .- YC) .^ 2)
    @printf("L2(e) = %.6e   max|e| = %.6e at (%.4f, %.4f)\n", L2(e), ea, x[ia], y[ia])
    @printf("q at nearest-peak node (%.4f, %.4f): %.12f   (exact %.12f)\n",
            x[ip], y[ip], q[ip], q_exact(x[ip], y[ip]))
    @printf("fit: displacement along c = %.4e   amplitude deficit = %.4e\n", d, b)
    @printf("     equivalent time offset dt = %.4e\n", d / cnorm)
    @printf("L2 after removing fit = %.6e   (fraction remaining: %.3f)\n", L2(r), L2(r) / L2(e))
end

main()