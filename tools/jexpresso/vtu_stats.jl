# vtu_stats.jl — field statistics of Jexpresso .vtu piece files.
# Promoted from JEXPRESSO_ENVIRONMENT.md §10 (evidentiary rule: script reads, never GUI).
# Usage: julia --project=. vtu_stats.jl <file1.vtu> [file2.vtu ...]
# Point at the PIECE file (iter_N/iter_N_1.vtu), not the .pvtu wrapper.
using ReadVTK

for path in ARGS
    vtk = VTKFile(path)
    pts = get_points(vtk)
    pd  = get_point_data(vtk)
    name = first(keys(pd))
    q = vec(get_data(pd[name]))
    x = pts[1, :]; y = pts[2, :]
    qmax, imax = findmax(q)
    w = max.(q, 0.0); W = sum(w)
    xb = sum(w .* x) / W; yb = sum(w .* y) / W
    sx = sqrt(2 * sum(w .* (x .- xb).^2) / W)   # for exp(-((x-xc)/s)^2): 2nd moment = s^2/2
    sy = sqrt(2 * sum(w .* (y .- yb).^2) / W)
    println(path)
    println("  field=", name, "  N=", length(q))
    println("  max=", qmax, " at (", x[imax], ", ", y[imax], ")   min=", minimum(q))
    println("  centroid=(", round(xb, digits=4), ", ", round(yb, digits=4),
            ")   width sx=", round(sx, digits=4), "  sy=", round(sy, digits=4))
end