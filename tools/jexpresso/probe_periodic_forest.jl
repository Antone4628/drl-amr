# probe_periodic_forest.jl — 8b read: does a natively periodic Gmsh mesh give a periodic p4est forest?
#
# Standalone; touches no Jexpresso code. Runs in the Jexpresso environment, because the GridapP4est
# under test is the arm64 fork Jexpresso resolves:
#   cd ~/projects/Jexpresso
#   julia --project=. ~/projects/drl-amr/tools/jexpresso/probe_periodic_forest.jl <mesh.msh>
#
# Run it twice on the same nx x ny base: once on a mesh from gate_family_periodic_native.geo
# (native periodicity, z = 0) and once on a mesh from conv_family_periodic.geo (tag-only, z != 0).
# The tag-only run is the control: it goes through the same flatten Jexpresso applies today.
#
# Scenario (nx, ny >= 3):
#   adapt 1 — refine the coarse cell at the (xmin, ymin) corner.
#   adapt 2 — refine that cell's child at the domain corner. Corner balance must then refine the
#             three coarse cells across the wraps: (nx-1, 0), (0, ny-1), (nx-1, ny-1) — but only
#             if p4est knows about the wrap.
#
# Every signal prints its value next to both predictions (periodic, tag-only) and says which it
# matches. S6 (face labels) is informational.

import Jansson_jll   # loads libjansson so libp4est's @rpath dependency resolves (as Jexpresso.jl:118 does)
using MPI
using PartitionedArrays
using Gridap
using Gridap.Geometry
using Gridap.Adaptivity
using GridapDistributed
using GridapGmsh
using GridapP4est

# Explicit bindings: several of these names are exported by more than one of the packages above,
# and Julia binds none of an ambiguous export.
import Gridap.Geometry: get_node_coordinates, get_cell_coordinates, get_cell_node_ids, get_grid,
                        get_grid_topology, get_faces, get_face_labeling, get_face_mask,
                        num_nodes, num_vertices, num_cells, num_cell_dims, num_point_dims
import GridapDistributed: local_views, get_cell_gids
import PartitionedArrays: partition

const REFINE  = isdefined(GridapP4est, :refine_flag)  ? GridapP4est.refine_flag  : Cint(1)
const NOTHING = isdefined(GridapP4est, :nothing_flag) ? GridapP4est.nothing_flag : Cint(0)

const MSHFILE = ARGS[1]

centroid(c) = (sum(p[1] for p in c) / length(c), sum(p[2] for p in c) / length(c))

function cell_nearest(model, target)
    best, bd = 0, Inf
    for (i, c) in enumerate(get_cell_coordinates(model))
        x, y = centroid(c)
        d = hypot(x - target[1], y - target[2])
        if d < bd
            best, bd = i, d
        end
    end
    return best, bd
end

function count_in_box(model, x0, x1, y0, y1)
    n = 0
    for c in get_cell_coordinates(model)
        x, y = centroid(c)
        if x0 < x < x1 && y0 < y < y1
            n += 1
        end
    end
    return n
end

function facet_cell_histogram(model)
    Dc  = num_cell_dims(model)
    f2c = get_faces(get_grid_topology(model), Dc - 1, Dc)
    h   = Dict{Int,Int}()
    for f in 1:length(f2c)
        k = length(f2c[f])
        h[k] = get(h, k, 0) + 1
    end
    return sort(collect(h))
end

function geometry_extents(model)
    wmax, hmax, area = 0.0, 0.0, 0.0
    for c in get_cell_coordinates(model)
        xs = [p[1] for p in c]
        ys = [p[2] for p in c]
        w  = maximum(xs) - minimum(xs)
        h  = maximum(ys) - minimum(ys)
        wmax = max(wmax, w)
        hmax = max(hmax, h)
        area += w * h
    end
    return wmax, hmax, area
end

function label_counts(model)
    labels = get_face_labeling(model)
    Dc  = num_cell_dims(model)
    out = String[]
    for name in labels.tag_to_name
        n0 = count(get_face_mask(labels, name, 0))
        n1 = count(get_face_mask(labels, name, Dc - 1))
        push!(out, "    $(rpad(name, 12)) vertices=$(n0)  facets=$(n1)")
    end
    return out
end

# Same operation as Jexpresso's _flatten_model_to_cell_dim (mesh.jl:1699) for the 2D case:
# rebuild the grid at Dp = Dc and the topology FROM THAT GRID'S NODE IDS.
function flatten_like_jexpresso(model)
    grid   = get_grid(model)
    Dc     = num_cell_dims(model)
    coords = [Gridap.Point(ntuple(i -> p[i], Dc)...) for p in get_node_coordinates(grid)]
    gflat  = Gridap.Geometry.UnstructuredGrid(coords,
                                              get_cell_node_ids(grid),
                                              Gridap.Geometry.get_reffes(grid),
                                              Gridap.Geometry.get_cell_type(grid),
                                              Gridap.Geometry.OrientationStyle(grid))
    tflat  = Gridap.Geometry.UnstructuredGridTopology(gflat)
    return Gridap.Geometry.UnstructuredDiscreteModel(gflat, tflat, get_face_labeling(model))
end

const TALLY = Dict("PERIODIC" => 0, "TAG-ONLY" => 0, "NEITHER" => 0, "BOTH" => 0)

function check(label, got, expect_p, expect_t)
    if got == expect_p && got == expect_t
        v = "BOTH"
    elseif got == expect_p
        v = "PERIODIC"
    elseif got == expect_t
        v = "TAG-ONLY"
    else
        v = "NEITHER"
    end
    TALLY[v] += 1
    println("  ", rpad(label, 44), "got=", got, "   periodic=", expect_p, "   tag-only=", expect_t, "   -> ", v)
end

function flags_for(ranks, model, cell)
    map(ranks, partition(get_cell_gids(model.dmodel))) do rank, indices
        f = fill(NOTHING, length(indices))
        f[cell] = REFINE
        f
    end
end

with_mpi() do distribute
    ranks = distribute(LinearIndices((MPI.Comm_size(MPI.COMM_WORLD),)))

    println("== mesh ==")
    println("  ", MSHFILE)
    println("  flag values: REFINE=", REFINE, "  NOTHING=", NOTHING)

    gmodel = GmshDiscreteModel(MSHFILE; renumber = true)
    Dc = num_cell_dims(gmodel)
    Dp = num_point_dims(gmodel)

    nodes = get_node_coordinates(gmodel)
    xmin, xmax = extrema(p[1] for p in nodes)
    ymin, ymax = extrema(p[2] for p in nodes)
    cents = [centroid(c) for c in get_cell_coordinates(gmodel)]
    nx = length(unique(round(c[1]; digits = 8) for c in cents))
    ny = length(unique(round(c[2]; digits = 8) for c in cents))
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    println("  Dc=", Dc, "  Dp=", Dp, "  domain=[", xmin, ",", xmax, "]x[", ymin, ",", ymax, "]",
            "  base=", nx, "x", ny, "  dx=", dx, "  dy=", dy)
    (nx >= 3 && ny >= 3) || error("probe predictions need nx, ny >= 3")

    nb  = 2 * (nx + ny)
    nf  = 2 * nx * ny
    hist_p = [2 => nf]
    hist_t = [1 => nb, 2 => nf - nx - ny]

    println("== S1 coarse model as read (GridapGmsh) ==")
    check("num_nodes (geometry)",        num_nodes(gmodel),    (nx + 1) * (ny + 1), (nx + 1) * (ny + 1))
    check("num_vertices (topology)",     num_vertices(gmodel), nx * ny,             (nx + 1) * (ny + 1))
    check("facet histogram (cells => facets)", facet_cell_histogram(gmodel), hist_p, hist_t)
    println("  labels:")
    foreach(println, label_counts(gmodel))

    if Dp != Dc
        println("== flatten ==")
        println("  Dp != Dc: applying the same flatten Jexpresso applies (topology rebuilt from node ids)")
        gmodel = flatten_like_jexpresso(gmodel)
        check("num_vertices after flatten",  num_vertices(gmodel), nx * ny, (nx + 1) * (ny + 1))
    else
        println("== flatten ==")
        println("  Dp == Dc: Jexpresso's flatten helper would be a no-op")
    end

    println("== S2 coarse octree model ==")
    model0 = OctreeDistributedDiscreteModel(ranks, gmodel)
    cm0 = PartitionedArrays.getany(local_views(model0))
    println("  connectivity branch: ", num_nodes(gmodel) == num_vertices(gmodel) ?
            "from_geometry (nodes == vertices)" : "with_topology (nodes != vertices)")
    check("coarse octree num_cells",            num_cells(cm0),            nx * ny, nx * ny)
    check("coarse octree num_vertices",         num_vertices(cm0),         nx * ny, (nx + 1) * (ny + 1))
    check("coarse octree facet histogram",      facet_cell_histogram(cm0), hist_p, hist_t)

    println("== S3 adapt 1: refine the (xmin, ymin) corner cell ==")
    c1, d1 = cell_nearest(cm0, (xmin + dx / 2, ymin + dy / 2))
    println("  flagged local cell ", c1, " (centroid distance ", d1, ")")
    model1, glue1 = Gridap.Adaptivity.adapt(model0, flags_for(ranks, model0, c1))
    cm1 = PartitionedArrays.getany(local_views(model1))
    ng1 = PartitionedArrays.getany(model1.non_conforming_glue)
    check("cells after adapt 1",                num_cells(cm1),             nx * ny + 3, nx * ny + 3)
    check("hanging facets after adapt 1",       ng1.num_hanging_faces[Dc],  8, 4)
    check("hanging vertices after adapt 1",     ng1.num_hanging_faces[1],   4, 2)
    println("  num_regular_faces=", ng1.num_regular_faces, "  num_hanging_faces=", ng1.num_hanging_faces)
    println("  facet histogram: ", facet_cell_histogram(cm1))

    println("== S4 adapt 2: refine that cell's child at the domain corner ==")
    c2, d2 = cell_nearest(cm1, (xmin + dx / 4, ymin + dy / 4))
    println("  flagged local cell ", c2, " (centroid distance ", d2, ")")
    model2, glue2 = Gridap.Adaptivity.adapt(model1, flags_for(ranks, model1, c2))
    cm2 = PartitionedArrays.getany(local_views(model2))
    ng2 = PartitionedArrays.getany(model2.non_conforming_glue)
    check("cells after adapt 2",                num_cells(cm2), nx * ny + 15, nx * ny + 6)
    check("cells in wrap box (nx-1, 0)",        count_in_box(cm2, xmax - dx, xmax, ymin, ymin + dy), 4, 1)
    check("cells in wrap box (0, ny-1)",        count_in_box(cm2, xmin, xmin + dx, ymax - dy, ymax), 4, 1)
    check("cells in wrap box (nx-1, ny-1)",     count_in_box(cm2, xmax - dx, xmax, ymax - dy, ymax), 4, 1)
    println("  num_regular_faces=", ng2.num_regular_faces, "  num_hanging_faces=", ng2.num_hanging_faces)
    println("  facet histogram: ", facet_cell_histogram(cm2))

    println("== S5 fine geometry is per side, not glued ==")
    wmax, hmax, area = geometry_extents(cm2)
    rtol = 1.0e-10
    println("  max cell width=", wmax, " (dx=", dx, ")  max cell height=", hmax, " (dy=", dy, ")")
    println("  total area=", area, " (domain=", (xmax - xmin) * (ymax - ymin), ")")
    s5 = wmax <= dx * (1 + rtol) && hmax <= dy * (1 + rtol) &&
         abs(area - (xmax - xmin) * (ymax - ymin)) <= rtol * (xmax - xmin) * (ymax - ymin)
    println("  S5 ", s5 ? "PASS" : "FAIL", " (applies to both meshes)")

    println("== S6 labels on the fine model (informational) ==")
    foreach(println, label_counts(cm2))

    println("== tally ==")
    println("  ", TALLY)
end
