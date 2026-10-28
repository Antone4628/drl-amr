function initialize(SD::NSD_2D, PT, mesh::St_mesh, inputs, OUTPUT_DIR::String, TFloat)
    """

        """
    println(" Initialize fields for 2D AD ........................ ")
    
    #---------------------------------------------------------------------------------
    # Solution variables:
    #
    # NOTICE: while these names can be arbitrary, the length of this tuple
    # defines neqs, which is the second dimension of q = define_q()
    # 
    #---------------------------------------------------------------------------------
    qvars = ("q")
    q = define_q(SD, mesh.nelem, mesh.npoin, mesh.ngl, qvars, TFloat, inputs[:backend]; neqs=length(qvars))
    #---------------------------------------------------------------------------------
    if (inputs[:backend] == CPU())
        xc = (maximum(mesh.x) + minimum(mesh.x))/2
        yc = 8.0 #m
        sx = 1.0
        sy = 1.0
        A = 1.0
        for iel_g = 1:mesh.nelem
            for j=1:mesh.ngl, i=1:mesh.ngl
            
                ip = mesh.connijk[iel_g,i,j]
                x, y = mesh.x[ip], mesh.y[ip]
            
                a1 = -((x - xc)/sx)^2
                a2 = -((y - yc)/sy)^2
                q.qn[ip,1] = A*exp(a1)*exp(a2)
            
             #Store initial background state for plotting and analysis of pertuebations
                q.qe[ip,1] = 0.0
            end
        end
    else
        xc = TFloat((maximum(mesh.x) + minimum(mesh.x))/2)
        yc = TFloat(8.0)
        sx = TFloat(1.0)
        sy = TFloat(1.0)
        A = TFloat(1.0)
        k = initialize_gpu!(inputs[:backend])
        k(q.qn, q.qe, mesh.x, mesh.y, xc, yc, sx, sy, A; ndrange = (mesh.npoin))
    end
    println(" Initialize fields for 2D AD ........................ DONE ")

    return q
end

@kernel function initialize_gpu!(qn, qe, x, y, xc, yc, sx, sy, A)
    ip = @index(Global, Linear)

    T = eltype(x)
    x = x[ip]
    y = y[ip]
    

    a1 = -((x - xc)/sx)^2
    a2 = -((y - yc)/sy)^2
    qn[ip,1] = A*exp(a1)*exp(a2)

             #Store initial background state for plotting and analysis of pertuebations
    qe[ip,1] = 0.0

end

# preadapt: refine the two elements of the 8x8 gate mesh ([-5,5]x[0,20],
# elements 1.25 x 2.5) that the pulse starts on -- x in [-1.25, 1.25],
# y in [7.5, 10]; the pulse starts at (0, 8), on their shared edge. Selected
# by element centroid ((-0.625, 8.75) and (0.625, 8.75)); the box half-widths
# sit strictly between those centroids and the neighbors' (x at +-1.875,
# y at 6.25 and 11.25). The level guard keeps the driver from re-flagging the
# children. Runs before initialize(), so it sees geometry only.
function user_get_preadapt_flags!(adapt_flags, inputs, mesh, old_ad_lvl, connijk, nelem, ngl, max_level)
    xc = 0.0
    yc = 8.75
    hx = 1.0
    hy = 1.0

    for iel = 1:nelem
        cx = 0.0
        cy = 0.0
        for (i, j) in ((1, 1), (ngl, 1), (1, ngl), (ngl, ngl))
            ip  = connijk[iel, i, j]
            cx += mesh.x[ip] / 4
            cy += mesh.y[ip] / 4
        end
        if abs(cx - xc) < hx && abs(cy - yc) < hy && old_ad_lvl[iel] < max_level
            adapt_flags[iel] = refine_flag
        else
            adapt_flags[iel] = nothing_flag
        end
    end
end

# In-solve flags (item 8a-ii). Per element, m = max|q| over its nodes, with a
# threshold pair per level transition:
#   refine  level 0 -> 1 if m > 0.05;   level 1 -> 2 if m > 0.3
#   coarsen level 1 -> 0 if m < 0.01;   level 2 -> 1 if m < 0.15
# Each coarsen threshold sits at least a factor of two under its refine
# threshold (hysteresis), so an element does not blink between levels.
#
# Wrap mask: refine only elements whose centroid lies in x in (-2.5, 2.5),
# y in (5, 15) -- two base-element rings in from every side of the 8x8 gate
# mesh. Corner balance may raise ring 2 to level 1; ring 1 stays at level 0,
# so no 2:1 face forms across the periodic boundary.
#
# INSOLVE_SCRIPT=1 replaces the indicator with fixed flags by adapt index
# (_INSOLVE_SCRIPT); each selection must match an exact element count or the
# run stops, and adapts past the last scripted one flag nothing.
#
# q is the solution (npoin x neqs) and coords the 3 x npoin node table; an
# element's centroid is the midpoint of its first and last lattice nodes.
const _INSOLVE_ADAPT_COUNT = Ref(0)

# adapt index => [(level, xlo, xhi, ylo, yhi, :refine or :coarsen, expected count)],
# selected by element centroid. Base cells named (column, row), 1-based from (-5, 0).
const _INSOLVE_SCRIPT = Dict(
    1 => [(0, -1.25,  0.0,  10.0, 12.5, :refine,  1)],   # refine (4,5)
    2 => [(1, -0.625, 0.0,  8.75, 10.0, :refine,  1)],   # NE child of (4,4) to level 2; corner balance refines (5,5)
    3 => [(1,  0.0,   1.25, 10.0, 12.5, :coarsen, 4)],   # children of (5,5): balance undoes the coarsen
    4 => [(0, -2.5,  -1.25,  7.5, 10.0, :refine,  1),    # refine (3,4) ...
          (2, -0.625, 0.0,  8.75, 10.0, :coarsen, 4)],   # ... and coarsen the level-2 family: the Table arm
    5 => [(1,  0.0,   1.25, 10.0, 12.5, :coarsen, 4)],   # children of (5,5) again: realized now
)

function user_get_adapt_flags!(adapt_flags, inputs, old_ad_lvl, q, qe,
                               Tabs, qn, qc, qi, qr,
                               qs, qg, Pr, Ps, Pg,
                               S_micro, qsatt,
                               connijk, nelem, ngl,
                               coords,
                               max_level)
    _INSOLVE_ADAPT_COUNT[] += 1
    nadapt = _INSOLVE_ADAPT_COUNT[]

    for iel = 1:nelem
        adapt_flags[iel] = nothing_flag
    end

    if get(ENV, "INSOLVE_SCRIPT", "0") == "1"
        for (lvl, xlo, xhi, ylo, yhi, act, nexp) in get(_INSOLVE_SCRIPT, nadapt, [])
            nhit = 0
            for iel = 1:nelem
                old_ad_lvl[iel] == lvl || continue
                cx = (coords[1, connijk[iel, 1, 1]] + coords[1, connijk[iel, ngl, ngl]]) / 2
                cy = (coords[2, connijk[iel, 1, 1]] + coords[2, connijk[iel, ngl, ngl]]) / 2
                if xlo < cx < xhi && ylo < cy < yhi
                    adapt_flags[iel] = act == :refine ? refine_flag : coarsen_flag
                    nhit += 1
                end
            end
            println(" # INSOLVE_SCRIPT adapt ", nadapt, ": level ", lvl, " box (", xlo, ", ", xhi,
                    ") x (", ylo, ", ", yhi, ") ", act, " ", nhit, " element(s)")
            nhit == nexp || error("INSOLVE_SCRIPT adapt $nadapt: expected $nexp level-$lvl element(s) in the box, found $nhit")
        end
        return nothing
    end

    θ_r1 = 0.05
    θ_c1 = 0.01
    θ_r2 = 0.3
    θ_c2 = 0.15

    for iel = 1:nelem
        m = 0.0
        for j = 1:ngl, i = 1:ngl
            m = max(m, abs(q[connijk[iel, i, j], 1]))
        end
        cx      = (coords[1, connijk[iel, 1, 1]] + coords[1, connijk[iel, ngl, ngl]]) / 2
        cy      = (coords[2, connijk[iel, 1, 1]] + coords[2, connijk[iel, ngl, ngl]]) / 2
        inblock = -2.5 < cx < 2.5 && 5.0 < cy < 15.0
        lvl     = old_ad_lvl[iel]

        if lvl < max_level && inblock && ((lvl == 0 && m > θ_r1) || (lvl == 1 && m > θ_r2))
            adapt_flags[iel] = refine_flag
        elseif (lvl == 1 && m < θ_c1) || (lvl == 2 && m < θ_c2)
            adapt_flags[iel] = coarsen_flag
        end
    end
    return nothing
end