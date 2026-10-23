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

# preadapt, chosen by WALL_PREADAPT (item 8e), on the 4x4 gate family
# ([-5,5]x[0,20], elements 2.5 x 5). Selected by element centroid so that each
# box picks exactly the elements named; the level guard keeps the driver from
# re-flagging the children. Runs before initialize(), so it sees geometry only.
#   start     : the two elements the pulse starts in, x in [-2.5, 2.5], y in
#               [5, 10] -- the 8d-i preadapt, for the in-solve runs
#   corner_ll : the bottom-left element, x in [-5, -2.5], y in [0, 5] -- two
#               inflow walls on the walled mesh; an inflow wall meeting the
#               x-wrap on the channel
#   corner_ur : the top-right element, x in [2.5, 5], y in [15, 20] -- two
#               outflow walls (the control: the ghost never enters F* there)
#   interior  : the element of 8.0 and item 3, x in [0, 2.5], y in [10, 15]
#               (touches no boundary) -- the regression tie to the 8.0 counts
function user_get_preadapt_flags!(adapt_flags, inputs, mesh, old_ad_lvl, connijk, nelem, ngl, max_level)
    sel = get(ENV, "WALL_PREADAPT", "start")
    if sel == "start"
        xc, yc, hx, hy = 0.0, 7.5, 2.0, 2.0
    elseif sel == "corner_ll"
        xc, yc, hx, hy = -3.75, 2.5, 1.0, 2.0
    elseif sel == "corner_ur"
        xc, yc, hx, hy = 3.75, 17.5, 1.0, 2.0
    elseif sel == "interior"
        xc, yc, hx, hy = 1.25, 12.5, 1.0, 2.0
    else
        error("WALL_PREADAPT must be one of start, corner_ll, corner_ur, interior; got \"$sel\"")
    end

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

# In-solve flags. Per element, m = max|q| over its nodes: refine if m > θ_r
# and the element is below max_level; coarsen if m < θ_c and it is above
# level 0; otherwise nothing. θ_c < θ_r is hysteresis, so an element does
# not blink between levels.
#
# No wrap mask (item 8d): on a natively periodic mesh a 2:1 face across the
# periodic boundary is a mortar like any interior one, so refinement follows
# the pulse across both wraps.
#
# q is the solution (npoin x neqs). connijk gives the element's nodes; coords
# is accepted for the hook's signature and not used.
function user_get_adapt_flags!(adapt_flags, inputs, old_ad_lvl, q, qe,
                               Tabs, qn, qc, qi, qr,
                               qs, qg, Pr, Ps, Pg,
                               S_micro, qsatt,
                               connijk, nelem, ngl,
                               coords,
                               max_level)
    θ_r = 0.05
    θ_c = 0.01

    for iel = 1:nelem
        m = 0.0
        for j = 1:ngl, i = 1:ngl
            m = max(m, abs(q[connijk[iel, i, j], 1]))
        end

        if m > θ_r && old_ad_lvl[iel] < max_level
            adapt_flags[iel] = refine_flag
        elseif m < θ_c && old_ad_lvl[iel] > 0
            adapt_flags[iel] = coarsen_flag
        else
            adapt_flags[iel] = nothing_flag
        end
    end
end