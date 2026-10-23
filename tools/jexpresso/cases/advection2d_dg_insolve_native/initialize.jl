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

# preadapt: refine the two elements of the 4x4 gate mesh ([-5,5]x[0,20],
# elements 2.5 x 5) that the pulse starts in -- x in [-2.5, 2.5], y in
# [5, 10]; the pulse starts at (0, 8), on their shared edge. Selected by
# element centroid ((-1.25, 7.5) and (1.25, 7.5)); the box half-widths sit
# strictly between those centroids and the neighbours'. The level guard keeps
# the driver from re-flagging the children. Runs before initialize(), so it
# sees geometry only.
function user_get_preadapt_flags!(adapt_flags, inputs, mesh, old_ad_lvl, connijk, nelem, ngl, max_level)
    xc = 0.0
    yc = 7.5
    hx = 2.0
    hy = 2.0

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