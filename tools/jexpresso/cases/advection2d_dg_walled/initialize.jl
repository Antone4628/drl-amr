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

# preadapt: refine exactly one element of the 4x4 gate mesh ([-5,5]x[0,20],
# elements 2.5 x 5) to level 1 before t=0 — the element with corners
# x in [0, 2.5], y in [10, 15], chosen because it touches no periodic
# boundary. Selected by element centroid rather than by node so that the
# box picks one element and only one; the level guard keeps the driver
# from re-flagging the children when it re-evaluates. Runs before
# initialize(), so it sees geometry only.
function user_get_preadapt_flags!(adapt_flags, inputs, mesh, old_ad_lvl, connijk, nelem, ngl, max_level)
    xc = 1.25
    yc = 12.5
    hx = 1.0    # half-width of the box in x — strictly inside the target element
    hy = 2.0    # half-width of the box in y

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

# In-solve adaptation is off for this case (:lamr => false); the driver still
# requires the function to exist. All nothing_flag: the mesh never changes
# after the preadapt.
function user_get_adapt_flags!(adapt_flags, inputs, old_ad_lvl, q, qe,
                               Tabs, qn, qc, qi, qr,
                               qs, qg, Pr, Ps, Pg,
                               S_micro, qsatt,
                               connijk, nelem, ngl,
                               coords,
                               max_level)
    for iel = 1:nelem
        adapt_flags[iel] = nothing_flag
    end
end
