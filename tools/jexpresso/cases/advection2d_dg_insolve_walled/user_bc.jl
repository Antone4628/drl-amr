"""
    qibdy is an Array{Floats} of size `nvars`

    src/equations/EQUATIONS_NAME/user_bc.jl contains a set of user-defined boundary conditions functions
    that can be modified as needed.

    The function defined in src/equations/EQUATIONS_NAME/user_bc.jl 
    are called by the b.c. functions defined in src/kernel/custom_bcs.jl
    within a boundary-edge loop that detects the "tag" string defined in the user-generated *.msh file.

    For example:
    If some domain boundaries of gmsh file mymesh.msh are tagged as "inflow" and "no_slip", then the user
    creating the functions in user_bc.jl must define the behavior of the unknown or its derivatives
    on those boundaries.

    ```math
    if (tag === "inflow")
        qibdy[1] = 3.0
    elseif (tag === "fix_temperature")
        qibdy[2] = 300.0
    end
    return qibdy
    ```
    where  `qibdy[i=1:nvar]` is the value unknown `i`
    
"""
# Scalar advection on a walled domain. Under DiscGal the condition is imposed
# weakly through the boundary flux (dg_boundary_ghost! in dg_fluxes.jl): a
# component written here becomes the mirror state 2*q_bc - q_interior on the
# ghost side; a component left untouched stays transmissive. The exact
# solution is zero away from the pulse, so every wall prescribes q = 0. Under
# the upwind flux an outflow wall ignores the ghost (F* = F_interior) and an
# inflow wall uses only the ghost, so this one condition serves both sides.
function user_bc_dirichlet!(q, coords, t::AbstractFloat, tag::String, qbdy::AbstractArray, nx, ny, qe, ::TOTAL)
    if tag == "wall"
        qbdy[1] = 0.0
    end
end

function user_bc_dirichlet!(q, coords, t::AbstractFloat, tag::String, qbdy::AbstractArray, nx::AbstractFloat, ny::AbstractFloat, qe, ::PERT)
    if tag == "wall"
        qbdy[1] = 0.0 - qe[1]
    end
end

function user_bc_neumann(q::AbstractArray, gradq::AbstractArray, coords, t::AbstractFloat, tag::String, inputs)
    flux = zeros(size(q,2),1)
    return flux
end

function user_bc_neumann(q::AbstractArray, gradq::AbstractArray, x::AbstractFloat, t::AbstractFloat, inputs)
    flux = zeros(size(q,2),1)
    return flux
end

function user_bc_dirichlet_gpu(q,qe,coords,t,nx,ny,qbdy,lpert)
    T = eltype(q)
    return T(0.0)
end
