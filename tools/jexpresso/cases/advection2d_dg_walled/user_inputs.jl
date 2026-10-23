function user_inputs()
    inputs = Dict(
        #---------------------------------------------------------------------------
        # User define your inputs below: the order doesn't matter
        #---------------------------------------------------------------------------
        :tend                 => 4.0,
        :ode_solver           => SSPRK33(), #ORK256(),#SSPRK33(), #SSPRK33(), #MSRK5(), #SSPRK54(),
        :Δt                   => 0.025,
        :ndiagnostics_outputs => 10,
        :diagnostics_at_times => collect(0.0:0.25:4.0),  # 17 frames: iter_1 (IC) … iter_17 (t=4); drives VTK cadence
        :lsource              => false,
        #:backend              => MetalBackend(),
        #:CL                   => NCL(), #CL() is defaults
        #:SOL_VARS_TYPE        => PERT(), #TOTAL() is default
        #---------------------------------------------------------------------------
        #Integration and quadrature properties
        #---------------------------------------------------------------------------
        :interpolation_nodes =>"lgl",
        :nop                 => 4,      # Polynomial order
        :lexact_integration  => false,  # collocated LGL — required for the entrywise operator verification
        :AD                  => DiscGal(),
        :numerical_flux      => upwind_flux(),
        #---------------------------------------------------------------------------
        # Physical parameters/constants:
        #---------------------------------------------------------------------------
        :lvisc                => false, # DG Phase 5: inviscid; the viscous path is flag-gated (rhs.jl), not μ-gated. 
        # NOTE: :lkep must stay unset/false under DiscGal — the KEP volume path
        # (_expansion_inviscid_KEP!) dispatches on AD and has no DiscGal method.
        :ivisc_equations      => [1],
        :μ                    => [0.0], #kinematic viscosity constant for θ equation
        #---------------------------------------------------------------------------
        # Mesh paramters and files:
        #---------------------------------------------------------------------------
        :lread_gmsh          => true, #If false, a 1D problem will be enforced
        :gmsh_filename       => get(ENV, "WALL_MESH", "./meshes/gmsh_grids/gate2d_4x4_walled.msh"),
        #---------------------------------------------------------------------------
        # Static 2:1 mesh for the DG mortar work: preadapt exactly one middle
        # element of the 4x4 gate mesh to level 1 before t=0 (see
        # user_get_preadapt_flags! in initialize.jl). No in-solve adaptation:
        # the mesh stays fixed so the mortar operator can be verified
        # deterministically. :lpreadapt forces :ladapt, so the adaptive mesh
        # path (p4est forest, non_conforming_facets, interp/project) is built.
        #---------------------------------------------------------------------------
        :lpreadapt           => get(ENV, "WALL_PREADAPT", "0") == "1",
        :lgrid_only          => true,   # census check only: stop after the mesh build, before any RHS
        :preadapt_max_level  => 1,
        :lamr                => false,
        #---------------------------------------------------------------------------
        # Plotting parameters
        #---------------------------------------------------------------------------
        :outformat           => "vtk",
        :output_dir          => "./output/",
        :loutput_pert        => true,  #this is only implemented for VTK for now
        #---------------------------------------------------------------------------
    ) #Dict
    #---------------------------------------------------------------------------
    # END User define your inputs below: the order doesn't matter
    #---------------------------------------------------------------------------

    return inputs
    
end
