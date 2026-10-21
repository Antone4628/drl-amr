function user_inputs()
    inputs = Dict(
        #---------------------------------------------------------------------------
        # User define your inputs below: the order doesn't matter
        #---------------------------------------------------------------------------
        :tinit                => 0.0,
        :tend                 => parse(Float64, get(ENV, "INSOLVE_TEND", "14.0")),
        :ode_solver           => SSPRK33(), #ORK256(),#SSPRK33(), #SSPRK33(), #MSRK5(), #SSPRK54(),
        :Δt                   => 0.025,
        :ndiagnostics_outputs => 10,
        :diagnostics_at_times => collect(0.0:0.5:parse(Float64, get(ENV, "INSOLVE_TEND", "14.0"))),  # every 0.5, on the adapt times (amr_freq*Δt)
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
        # Item 8e: in-solve adaptation, level 1 at most, on the walled 4x4 and on
        # the natively periodic channel 4x4 (WALL_MESH). The preadapt is chosen
        # by WALL_PREADAPT (initialize.jl). WALL_GRID_ONLY=1 stops after the mesh
        # builds (the face-census legs); WALL_LAMR=0 holds the preadapted mesh
        # fixed for the operator-gate probe. The exact solution leaves through
        # the right wall (centre at x = 5 at t = 10) and the top wall (y = 20 at
        # t = 12); by t = 14 the domain is empty to 1e-5 of the initial mass.
        #---------------------------------------------------------------------------
        :lpreadapt           => get(ENV, "WALL_STATIC", "0") != "1",
        :preadapt_max_level  => 1,
        :lgrid_only          => get(ENV, "WALL_GRID_ONLY", "0") == "1",
        :lamr                => get(ENV, "WALL_LAMR", "1") == "1",
        :amr_freq            => 20,
        :amr_max_level       => 1,
        :amr_start_time      => 0.0,
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
