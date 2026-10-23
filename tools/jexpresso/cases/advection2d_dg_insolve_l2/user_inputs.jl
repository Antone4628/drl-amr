function user_inputs()
    inputs = Dict(
        #---------------------------------------------------------------------------
        # User define your inputs below: the order doesn't matter
        #---------------------------------------------------------------------------
        :tinit                => 0.0,
        :tend                 => parse(Float64, get(ENV, "INSOLVE_TEND", "7.0")),
        :ode_solver           => SSPRK33(), #ORK256(),#SSPRK33(), #SSPRK33(), #MSRK5(), #SSPRK54(),
        :Δt                   => 0.025,
        :ndiagnostics_outputs => 10,
        :diagnostics_at_times => collect(0.0:0.25:parse(Float64, get(ENV, "INSOLVE_TEND", "7.0"))),  # every 0.25; adapts every 0.5 (amr_freq*Δt)
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
        :gmsh_filename       => "./meshes/gmsh_grids/gate2d_8x8_periodic.msh", #for nop=4
        # In-solve adaptation to level 2 on the 8x8 gate mesh (item 8a-ii), kept
        # off the periodic boundary by the two-ring wrap mask in
        # user_get_adapt_flags! (initialize.jl). The preadapt refines the two
        # elements the pulse starts on; after that the flag function re-flags
        # from the solution every :amr_freq steps, or follows a fixed script
        # when INSOLVE_SCRIPT=1. The first solve segment ends at
        # :tinit + :amr_start_time + :amr_freq*:Δt.
        :lpreadapt           => true,
        :preadapt_max_level  => 1,
        :lamr                => true,
        :amr_freq            => 20,
        :amr_max_level       => 2,
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
