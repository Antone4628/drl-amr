function user_inputs()
    inputs = Dict(
        #---------------------------------------------------------------------------
        # User define your inputs below: the order doesn't matter
        #---------------------------------------------------------------------------
        :tinit                => 0.0,
        :tend                 => parse(Float64, get(ENV, "INSOLVE_TEND", "20.0")),
        :ode_solver           => SSPRK33(), #ORK256(),#SSPRK33(), #SSPRK33(), #MSRK5(), #SSPRK54(),
        :Δt                   => 0.025,
        :ndiagnostics_outputs => 10,
        :diagnostics_at_times => collect(0.0:0.5:parse(Float64, get(ENV, "INSOLVE_TEND", "20.0"))),  # every 0.5, on the adapt times (amr_freq*Δt)
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
        :gmsh_filename       => "./meshes/gmsh_grids/gate2d_8x8_periodic_native.msh", #for nop=4
        # In-solve adaptation to level 2 on the native 8x8 gate mesh (item
        # 8d-ii), across both periodic wraps: no wrap mask -- a 2:1 face across
        # the wrap is a mortar like any other, and p4est's corner balance acts
        # across the wrap. The preadapt refines the two elements the pulse
        # starts on; after that the flag function re-flags from the solution
        # every :amr_freq steps, or follows a fixed script when INSOLVE_SCRIPT=1.
        # The first solve segment ends at :tinit + :amr_start_time + :amr_freq*:Δt.
        # To t = 20 the exact solution is the initial condition again.
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
