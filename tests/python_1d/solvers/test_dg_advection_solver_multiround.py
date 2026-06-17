"""Solver-level tests for the multiround DGAdvectionSolver (RESTRUCTURE Phase 7).

Salvaged from drl-amr-1d tests/solvers/test_dg_advection_solver.py (classified
NOT-PORTED in Phase 4 because it targeted the old solver + dead API). Reworked to
the multiround API: mesh adaptation goes through refine_element/coarsen_element
(the old marks_override is gone) followed by an explicit _update_matrices() (the
primitives intentionally do not rebuild operators — the caller owns that, as the
contract does). Dropped: solve / steady_solve_* / pseudo_step / get_forcing (dead
methods); element_budget (moved to the agent's committed-budget counter); the
marks_override out-of-bounds ValueError contract (masking prevents bad indices).
Balance and full-workflow integration are covered elsewhere (test_contract_impl
cascade test + balance_test tool; tests/integration/test_multiround_smoke.py).
"""

import numpy as np
import pytest

from backends.python_1d.solvers.dg_advection_solver_multiround import DGAdvectionSolver


@pytest.fixture
def default_xelem():
    """Default 4-element grid on [-1, 1]."""
    return np.array([-1.0, -0.5, 0.0, 0.5, 1.0])


@pytest.fixture
def solver(default_xelem):
    """Default solver instance for testing (non-periodic, no balance)."""
    return DGAdvectionSolver(
        nop=4,
        xelem=default_xelem,
        max_elements=64,
        max_level=4,
        courant_max=0.1,
        icase=1,
        periodic=False,
        verbose=False,
        balance=False,
    )


class TestInit:
    """Solver initialization and derived state."""

    def test_basic_initialization(self, default_xelem):
        s = DGAdvectionSolver(nop=4, xelem=default_xelem, max_elements=64, max_level=4)
        assert s is not None

    def test_polynomial_order_stored(self, solver):
        assert solver.nop == 4
        assert solver.ngl == 5  # nop + 1

    def test_initial_element_count(self, solver, default_xelem):
        expected = len(default_xelem) - 1
        assert solver.nelem == expected
        assert len(solver.active) == expected

    def test_solution_initialized(self, solver):
        assert len(solver.q) == solver.ngl * solver.nelem
        assert solver.npoin_dg == solver.ngl * solver.nelem

    def test_solution_is_finite(self, solver):
        assert np.all(np.isfinite(solver.q))

    def test_time_initialized_to_zero(self, solver):
        assert solver.time == 0.0

    def test_timestep_is_positive(self, solver):
        assert solver.dt > 0

    def test_wave_speed_set(self, solver):
        assert hasattr(solver, "wave_speed")
        assert solver.wave_speed > 0

    def test_forest_structure_created(self, solver):
        assert solver.label_mat is not None
        assert solver.info_mat is not None
        assert len(solver.active) > 0

    def test_projection_matrices_created(self, solver):
        for attr in ("PS1", "PS2", "PG1", "PG2"):
            assert hasattr(solver, attr)

    def test_different_polynomial_orders(self, default_xelem):
        for nop in (2, 3, 4, 5):
            s = DGAdvectionSolver(nop=nop, xelem=default_xelem, max_elements=64, max_level=4)
            assert s.nop == nop
            assert s.ngl == nop + 1

    def test_different_icase_values(self, default_xelem):
        for icase in (1, 10, 14, 16):
            s = DGAdvectionSolver(nop=4, xelem=default_xelem, max_elements=64, max_level=4, icase=icase)
            assert s.icase == icase
            assert np.all(np.isfinite(s.q))


class TestMeshQuality:
    """Mesh-quality checking and state verification."""

    def test_uniform_mesh_passes(self, solver):
        is_valid, issues = solver.check_mesh_quality(np.linspace(-1, 1, 5))
        assert is_valid
        assert issues == ""

    def test_moderately_graded_mesh_passes(self, solver):
        grid = np.array([-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0])
        is_valid, _ = solver.check_mesh_quality(grid)
        assert is_valid

    def test_extreme_size_ratio_fails(self, solver):
        is_valid, issues = solver.check_mesh_quality(np.array([-1.0, 0.999, 1.0]))
        assert not is_valid
        assert "ratio" in issues.lower()

    def test_verify_state_passes_for_valid_solver(self, solver):
        solver.verify_state()  # must not raise

    def test_verify_state_fails_for_nan_solution(self, solver):
        solver.q[0] = np.nan
        with pytest.raises(ValueError, match="Invalid solution"):
            solver.verify_state()

    def test_verify_state_fails_for_inf_solution(self, solver):
        solver.q[0] = np.inf
        with pytest.raises(ValueError, match="Invalid solution"):
            solver.verify_state()


class TestQueryMethods:
    """Refinement-level and exact-solution queries."""

    def test_get_current_max_refinement_level_initial(self, solver):
        assert solver.get_current_max_refinement_level() == 0

    def test_get_active_levels_initial(self, solver):
        levels = solver.get_active_levels()
        assert len(levels) == len(solver.active)
        assert np.all(levels == 0)

    def test_get_exact_solution_shape(self, solver):
        assert len(solver.get_exact_solution()) == solver.npoin_dg

    def test_get_exact_solution_finite_and_shaped(self, solver):
        qe = solver.get_exact_solution()
        assert solver.q.shape == qe.shape
        assert np.all(np.isfinite(solver.q))
        assert np.all(np.isfinite(qe))


class TestMeshAdaptation:
    """Refine/coarsen primitives (reworked from marks_override)."""

    def test_refine_single_element(self, solver):
        initial = len(solver.active)
        assert solver.refine_element(0) is True
        assert len(solver.active) == initial + 1  # split into 2, net +1

    def test_refine_increases_max_level(self, solver):
        initial_level = solver.get_current_max_refinement_level()
        solver.refine_element(0)
        assert solver.get_current_max_refinement_level() == initial_level + 1

    def test_solution_preserved_after_refinement(self, solver):
        norm_before = np.linalg.norm(solver.q)
        solver.refine_element(0)
        norm_after = np.linalg.norm(solver.q)
        assert abs(norm_after - norm_before) / norm_before < 0.5

    def test_refinement_respects_max_level(self, solver):
        for _ in range(solver.max_level + 2):
            solver.refine_element(0)  # refines active[0]'s leaf each time; False past max
        assert solver.get_current_max_refinement_level() <= solver.max_level

    def test_coarsen_requires_sibling(self, solver):
        initial = len(solver.active)
        solver.refine_element(0)  # active[0], active[1] become a sibling pair
        assert solver.coarsen_element(0) is True
        assert len(solver.active) == initial  # merged back to parent

    def test_refine_then_rebuild_updates_matrices(self, solver):
        old_shape = solver.M.shape
        solver.refine_element(0)
        solver._update_matrices()  # caller-owned rebuild (contract.rebuild path)
        assert solver.M.shape != old_shape

    def test_state_valid_after_multiple_adaptations(self, solver):
        for i in range(5):
            if i % 2 == 0:
                solver.refine_element(0)
            else:
                solver.coarsen_element(0)
        solver._update_matrices()
        solver.verify_state()  # must not raise


class TestInitialRefinement:
    """initialize_with_refinement modes."""

    def test_fixed_refinement(self, default_xelem):
        s = DGAdvectionSolver(nop=4, xelem=default_xelem, max_elements=64, max_level=4)
        initial_nelem = s.nelem
        s.initialize_with_refinement(refinement_mode="fixed", refinement_level=2)
        assert s.nelem > initial_nelem
        levels = s.get_active_levels()
        assert np.all(levels == levels[0])  # uniform

    def test_random_refinement(self, default_xelem):
        np.random.seed(42)
        s = DGAdvectionSolver(nop=4, xelem=default_xelem, max_elements=64, max_level=4)
        s.initialize_with_refinement(refinement_mode="random", refinement_level=2, refinement_probability=0.5)
        assert s.nelem >= 4
        s.verify_state()

    def test_no_refinement_mode(self, default_xelem):
        s = DGAdvectionSolver(nop=4, xelem=default_xelem, max_elements=64, max_level=4)
        initial_nelem = s.nelem
        s.initialize_with_refinement(refinement_mode="none")
        assert s.nelem == initial_nelem


class TestReset:
    """Episode reset (hardcoded 4-element base mesh)."""

    def test_reset_restores_initial_grid(self, solver):
        solver.refine_element(0)
        solver.refine_element(1)
        solver.reset()
        assert solver.nelem == 4

    def test_reset_restores_time_zero(self, solver):
        solver.time = 1.5
        solver.reset()
        assert solver.time == 0.0

    def test_reset_with_fixed_refinement(self, solver):
        solver.reset(refinement_mode="fixed", refinement_level=1)
        assert solver.nelem > 4
        solver.verify_state()

    def test_reset_with_random_refinement(self, solver):
        np.random.seed(42)
        solver.reset(refinement_mode="random", refinement_max_level=2, refinement_probability=0.5)
        solver.verify_state()

    def test_reset_returns_solution(self, solver):
        q = solver.reset()
        assert q is not None
        assert len(q) == solver.npoin_dg
        assert np.all(np.isfinite(q))


class TestTimeStepping:
    """Low-storage RK time integration."""

    def test_step_advances_time(self, solver):
        t0 = solver.time
        solver.step()
        assert solver.time > t0

    def test_step_advances_by_dt(self, solver):
        t0 = solver.time
        solver.step()
        assert np.isclose(solver.time, t0 + solver.dt)

    def test_step_with_custom_dt(self, solver):
        t0 = solver.time
        custom_dt = solver.dt / 2
        solver.step(dt=custom_dt)
        assert np.isclose(solver.time, t0 + custom_dt)

    def test_solution_remains_finite_after_steps(self, solver):
        for _ in range(10):
            solver.step()
        assert np.all(np.isfinite(solver.q))


class TestEdgeCases:
    """Boundary configurations."""

    def test_single_element_mesh(self):
        s = DGAdvectionSolver(nop=4, xelem=np.array([-1.0, 1.0]), max_elements=64, max_level=4)
        assert s.nelem == 1
        assert np.all(np.isfinite(s.q))
        s.verify_state()

    def test_many_elements_initial(self):
        s = DGAdvectionSolver(nop=4, xelem=np.linspace(-1, 1, 17), max_elements=128, max_level=4)
        assert s.nelem == 16
        s.verify_state()

    def test_low_polynomial_order(self):
        s = DGAdvectionSolver(nop=1, xelem=np.array([-1.0, 0.0, 1.0]), max_elements=64, max_level=4)
        assert s.nop == 1
        s.verify_state()