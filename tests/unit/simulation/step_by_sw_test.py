"""
Tests for the sw-controlled stepping path (_step_by_sw) in MultiStepSimulation.

These tests verify that:
  1. The optimised update_pc (skipping the redundant end geometry call when
     invaded_pc == pc) produces physically correct results.
  2. sw-controlled and pc-controlled stepping converge to the same final Sw
     for simple networks.
  3. Border-element add/remove via Python sets stays consistent with the
     boolean pore.in_border / throat.in_border flags maintained by the njit
     functions.
"""
import json
import os
import unittest

import numpy as np

from py_pore_flow.network.network import initialize_pn
from py_pore_flow.simulation.simulation import Simulation
from tests.unit.simulation.resources.simple_network import get_simple_network


def _load_config() -> tuple[dict, dict]:
    config_path = os.path.join(os.path.dirname(__file__), "resources", "simulation_config.json")
    with open(config_path) as f:
        full_config = json.load(f)
    return full_config["parameters"], full_config["config"]


class TestStepBySw(unittest.TestCase):
    """Verify the sw-controlled stepping path produces correct results."""

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _run_drainage(parameters: dict, config: dict) -> tuple[list, list, list, list]:
        pn = get_simple_network(initial_saturation="water_filled")
        initialize_pn(pn)
        sim = Simulation(pn, parameters_dict=parameters, config=config,
                         generate_vtu=False, generate_cas=True, setup_network=False)
        return sim.run_next_cycle()

    # ------------------------------------------------------------------ #
    # Correctness: basic physical properties
    # ------------------------------------------------------------------ #

    def test_drainage_sw_decreases(self):
        """Water saturation must monotonically decrease during drainage."""
        params, cfg = _load_config()
        params["control_by_pc"] = False  # use _step_by_sw
        _, sw, _, _ = self._run_drainage(params, cfg)

        self.assertGreater(len(sw), 1, "Drainage produced no steps")
        # Allow equal values (plateau) but never an increase.
        for i in range(1, len(sw)):
            self.assertLessEqual(sw[i], sw[i - 1] + 1e-9,
                                 f"Sw increased from {sw[i-1]:.6f} to {sw[i]:.6f} at step {i}")

    def test_drainage_sw_in_physical_range(self):
        """All saturation values must be in [0, 1]."""
        params, cfg = _load_config()
        params["control_by_pc"] = False
        _, sw, krw, kro = self._run_drainage(params, cfg)

        sw = np.array(sw)
        krw = np.array(krw)
        kro = np.array(kro)

        self.assertTrue(np.all(sw >= -1e-9), "Negative Sw encountered")
        self.assertTrue(np.all(sw <= 1.0 + 1e-9), "Sw > 1 encountered")
        self.assertTrue(np.all(krw >= -1e-9), "Negative Krw encountered")
        self.assertTrue(np.all(kro >= -1e-9), "Negative Kro encountered")

    def test_drainage_oil_invades(self):
        """Oil must actually invade: final Sw must be below initial."""
        params, cfg = _load_config()
        params["control_by_pc"] = False
        _, sw, _, kro = self._run_drainage(params, cfg)

        self.assertLess(sw[-1], sw[0],
                        "Drainage did not reduce water saturation")
        # Oil must be flowing at final state.
        self.assertGreater(kro[-1], 0.0,
                           "Kro is zero at end of drainage (oil not percolating)")

    # ------------------------------------------------------------------ #
    # Consistency: sw-step vs pc-step converge to same final Sw
    # ------------------------------------------------------------------ #

    def test_sw_step_vs_pc_step_final_sw(self):
        """sw-controlled and pc-controlled drainage must reach the same final Sw.

        Both modes execute the same physics; they differ only in how the output
        points are sampled on the Pc–Sw curve.  The final irreducible water
        saturation is determined by network topology and contact angles, so it
        should be identical for both stepping modes.
        """
        params_base, cfg_base = _load_config()

        # sw-controlled (uses _step_by_sw)
        params_sw = dict(params_base, control_by_pc=False, drainage_sw_step=0.02)
        _, sw_sw, _, _ = self._run_drainage(params_sw, cfg_base)

        # pc-controlled (uses _step_by_pc)
        params_pc = dict(params_base, control_by_pc=True)
        _, sw_pc, _, _ = self._run_drainage(params_pc, cfg_base)

        self.assertAlmostEqual(
            sw_sw[-1], sw_pc[-1], places=4,
            msg=f"Final Sw differs: sw-step={sw_sw[-1]:.6f}, pc-step={sw_pc[-1]:.6f}",
        )

    # ------------------------------------------------------------------ #
    # Border-element consistency
    # ------------------------------------------------------------------ #

    def test_border_elements_consistent_with_in_border_flag(self):
        """pore.in_border / throat.in_border must match BorderElements after drainage.

        The njit functions set in_border[id]=True when adding an element and
        in_border[id]=False when removing it.  The Python-side BorderElements
        (now backed by sets) must reflect the same membership.
        """
        params, cfg = _load_config()
        params["control_by_pc"] = False

        pn = get_simple_network(initial_saturation="water_filled")
        initialize_pn(pn)
        sim = Simulation(pn, parameters_dict=params, config=cfg,
                         generate_vtu=False, generate_cas=True, setup_network=False)
        sim.run_next_cycle()  # drainage

        # Access the inner simulation object after the cycle.
        inner_sim = sim.step_simulation.simulation  # DrainageSimulation

        border_pore_ids = set(inner_sim.border_elements.get_pores_ids().tolist())
        border_throat_ids = set(inner_sim.border_elements.get_throats_ids().tolist())

        pore_in_border_flags = inner_sim.network.pn_core.pore.in_border
        throat_in_border_flags = inner_sim.network.pn_core.throat.in_border

        # Every pore with in_border=True must be in BorderElements and vice versa.
        expected_pore_ids = set(np.where(pore_in_border_flags)[0].tolist())
        expected_throat_ids = set(np.where(throat_in_border_flags)[0].tolist())

        self.assertEqual(
            border_pore_ids, expected_pore_ids,
            f"BorderElements pores {border_pore_ids} ≠ in_border pores {expected_pore_ids}",
        )
        self.assertEqual(
            border_throat_ids, expected_throat_ids,
            f"BorderElements throats {border_throat_ids} ≠ in_border throats {expected_throat_ids}",
        )


if __name__ == "__main__":
    unittest.main()
