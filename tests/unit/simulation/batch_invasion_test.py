import unittest
import json
import os

from py_pore_flow.network.network import initialize_pn
from py_pore_flow.simulation.simulation import Simulation
from tests.unit.simulation.resources.simple_network import get_simple_network


def _run_simulation(batch_invasions):
    config_path = os.path.join(os.path.dirname(__file__), "resources", "simulation_config.json")
    with open(config_path, "r") as f:
        full_config = json.load(f)

    parameters = full_config["parameters"]
    parameters["batch_invasions"] = batch_invasions
    config = full_config["config"]

    pn = get_simple_network(initial_saturation="water_filled")
    initialize_pn(pn)

    sim = Simulation(
        pn, parameters_dict=parameters, config=config,
        generate_vtu=False, generate_cas=True, setup_network=False,
    )

    pc_drainage, sw_drainage, krw_drainage, kro_drainage = sim.run_next_cycle()
    pc_imbibition, sw_imbibition, krw_imbibition, kro_imbibition = sim.run_next_cycle()

    return {
        "drainage": (pc_drainage, sw_drainage, krw_drainage, kro_drainage),
        "imbibition": (pc_imbibition, sw_imbibition, krw_imbibition, kro_imbibition),
    }


class TestBatchInvasion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = _run_simulation(batch_invasions="none")
        cls.full_batch = _run_simulation(batch_invasions="full")
        cls.partial_batch = _run_simulation(batch_invasions="partial")

    # ---- Full batch tests ----

    def test_full_drainage_produces_multiple_steps(self):
        ref_steps = len(self.reference["drainage"][0])
        batch_steps = len(self.full_batch["drainage"][0])
        print(f"Full batch drainage steps: reference={ref_steps}, batch={batch_steps}")
        self.assertGreater(batch_steps, 2, "Full batch drainage stopped too early")

    def test_full_imbibition_produces_multiple_steps(self):
        ref_steps = len(self.reference["imbibition"][0])
        batch_steps = len(self.full_batch["imbibition"][0])
        print(f"Full batch imbibition steps: reference={ref_steps}, batch={batch_steps}")
        self.assertGreater(batch_steps, 2, "Full batch imbibition stopped too early")

    def test_full_drainage_final_sw_similar(self):
        ref_sw = self.reference["drainage"][1][-1]
        batch_sw = self.full_batch["drainage"][1][-1]
        print(f"Full batch drainage final Sw: reference={ref_sw:.4f}, batch={batch_sw:.4f}")
        self.assertAlmostEqual(ref_sw, batch_sw, delta=max(0.1, abs(ref_sw) * 0.1))

    def test_full_imbibition_final_sw_similar(self):
        ref_sw = self.reference["imbibition"][1][-1]
        batch_sw = self.full_batch["imbibition"][1][-1]
        print(f"Full batch imbibition final Sw: reference={ref_sw:.4f}, batch={batch_sw:.4f}")
        self.assertAlmostEqual(ref_sw, batch_sw, delta=max(0.1, abs(ref_sw) * 0.1))

    def test_full_drainage_sw_monotonic(self):
        sw = self.full_batch["drainage"][1]
        for i in range(1, len(sw)):
            self.assertLessEqual(sw[i], sw[i - 1] + 1e-10,
                                 msg=f"Full batch drainage Sw increased at step {i}")

    def test_full_imbibition_sw_monotonic(self):
        sw = self.full_batch["imbibition"][1]
        for i in range(1, len(sw)):
            self.assertGreaterEqual(sw[i], sw[i - 1] - 1e-10,
                                    msg=f"Full batch imbibition Sw decreased at step {i}")

    def test_full_drainage_krel_range(self):
        krw = self.full_batch["drainage"][2]
        kro = self.full_batch["drainage"][3]
        for i, (w, o) in enumerate(zip(krw, kro)):
            self.assertGreaterEqual(w, -1e-10, msg=f"Krw < 0 at step {i}")
            self.assertGreaterEqual(o, -1e-10, msg=f"Kro < 0 at step {i}")
            self.assertLessEqual(w, 1.0 + 1e-10, msg=f"Krw > 1 at step {i}")
            self.assertLessEqual(o, 1.0 + 1e-10, msg=f"Kro > 1 at step {i}")

    # ---- Partial batch tests ----

    def test_partial_drainage_produces_multiple_steps(self):
        ref_steps = len(self.reference["drainage"][0])
        batch_steps = len(self.partial_batch["drainage"][0])
        print(f"Partial batch drainage steps: reference={ref_steps}, batch={batch_steps}")
        self.assertGreater(batch_steps, 2, "Partial batch drainage stopped too early")

    def test_partial_imbibition_produces_multiple_steps(self):
        ref_steps = len(self.reference["imbibition"][0])
        batch_steps = len(self.partial_batch["imbibition"][0])
        print(f"Partial batch imbibition steps: reference={ref_steps}, batch={batch_steps}")
        self.assertGreater(batch_steps, 2, "Partial batch imbibition stopped too early")

    def test_partial_drainage_final_sw_similar(self):
        ref_sw = self.reference["drainage"][1][-1]
        batch_sw = self.partial_batch["drainage"][1][-1]
        print(f"Partial batch drainage final Sw: reference={ref_sw:.4f}, batch={batch_sw:.4f}")
        self.assertAlmostEqual(ref_sw, batch_sw, delta=max(0.1, abs(ref_sw) * 0.1))

    def test_partial_imbibition_final_sw_similar(self):
        ref_sw = self.reference["imbibition"][1][-1]
        batch_sw = self.partial_batch["imbibition"][1][-1]
        print(f"Partial batch imbibition final Sw: reference={ref_sw:.4f}, batch={batch_sw:.4f}")
        # Partial batch can differ more on small networks because inter-layer
        # trapping checks trap elements before later layers reach them.
        self.assertAlmostEqual(ref_sw, batch_sw, delta=max(0.4, abs(ref_sw) * 0.4))

    def test_partial_drainage_sw_monotonic(self):
        sw = self.partial_batch["drainage"][1]
        for i in range(1, len(sw)):
            self.assertLessEqual(sw[i], sw[i - 1] + 1e-10,
                                 msg=f"Partial batch drainage Sw increased at step {i}")

    def test_partial_imbibition_sw_monotonic(self):
        sw = self.partial_batch["imbibition"][1]
        for i in range(1, len(sw)):
            self.assertGreaterEqual(sw[i], sw[i - 1] - 1e-10,
                                    msg=f"Partial batch imbibition Sw decreased at step {i}")

    def test_partial_drainage_krel_range(self):
        krw = self.partial_batch["drainage"][2]
        kro = self.partial_batch["drainage"][3]
        for i, (w, o) in enumerate(zip(krw, kro)):
            self.assertGreaterEqual(w, -1e-10, msg=f"Krw < 0 at step {i}")
            self.assertGreaterEqual(o, -1e-10, msg=f"Kro < 0 at step {i}")
            self.assertLessEqual(w, 1.0 + 1e-10, msg=f"Krw > 1 at step {i}")
            self.assertLessEqual(o, 1.0 + 1e-10, msg=f"Kro > 1 at step {i}")

    # ---- Both batch modes should invade in the right direction ----

    def test_drainage_direction_consistent(self):
        """All modes should reduce Sw during drainage (oil invades)."""
        for label, result in [("full", self.full_batch), ("partial", self.partial_batch)]:
            sw = result["drainage"][1]
            self.assertLess(sw[-1], sw[0],
                            msg=f"{label} batch drainage did not reduce Sw")


if __name__ == "__main__":
    unittest.main()
