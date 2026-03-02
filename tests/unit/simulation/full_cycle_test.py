import unittest
import json
import os
from py_pore_flow.network import initialize_pn
from py_pore_flow.simulation.simulation import Simulation
from tests.unit.simulation.resources.simple_network import get_simple_network

class TestFullCycle(unittest.TestCase):
    def test_simple_network_full_cycle(self):
        # Load configuration
        config_path = os.path.join(os.path.dirname(__file__), "resources", "simulation_config.json")
        with open(config_path, "r") as f:
            full_config = json.load(f)
        
        parameters = full_config["parameters"]
        config = full_config["config"]
        
        # Get the simple network
        pn = get_simple_network(initial_saturation="water_filled")
        
        # Initialize the network
        initialize_pn(pn)
        
        # Create simulation
        sim = Simulation(pn, parameters_dict=parameters, config=config, 
                         generate_vtu=False, generate_cas=True, setup_network=False)
        
        # 1. Drainage Cycle
        print("\nStarting Drainage Cycle")
        pc_drainage, sw_drainage, krw_drainage, kro_drainage = sim.run_next_cycle()
        
        # Check that drainage actually happened
        self.assertLess(sw_drainage[-1], 1.0)
        self.assertGreater(len(pc_drainage), 1)
        
        # 2. Imbibition Cycle
        print("\nStarting Imbibition Cycle")
        pc_imbibition, sw_imbibition, krw_imbibition, kro_imbibition = sim.run_next_cycle()
        
        # Check that imbibition actually happened
        self.assertGreater(sw_imbibition[-1], sw_imbibition[0])
        self.assertGreater(len(pc_imbibition), 1)
        
        print(pc_imbibition)
        print()
        print(sw_imbibition)
        print()
        print(krw_imbibition)
        print()
        print(kro_imbibition)
        print()
        print(f"Final Sw: {sw_imbibition[-1]:.4f}")

if __name__ == '__main__':
    unittest.main()
