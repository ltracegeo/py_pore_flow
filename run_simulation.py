import json
import sys
from pathlib import Path
import os
from datetime import datetime

# Add src to python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from py_pore_flow.network.network import (
    initialize_pn,
    Network,
)
from py_pore_flow.simulation.simulation import Simulation # Import the main Simulation class
from tests.unit.simulation.resources.simple_network import get_simple_network


def main():
    """
    Main function to run the step-by-step simulation.
    """
    # 0. Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"simulation_{timestamp}.log"
    log_file = open(log_filename, "w")

    class LoggerMultiplexer:
        def __init__(self, file, terminal):
            self.terminal = terminal
            self.file = file

        def write(self, message):
            self.terminal.write(message)
            self.file.write(message)
            self.file.flush()

        def flush(self):
            self.terminal.flush()
            self.file.flush()

    original_stdout = sys.stdout
    sys.stdout = LoggerMultiplexer(log_file, original_stdout)

    def log_and_print(msg):
        print(msg)

    # 1. Load the network
    log_and_print("Loading network...")
    pn = get_simple_network()

    # 2. Initialize the network
    log_and_print("Initializing network...")
    initialize_pn(pn)

    # 3. Wrap the network (although Simulation class might do this internally, it's good to keep consistent)
    network = Network(pn) # Keep this for consistent initialization, but it's passed as pn to Simulation

    # 4. Load simulation config
    log_and_print("Loading simulation config...")
    config_path = Path(__file__).parent / "tests" / "unit" / "simulation" / "resources" / "simulation_config.json"
    with open(config_path, 'r') as f:
        config_data = json.load(f)

    params = config_data["parameters"]
    simulation_config = config_data["config"] # This is the 'config' dictionary for the Simulation constructor
    
    # Enable conductivity logging
    conductivity_log_filename = f"conductivity_{timestamp}.tsv"
    simulation_config["log_conductivity_file"] = conductivity_log_filename
    log_and_print(f"Conductivity logging enabled. Results will be saved to {conductivity_log_filename}")


    # 5. Instantiate Simulation
    log_and_print("Initializing main simulation object...")
    # The Simulation class internally manages the cycles (drainage, imbibition, etc.)
    # We pass the raw OpenPNM network object 'pn' to it, along with parameters and config
    main_simulation = Simulation(
        pn=pn,
        parameters_dict=params,
        generate_vtu=False, # As per previous run, not explicitly requested
        generate_cas=True,  # As per previous run
        setup_network=True, # As per previous run
        config=simulation_config,
    )
    
    # 6. Run the next cycle (Drainage in this case) using the generator
    log_and_print("\nStarting simulation cycle...\n")
    
    # The run_next_cycle method returns a generator when return_generator=True
    # It will run one full cycle (e.g., drainage)
    simulation_steps_generator = main_simulation.run_next_cycle(return_generator=True, verbose=True)

    for step_output in simulation_steps_generator:
        # The generator can yield string messages (verbose output) or the final results tuple
        if isinstance(step_output, str):
            log_and_print(step_output)
        elif isinstance(step_output, tuple) and len(step_output) == 4:
            # This is likely the final Pc, Sw, Krw, Kro for the current cycle
            pc_vals, sw_vals, krw_vals, kro_vals = step_output
            log_and_print(f"\nCycle Results:")
            log_and_print(f"  Final Pc: {pc_vals[-1]:.2f}")
            log_and_print(f"  Final Sw: {sw_vals[-1]:.6f}")
            log_and_print(f"  Final Krw: {krw_vals[-1]:.6f}")
            log_and_print(f"  Final Kro: {kro_vals[-1]:.6f}")
            log_and_print(f"  Total steps in cycle: {len(pc_vals)}")
            log_and_print("=" * 40)
        else:
            log_and_print(f"Unexpected output from generator: {step_output}")

    log_and_print("\nSimulation finished.")
    
    # Restore stdout and close file
    sys.stdout = original_stdout
    log_file.close()
    print(f"\nLog saved to {log_filename}")


if __name__ == "__main__":
    main()
