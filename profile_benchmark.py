import cProfile
import pstats
import os
import sys

# Add src to sys.path
sys.path.insert(0, os.path.abspath('src'))

from py_pore_flow.__main__ import run

def profile_simulation():
    pr = cProfile.Profile()
    pr.enable()
    
    # Run the benchmark manually inside the profiler
    # run(sample_directory, manual, generate_vtu, debug, show_final_plot, enable_profiling, drainage_snapshot_pc, imbibition_snapshot_pc, second_drainage_snapshot_pc)
    run("tests/benchmarks/singlescale/4096/", False, False, False, False, False, None, None, None)
    
    pr.disable()
    stats = pstats.Stats(pr).sort_stats('tottime')
    stats.print_stats(50)

if __name__ == '__main__':
    profile_simulation()
