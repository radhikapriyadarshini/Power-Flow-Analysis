from IEEE14bus import IEEE14BusSystem
from solve_power_flow import PowerFlowSolver
import pandas as pd

def main():
    # Initialize system and solver
    system = IEEE14BusSystem()
    solver = PowerFlowSolver(tolerance=1e-6, max_iterations=50)

    # Get data for power flow
    bus_data, line_data = system.get_power_flow_data()
    base_mva = system.base_mva

    # Solve power flow
    results = solver.solve_power_flow(bus_data, line_data, base_mva)

    if results['success']:
        print("\n✓ Power Flow Converged")
        print(f"Iterations: {results['iterations']}")
        print(f"Max Mismatch: {results['max_mismatch']:.6f} p.u.\n")

        print("=== BUS VOLTAGES ===")
        print(results['bus_results'][['Bus', 'V_mag_pu', 'V_ang_deg']].round(4).to_string(index=False))

        print("\n=== GENERATION RESULTS ===")
        print(results['bus_results'][['Bus', 'P_gen_MW', 'Q_gen_MVAr']].round(4).to_string(index=False))

        print("\n=== BRANCH FLOWS ===")
        print(results['line_results'][['From', 'To', 'P_from_MW', 'P_to_MW', 'P_loss_MW']].round(4).to_string(index=False))

        print("\n=== SYSTEM LOSSES ===")
        summary = results['system_summary']
        print(f"Total Generation: {summary['total_generation_MW']:.2f} MW")
        print(f"Total Load:       {summary['total_load_MW']:.2f} MW")
        print(f"Total Losses:     {summary['total_losses_MW']:.4f} MW")

        print("\n=== MISMATCH HISTORY ===")
        for i, mismatch in enumerate(results['convergence_history'], 1):
            print(f" Iter {i:2d}: {mismatch:.4e}")
    else:
        print("\n✗ Power flow did not converge.")
        print(f"Error: {results['error']}")

if __name__ == "__main__":
    main()
