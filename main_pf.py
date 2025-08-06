"""
Simple IEEE Power Flow Analysis using pandapower
"""

from solve_power_flow import PowerFlowSolver
import pandas as pd
import numpy as np

try:
    import pandapower.networks as pn
    SYSTEMS = {
        '4': pn.case4gs,
        '5': pn.case5, 
        '9': pn.case9,
        '14': pn.case14,
        '30': pn.case30,
        '57': pn.case57,
        '118': pn.case118
    }
except ImportError:
    print("Install pandapower: pip install pandapower")
    exit()

def pandapower_to_dataframes(net):
    """Convert pandapower network to our format"""
    # Bus data
    bus_data = pd.DataFrame({
        'Bus': net.bus.index + 1,
        'Type': 1,  # Default PQ
        'Pd': 0.0,
        'Qd': 0.0, 
        'Vm': 1.0,
        'Va': 0.0
    })
    
    # Set slack buses
    if len(net.ext_grid) > 0:
        slack_buses = net.ext_grid['bus'].values + 1
        bus_data.loc[bus_data['Bus'].isin(slack_buses), 'Type'] = 3
        bus_data.loc[bus_data['Bus'].isin(slack_buses), 'Vm'] = 1.0
    
    # Set PV buses
    if len(net.gen) > 0:
        pv_buses = net.gen['bus'].values + 1
        bus_data.loc[bus_data['Bus'].isin(pv_buses), 'Type'] = 2
    
    # Set loads
    if len(net.load) > 0:
        for _, load in net.load.iterrows():
            bus_idx = load['bus'] + 1
            bus_data.loc[bus_data['Bus'] == bus_idx, 'Pd'] = load['p_mw']
            bus_data.loc[bus_data['Bus'] == bus_idx, 'Qd'] = load['q_mvar']
    
    # Line data
    line_data = pd.DataFrame({
        'From': net.line['from_bus'] + 1,
        'To': net.line['to_bus'] + 1,
        'R': net.line['r_ohm_per_km'] * net.line['length_km'] / 100,  # Approximate p.u.
        'X': net.line['x_ohm_per_km'] * net.line['length_km'] / 100,
        'B': net.line['c_nf_per_km'] * net.line['length_km'] * 1e-6   # Approximate p.u.
    })
    
    return bus_data, line_data

def main():
    """Main function"""
    print("IEEE Power Flow Analysis")
    print("Available systems:", list(SYSTEMS.keys()))
    
    choice = input("Select system: ").strip()
    if choice not in SYSTEMS:
        print("Invalid choice!")
        return
    
    # Load pandapower network
    net = SYSTEMS[choice]()
    print(f"Loaded IEEE {choice}-bus system")
    
    # Convert to our format
    bus_data, line_data = pandapower_to_dataframes(net)
    
    # Run power flow
    solver = PowerFlowSolver()
    results = solver.solve_power_flow(bus_data, line_data)
    
    if results['success']:
        print(f"✓ Converged in {results['iterations']} iterations")
        print(f"✓ Max mismatch: {results['max_mismatch']:.2e}")
        print("\nBus Results:")
        print(results['bus_results'][['Bus', 'V_mag_pu', 'V_ang_deg']].round(4))
    else:
        print("✗ Failed:", results.get('error', 'Unknown error'))

if __name__ == "__main__":
    main()
