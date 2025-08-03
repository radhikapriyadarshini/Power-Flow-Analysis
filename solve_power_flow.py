"""
Comprehensive Power Flow Solution
A robust Newton-Raphson based power flow solver for electrical power systems

Key Features:
- Newton-Raphson method with automatic convergence control
- Support for PQ, PV, and slack bus types
- Comprehensive network analysis and results
- Flexible input format for easy integration
- Detailed convergence monitoring and error handling

References:
1. Grainger, J.J. and Stevenson, W.D. "Power System Analysis" McGraw-Hill
2. Saadat, H. "Power System Analysis" WCB/McGraw-Hill
3. Glover, J.D., Sarma, M.S., and Overbye, T.J. "Power System Analysis and Design"
"""

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class PowerFlowSolver:
    """
    Comprehensive Power Flow Solver using Newton-Raphson method
    
    This class provides a complete solution for power flow analysis in electrical
    power systems, supporting various bus types and network configurations.
    """
    
    def __init__(self, tolerance: float = 1e-6, max_iterations: int = 100):
        """
        Initialize the Power Flow Solver
        
        Args:
            tolerance: Convergence tolerance for mismatch
            max_iterations: Maximum number of iterations
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.results = {}
        self.convergence_history = []
        
    def solve_power_flow(self, bus_data: pd.DataFrame, line_data: pd.DataFrame, 
                        base_mva: float = 100.0) -> Dict:
        """
        Main power flow solution function
        
        Args:
            bus_data: DataFrame with bus information
            line_data: DataFrame with line/transformer data
            base_mva: System base MVA
            
        Returns:
            Dictionary containing complete power flow results
        """
        try:
            # Validate and process input data
            self._validate_input_data(bus_data, line_data)
            
            # Build system matrices
            n_buses = len(bus_data)
            Y_bus = self._build_ybus_matrix(line_data, n_buses)
            
            # Initialize variables
            V_mag, V_ang, bus_types = self._initialize_variables(bus_data)
            
            # Perform Newton-Raphson iterations
            converged, iterations = self._newton_raphson_solver(
                bus_data, Y_bus, V_mag, V_ang, bus_types, base_mva
            )
            
            # Calculate final results
            results = self._calculate_final_results(
                bus_data, line_data, Y_bus, V_mag, V_ang, base_mva, 
                converged, iterations
            )
            
            return results
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Power flow solution failed'
            }
    
    def _validate_input_data(self, bus_data: pd.DataFrame, line_data: pd.DataFrame):
        """Validate input data format and completeness"""
        required_bus_cols = ['Bus', 'Type', 'Pd', 'Qd', 'Vm', 'Va']
        required_line_cols = ['From', 'To', 'R', 'X', 'B']
        
        # Check bus data
        missing_bus_cols = [col for col in required_bus_cols if col not in bus_data.columns]
        if missing_bus_cols:
            raise ValueError(f"Missing bus data columns: {missing_bus_cols}")
            
        # Check line data
        missing_line_cols = [col for col in required_line_cols if col not in line_data.columns]
        if missing_line_cols:
            raise ValueError(f"Missing line data columns: {missing_line_cols}")
        
        # Validate bus types
        valid_types = [1, 2, 3]  # PQ, PV, Slack
        if not all(bus_data['Type'].isin(valid_types)):
            raise ValueError("Invalid bus types. Use 1=PQ, 2=PV, 3=Slack")
            
        # Check for exactly one slack bus
        slack_count = sum(bus_data['Type'] == 3)
        if slack_count != 1:
            raise ValueError(f"Must have exactly one slack bus, found {slack_count}")
    
    def _build_ybus_matrix(self, line_data: pd.DataFrame, n_buses: int) -> np.ndarray:
        """
        Build the bus admittance matrix (Y-bus)
        
        Args:
            line_data: Line parameters DataFrame
            n_buses: Number of buses
            
        Returns:
            Complex Y-bus matrix
        """
        Y_bus = np.zeros((n_buses, n_buses), dtype=complex)
        
        for _, line in line_data.iterrows():
            from_bus = int(line['From']) - 1  # Convert to 0-based indexing
            to_bus = int(line['To']) - 1
            
            # Line parameters
            r = line['R']
            x = line['X']
            b = line['B'] if 'B' in line else 0.0
            
            # Calculate line admittance
            z = r + 1j * x
            y = 1 / z
            
            # Shunt admittance (half on each side)
            y_shunt = 1j * b / 2
            
            # Fill Y-bus matrix
            Y_bus[from_bus, to_bus] -= y
            Y_bus[to_bus, from_bus] -= y
            Y_bus[from_bus, from_bus] += y + y_shunt
            Y_bus[to_bus, to_bus] += y + y_shunt
        
        return Y_bus
    
    def _initialize_variables(self, bus_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Initialize voltage magnitudes, angles, and bus types"""
        n_buses = len(bus_data)
        
        # Initialize voltage magnitudes and angles
        V_mag = np.ones(n_buses)
        V_ang = np.zeros(n_buses)
        bus_types = np.zeros(n_buses, dtype=int)
        
        for i, (_, bus) in enumerate(bus_data.iterrows()):
            bus_types[i] = int(bus['Type'])
            
            # Set initial values based on bus type
            if bus['Type'] == 3:  # Slack bus
                V_mag[i] = bus['Vm']
                V_ang[i] = np.radians(bus['Va'])
            elif bus['Type'] == 2:  # PV bus
                V_mag[i] = bus['Vm']
                V_ang[i] = np.radians(bus['Va']) if bus['Va'] != 0 else 0.0
            else:  # PQ bus
                V_mag[i] = bus['Vm'] if bus['Vm'] != 0 else 1.0
                V_ang[i] = np.radians(bus['Va']) if bus['Va'] != 0 else 0.0
        
        return V_mag, V_ang, bus_types
    
    def _newton_raphson_solver(self, bus_data: pd.DataFrame, Y_bus: np.ndarray, 
                              V_mag: np.ndarray, V_ang: np.ndarray, 
                              bus_types: np.ndarray, base_mva: float) -> Tuple[bool, int]:
        """
        Newton-Raphson power flow solver
        
        Returns:
            Tuple of (converged, iterations)
        """
        n_buses = len(bus_data)
        self.convergence_history = []
        
        for iteration in range(self.max_iterations):
            # Calculate power mismatches
            P_calc, Q_calc = self._calculate_power_injections(Y_bus, V_mag, V_ang)
            
            # Get specified powers
            P_spec = np.array([bus['Pd'] for _, bus in bus_data.iterrows()]) / base_mva
            Q_spec = np.array([bus['Qd'] for _, bus in bus_data.iterrows()]) / base_mva
            
            # Calculate mismatches
            delta_P = P_spec - P_calc
            delta_Q = Q_spec - Q_calc
            
            # Remove slack bus from P mismatch and PV buses from Q mismatch
            P_mismatch = []
            Q_mismatch = []
            
            for i in range(n_buses):
                if bus_types[i] != 3:  # Not slack bus
                    P_mismatch.append(delta_P[i])
                if bus_types[i] == 1:  # PQ bus only
                    Q_mismatch.append(delta_Q[i])
            
            mismatch = np.concatenate([P_mismatch, Q_mismatch])
            max_mismatch = np.max(np.abs(mismatch))
            
            self.convergence_history.append(max_mismatch)
            
            # Check convergence
            if max_mismatch < self.tolerance:
                return True, iteration + 1
            
            # Build Jacobian matrix
            J = self._build_jacobian(Y_bus, V_mag, V_ang, bus_types)
            
            # Solve for corrections
            try:
                corrections = np.linalg.solve(J, mismatch)
            except np.linalg.LinAlgError:
                raise RuntimeError("Jacobian matrix is singular - power flow diverged")
            
            # Update variables
            self._update_variables(corrections, V_mag, V_ang, bus_types)
        
        return False, self.max_iterations
    
    def _calculate_power_injections(self, Y_bus: np.ndarray, V_mag: np.ndarray, 
                                   V_ang: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate active and reactive power injections"""
        n_buses = len(V_mag)
        P = np.zeros(n_buses)
        Q = np.zeros(n_buses)
        
        for i in range(n_buses):
            for j in range(n_buses):
                angle_diff = V_ang[i] - V_ang[j]
                Y_mag = abs(Y_bus[i, j])
                Y_ang = np.angle(Y_bus[i, j])
                
                P[i] += V_mag[i] * V_mag[j] * Y_mag * np.cos(angle_diff - Y_ang)
                Q[i] += V_mag[i] * V_mag[j] * Y_mag * np.sin(angle_diff - Y_ang)
        
        return P, Q
    
    def _build_jacobian(self, Y_bus: np.ndarray, V_mag: np.ndarray, 
                       V_ang: np.ndarray, bus_types: np.ndarray) -> np.ndarray:
        """Build the Jacobian matrix for Newton-Raphson method"""
        n_buses = len(V_mag)
        
        # Count variables
        n_P = sum(1 for bt in bus_types if bt != 3)  # All except slack
        n_Q = sum(1 for bt in bus_types if bt == 1)  # Only PQ buses
        
        J = np.zeros((n_P + n_Q, n_P + n_Q))
        
        # Calculate partial derivatives
        P_calc, Q_calc = self._calculate_power_injections(Y_bus, V_mag, V_ang)
        
        # Fill Jacobian submatrices
        row_P = 0
        for i in range(n_buses):
            if bus_types[i] == 3:  # Skip slack bus
                continue
                
            col_ang = 0
            for j in range(n_buses):
                if bus_types[j] == 3:  # Skip slack bus
                    continue
                    
                if i == j:
                    # Diagonal elements dP/dθ
                    J[row_P, col_ang] = -Q_calc[i] - V_mag[i]**2 * Y_bus[i, i].imag
                else:
                    # Off-diagonal elements dP/dθ
                    angle_diff = V_ang[i] - V_ang[j]
                    Y_mag = abs(Y_bus[i, j])
                    Y_ang = np.angle(Y_bus[i, j])
                    J[row_P, col_ang] = V_mag[i] * V_mag[j] * Y_mag * np.sin(angle_diff - Y_ang)
                
                col_ang += 1
            
            # dP/dV terms
            if bus_types[i] == 1:  # PQ bus
                col_V = n_P
                for j in range(n_buses):
                    if bus_types[j] != 1:  # Skip non-PQ buses
                        continue
                        
                    if i == j:
                        # Diagonal dP/dV
                        J[row_P, col_V] = P_calc[i] / V_mag[i] + V_mag[i] * Y_bus[i, i].real
                    else:
                        # Off-diagonal dP/dV
                        angle_diff = V_ang[i] - V_ang[j]
                        Y_mag = abs(Y_bus[i, j])
                        Y_ang = np.angle(Y_bus[i, j])
                        J[row_P, col_V] = V_mag[i] * Y_mag * np.cos(angle_diff - Y_ang)
                    
                    col_V += 1
            
            row_P += 1
        
        # Q equation rows (only for PQ buses)
        row_Q = n_P
        for i in range(n_buses):
            if bus_types[i] != 1:  # Skip non-PQ buses
                continue
                
            col_ang = 0
            for j in range(n_buses):
                if bus_types[j] == 3:  # Skip slack bus
                    continue
                    
                if i == j:
                    # Diagonal dQ/dθ
                    J[row_Q, col_ang] = P_calc[i] - V_mag[i]**2 * Y_bus[i, i].real
                else:
                    # Off-diagonal dQ/dθ
                    angle_diff = V_ang[i] - V_ang[j]
                    Y_mag = abs(Y_bus[i, j])
                    Y_ang = np.angle(Y_bus[i, j])
                    J[row_Q, col_ang] = -V_mag[i] * V_mag[j] * Y_mag * np.cos(angle_diff - Y_ang)
                
                col_ang += 1
            
            # dQ/dV terms
            col_V = n_P
            for j in range(n_buses):
                if bus_types[j] != 1:  # Skip non-PQ buses
                    continue
                    
                if i == j:
                    # Diagonal dQ/dV
                    J[row_Q, col_V] = Q_calc[i] / V_mag[i] - V_mag[i] * Y_bus[i, i].imag
                else:
                    # Off-diagonal dQ/dV
                    angle_diff = V_ang[i] - V_ang[j]
                    Y_mag = abs(Y_bus[i, j])
                    Y_ang = np.angle(Y_bus[i, j])
                    J[row_Q, col_V] = V_mag[i] * Y_mag * np.sin(angle_diff - Y_ang)
                
                col_V += 1
            
            row_Q += 1
        
        return J
    
    def _update_variables(self, corrections: np.ndarray, V_mag: np.ndarray, 
                         V_ang: np.ndarray, bus_types: np.ndarray):
        """Update voltage magnitudes and angles with corrections"""
        n_buses = len(V_mag)
        n_P = sum(1 for bt in bus_types if bt != 3)
        
        # Update angles (all buses except slack)
        angle_idx = 0
        for i in range(n_buses):
            if bus_types[i] != 3:  # Not slack bus
                V_ang[i] += corrections[angle_idx]
                angle_idx += 1
        
        # Update voltage magnitudes (PQ buses only)
        voltage_idx = n_P
        for i in range(n_buses):
            if bus_types[i] == 1:  # PQ bus
                V_mag[i] += corrections[voltage_idx]
                voltage_idx += 1
    
    def _calculate_final_results(self, bus_data: pd.DataFrame, line_data: pd.DataFrame,
                               Y_bus: np.ndarray, V_mag: np.ndarray, V_ang: np.ndarray,
                               base_mva: float, converged: bool, iterations: int) -> Dict:
        """Calculate comprehensive power flow results"""
        
        # Bus results
        P_gen, Q_gen = self._calculate_power_injections(Y_bus, V_mag, V_ang)
        
        bus_results = []
        for i, (_, bus) in enumerate(bus_data.iterrows()):
            P_load = bus['Pd'] / base_mva
            Q_load = bus['Qd'] / base_mva
            
            bus_results.append({
                'Bus': int(bus['Bus']),
                'Type': bus['Type'],
                'V_mag_pu': V_mag[i],
                'V_ang_deg': np.degrees(V_ang[i]),
                'P_gen_MW': (P_gen[i] + P_load) * base_mva,
                'Q_gen_MVAr': (Q_gen[i] + Q_load) * base_mva,
                'P_load_MW': bus['Pd'],
                'Q_load_MVAr': bus['Qd']
            })
        
        # Line flow results
        line_results = []
        for _, line in line_data.iterrows():
            from_bus = int(line['From']) - 1
            to_bus = int(line['To']) - 1
            
            # Line parameters
            r = line['R']
            x = line['X']
            b = line['B'] if 'B' in line else 0.0
            
            z = r + 1j * x
            y = 1 / z
            y_shunt = 1j * b / 2
            
            # Calculate line flows
            V_from = V_mag[from_bus] * np.exp(1j * V_ang[from_bus])
            V_to = V_mag[to_bus] * np.exp(1j * V_ang[to_bus])
            
            I_from_to = (V_from - V_to) * y + V_from * y_shunt
            I_to_from = (V_to - V_from) * y + V_to * y_shunt
            
            S_from_to = V_from * np.conj(I_from_to) * base_mva
            S_to_from = V_to * np.conj(I_to_from) * base_mva
            
            # Line losses
            S_loss = S_from_to + S_to_from
            
            line_results.append({
                'From': int(line['From']),
                'To': int(line['To']),
                'P_from_MW': S_from_to.real,
                'Q_from_MVAr': S_from_to.imag,
                'P_to_MW': S_to_from.real,
                'Q_to_from_MVAr': S_to_from.imag,
                'P_loss_MW': S_loss.real,
                'Q_loss_MVAr': S_loss.imag,
                'I_from_A': abs(I_from_to),
                'I_to_A': abs(I_to_from)
            })
        
        # System summary
        total_generation = sum(result['P_gen_MW'] for result in bus_results)
        total_load = sum(result['P_load_MW'] for result in bus_results)
        total_losses = sum(result['P_loss_MW'] for result in line_results)
        
        return {
            'success': True,
            'converged': converged,
            'iterations': iterations,
            'max_mismatch': self.convergence_history[-1] if self.convergence_history else None,
            'bus_results': pd.DataFrame(bus_results),
            'line_results': pd.DataFrame(line_results),
            'system_summary': {
                'total_generation_MW': total_generation,
                'total_load_MW': total_load,
                'total_losses_MW': total_losses,
                'base_mva': base_mva
            },
            'convergence_history': self.convergence_history
        }


def create_sample_system() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a sample 5-bus power system for testing
    
    Returns:
        Tuple of (bus_data, line_data) DataFrames
    """
    # Sample bus data
    bus_data = pd.DataFrame({
        'Bus': [1, 2, 3, 4, 5],
        'Type': [3, 2, 1, 1, 1],  # 3=Slack, 2=PV, 1=PQ
        'Pd': [0, 20, 45, 40, 60],  # Load MW
        'Qd': [0, 10, 15, 5, 10],   # Load MVAr
        'Vm': [1.05, 1.02, 1.0, 1.0, 1.0],  # Voltage magnitude
        'Va': [0, 0, 0, 0, 0]       # Voltage angle
    })
    
    # Sample line data
    line_data = pd.DataFrame({
        'From': [1, 1, 2, 2, 3],
        'To': [2, 3, 3, 4, 5],
        'R': [0.02, 0.08, 0.06, 0.06, 0.04],  # Resistance p.u.
        'X': [0.06, 0.24, 0.18, 0.18, 0.12],  # Reactance p.u.
        'B': [0.03, 0.025, 0.02, 0.02, 0.015] # Susceptance p.u.
    })
    
    return bus_data, line_data


def demo_power_flow():
    """Demonstration of the power flow solver"""
    print("=" * 60)
    print("COMPREHENSIVE POWER FLOW SOLVER DEMONSTRATION")
    print("=" * 60)
    
    # Create sample system
    bus_data, line_data = create_sample_system()
    
    print("\nSample 5-Bus System:")
    print("\nBus Data:")
    print(bus_data.to_string(index=False))
    print("\nLine Data:")
    print(line_data.to_string(index=False))
    
    # Solve power flow
    solver = PowerFlowSolver(tolerance=1e-6, max_iterations=50)
    results = solver.solve_power_flow(bus_data, line_data, base_mva=100.0)
    
    if results['success']:
        print(f"\n✓ Power Flow Converged in {results['iterations']} iterations")
        print(f"✓ Maximum Mismatch: {results['max_mismatch']:.2e} p.u.")
        
        print("\nBUS RESULTS:")
        print("=" * 80)
        bus_df = results['bus_results']
        print(bus_df.round(4).to_string(index=False))
        
        print("\nLINE FLOW RESULTS:")
        print("=" * 80)
        line_df = results['line_results']
        print(line_df.round(4).to_string(index=False))
        
        print("\nSYSTEM SUMMARY:")
        print("=" * 40)
        summary = results['system_summary']
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"{key.replace('_', ' ').title()}: {value:.2f}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")
        
    else:
        print(f"\n✗ Power Flow Failed: {results['error']}")


if __name__ == "__main__":
    demo_power_flow()
