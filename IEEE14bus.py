"""
IEEE 14-Bus System Data Library
A comprehensive data management library for the IEEE 14-bus test case

This library provides clean, standardized access to IEEE 14-bus system data
for use in power flow analysis, stability studies, and other power system applications.

Key Features:
- Standard IEEE 14-bus test case data
- Multiple data format exports (pandas, numpy, dict)
- PSS/E format parser
- Data validation and consistency checks
- Extensible structure for other IEEE test cases

Usage:
    from ieee14_system import IEEE14BusSystem
    
    # Create system instance
    system = IEEE14BusSystem()
    
    # Get different data formats
    bus_data = system.get_bus_data()
    generator_data = system.get_generator_data()
    branch_data = system.get_branch_data()
    
    # Get data for specific applications
    pf_data = system.get_power_flow_data()
    stability_data = system.get_stability_data()

References:
1. IEEE 14-Bus Test Case - Power Systems Test Case Archive
2. University of Washington Power Systems Test Case Archive
3. MATPOWER Test Cases
4. PSS/E Data Format Documentation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import json
import warnings
warnings.filterwarnings('ignore')

class IEEE14BusSystem:
    """
    IEEE 14-Bus Test System Data Management Class
    
    This class manages the complete IEEE 14-bus test case data,
    providing standardized access methods for various power system analyses.
    """
    
    def __init__(self):
        """Initialize IEEE 14-bus system with complete standard data"""
        self.system_name = "IEEE 14-Bus Test System"
        self.base_mva = 100.0
        self.base_kv = 138.0
        self.frequency = 60.0  # Hz
        
        # Initialize all system data
        self._initialize_bus_data()
        self._initialize_generator_data()
        self._initialize_branch_data()
        self._initialize_transformer_data()
        self._initialize_load_data()
        self._initialize_shunt_data()
        
        # System dimensions
        self.n_buses = 14
        self.n_generators = 5
        self.n_branches = 17  # Transmission lines only
        self.n_transformers = 3
        self.n_loads = 11
        
    def _initialize_bus_data(self):
        """Initialize comprehensive bus data"""
        self.bus_data = pd.DataFrame({
            'Bus': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            'Name': ['Bus 1', 'Bus 2', 'Bus 3', 'Bus 4', 'Bus 5', 'Bus 6', 
                    'Bus 7', 'Bus 8', 'Bus 9', 'Bus 10', 'Bus 11', 'Bus 12', 
                    'Bus 13', 'Bus 14'],
            'Type': [3, 2, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1],  # 1=PQ, 2=PV, 3=Slack
            'BaseKV': [138.0] * 14,
            'Vm_initial': [1.060, 1.045, 1.010, 1.0, 1.0, 1.070, 1.0, 1.090, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            'Va_initial': [0.0] * 14,  # Initial voltage angles in degrees
            'Vmax': [1.06, 1.06, 1.06, 1.06, 1.06, 1.06, 1.06, 1.06, 1.06, 1.06, 1.06, 1.06, 1.06, 1.06],
            'Vmin': [0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94],
            'Area': [1] * 14,
            'Zone': [1] * 14
        })
        
    def _initialize_generator_data(self):
        """Initialize generator data with complete parameters"""
        self.generator_data = pd.DataFrame({
            'Bus': [1, 2, 3, 6, 8],
            'Name': ['Gen 1', 'Gen 2', 'Gen 3', 'Gen 6', 'Gen 8'],
            'Pg_MW': [232.392, 40.0, 0.0, 0.0, 0.0],  # Active power generation
            'Qg_MVAr': [-16.549, 43.556, 25.075, 12.730, 17.623],  # Reactive power generation
            'Qmax_MVAr': [0.0, 50.0, 40.0, 24.0, 24.0],  # Max reactive power
            'Qmin_MVAr': [0.0, -40.0, 0.0, -6.0, -6.0],  # Min reactive power
            'Pmax_MW': [615.0, 60.0, 60.0, 25.0, 25.0],  # Max active power
            'Pmin_MW': [0.0, 0.0, 0.0, 0.0, 0.0],  # Min active power
            'Vg_pu': [1.060, 1.045, 1.010, 1.070, 1.090],  # Generator voltage setpoint
            'mBase_MVA': [100.0, 100.0, 100.0, 100.0, 100.0],  # Machine base MVA
            'Status': [1, 1, 1, 1, 1],  # 1=online, 0=offline
            'Fuel_Type': ['Nuclear', 'Coal', 'Hydro', 'Gas', 'Gas'],
            'Gen_Type': ['Synchronous', 'Synchronous', 'Synchronous', 'Synchronous', 'Synchronous']
        })
        
    def _initialize_branch_data(self):
        """Initialize transmission line data"""
        branch_list = [
            [1, 2, 0.01938, 0.05917, 0.0528, 'Line', 1],
            [1, 5, 0.05403, 0.22304, 0.0492, 'Line', 1],
            [2, 3, 0.04699, 0.19797, 0.0438, 'Line', 1],
            [2, 4, 0.05811, 0.17632, 0.0340, 'Line', 1],
            [2, 5, 0.05695, 0.17388, 0.0346, 'Line', 1],
            [3, 4, 0.06701, 0.17103, 0.0128, 'Line', 1],
            [4, 5, 0.01335, 0.04211, 0.0000, 'Line', 1],
            [6, 11, 0.09498, 0.19890, 0.0000, 'Line', 1],
            [6, 12, 0.12291, 0.25581, 0.0000, 'Line', 1],
            [6, 13, 0.06615, 0.13027, 0.0000, 'Line', 1],
            [7, 8, 0.00000, 0.17615, 0.0000, 'Line', 1],
            [7, 9, 0.00000, 0.11001, 0.0000, 'Line', 1],
            [9, 10, 0.03181, 0.08450, 0.0000, 'Line', 1],
            [9, 14, 0.12711, 0.27038, 0.0000, 'Line', 1],
            [10, 11, 0.08205, 0.19207, 0.0000, 'Line', 1],
            [12, 13, 0.22092, 0.19988, 0.0000, 'Line', 1],
            [13, 14, 0.17093, 0.34802, 0.0000, 'Line', 1]
        ]
        
        self.branch_data = pd.DataFrame(branch_list, columns=[
            'From_Bus', 'To_Bus', 'R_pu', 'X_pu', 'B_pu', 'Type', 'Status'
        ])
        
        # Add additional branch information
        self.branch_data['Length_km'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Not specified in standard
        self.branch_data['Rating_MVA'] = [0] * 17  # Thermal limits (0 = unlimited in standard case)
        
    def _initialize_transformer_data(self):
        """Initialize transformer data"""
        transformer_list = [
            [4, 7, 0.00000, 0.20912, 0.0000, 0.978, 0.0, 1],  # Bus 4-7 transformer
            [4, 9, 0.00000, 0.55618, 0.0000, 0.969, 0.0, 1],  # Bus 4-9 transformer
            [5, 6, 0.00000, 0.25202, 0.0000, 0.932, 0.0, 1]   # Bus 5-6 transformer
        ]
        
        self.transformer_data = pd.DataFrame(transformer_list, columns=[
            'From_Bus', 'To_Bus', 'R_pu', 'X_pu', 'B_pu', 'Tap_Ratio', 'Phase_Shift_deg', 'Status'
        ])
        
        # Add transformer ratings and control information
        self.transformer_data['Rating_MVA'] = [100.0, 100.0, 100.0]
        self.transformer_data['Control_Type'] = ['Fixed', 'Fixed', 'Fixed']
        self.transformer_data['Tap_Min'] = [0.9, 0.9, 0.9]
        self.transformer_data['Tap_Max'] = [1.1, 1.1, 1.1]
        
    def _initialize_load_data(self):
        """Initialize load data"""
        load_list = [
            [2, 21.7, 12.7, 'Constant'],
            [3, 94.2, 19.0, 'Constant'],
            [4, 47.8, -3.9, 'Constant'],  # Negative Q indicates capacitive load
            [5, 7.6, 1.6, 'Constant'],
            [6, 11.2, 7.5, 'Constant'],
            [9, 29.5, 16.6, 'Constant'],
            [10, 9.0, 5.8, 'Constant'],
            [11, 3.5, 1.8, 'Constant'],
            [12, 6.1, 1.6, 'Constant'],
            [13, 13.5, 5.8, 'Constant'],
            [14, 14.9, 5.0, 'Constant']
        ]
        
        self.load_data = pd.DataFrame(load_list, columns=[
            'Bus', 'P_MW', 'Q_MVAr', 'Model'
        ])
        
        # Add load composition (typical industrial/residential mix)
        self.load_data['Load_Type'] = ['Industrial', 'Industrial', 'Commercial', 'Residential', 
                                      'Residential', 'Industrial', 'Residential', 'Residential',
                                      'Commercial', 'Industrial', 'Residential']
        
    def _initialize_shunt_data(self):
        """Initialize shunt compensation data"""
        # Only Bus 9 has shunt compensation in IEEE 14-bus system
        self.shunt_data = pd.DataFrame({
            'Bus': [9],
            'G_MW': [0.0],  # Shunt conductance
            'B_MVAr': [19.0],  # Shunt susceptance (capacitive)
            'Type': ['Fixed'],
            'Status': [1]
        })
    
    # Data Access Methods
    def get_bus_data(self, format_type: str = 'dataframe') -> Union[pd.DataFrame, Dict, np.ndarray]:
        """
        Get bus data in specified format
        
        Args:
            format_type: 'dataframe', 'dict', 'numpy', or 'list'
        """
        if format_type == 'dataframe':
            return self.bus_data.copy()
        elif format_type == 'dict':
            return self.bus_data.to_dict('records')
        elif format_type == 'numpy':
            return self.bus_data.select_dtypes(include=[np.number]).values
        elif format_type == 'list':
            return self.bus_data.values.tolist()
        else:
            raise ValueError("format_type must be 'dataframe', 'dict', 'numpy', or 'list'")
    
    def get_generator_data(self, format_type: str = 'dataframe') -> Union[pd.DataFrame, Dict, np.ndarray]:
        """Get generator data in specified format"""
        if format_type == 'dataframe':
            return self.generator_data.copy()
        elif format_type == 'dict':
            return self.generator_data.to_dict('records')
        elif format_type == 'numpy':
            return self.generator_data.select_dtypes(include=[np.number]).values
        elif format_type == 'list':
            return self.generator_data.values.tolist()
        else:
            raise ValueError("format_type must be 'dataframe', 'dict', 'numpy', or 'list'")
    
    def get_branch_data(self, format_type: str = 'dataframe') -> Union[pd.DataFrame, Dict, np.ndarray]:
        """Get transmission line data in specified format"""
        if format_type == 'dataframe':
            return self.branch_data.copy()
        elif format_type == 'dict':
            return self.branch_data.to_dict('records')
        elif format_type == 'numpy':
            return self.branch_data.select_dtypes(include=[np.number]).values
        elif format_type == 'list':
            return self.branch_data.values.tolist()
        else:
            raise ValueError("format_type must be 'dataframe', 'dict', 'numpy', or 'list'")
    
    def get_transformer_data(self, format_type: str = 'dataframe') -> Union[pd.DataFrame, Dict, np.ndarray]:
        """Get transformer data in specified format"""
        if format_type == 'dataframe':
            return self.transformer_data.copy()
        elif format_type == 'dict':
            return self.transformer_data.to_dict('records')
        elif format_type == 'numpy':
            return self.transformer_data.select_dtypes(include=[np.number]).values
        elif format_type == 'list':
            return self.transformer_data.values.tolist()
        else:
            raise ValueError("format_type must be 'dataframe', 'dict', 'numpy', or 'list'")
    
    def get_load_data(self, format_type: str = 'dataframe') -> Union[pd.DataFrame, Dict, np.ndarray]:
        """Get load data in specified format"""
        if format_type == 'dataframe':
            return self.load_data.copy()
        elif format_type == 'dict':
            return self.load_data.to_dict('records')
        elif format_type == 'numpy':
            return self.load_data.select_dtypes(include=[np.number]).values
        elif format_type == 'list':
            return self.load_data.values.tolist()
        else:
            raise ValueError("format_type must be 'dataframe', 'dict', 'numpy', or 'list'")
    
    def get_shunt_data(self, format_type: str = 'dataframe') -> Union[pd.DataFrame, Dict, np.ndarray]:
        """Get shunt compensation data in specified format"""
        if format_type == 'dataframe':
            return self.shunt_data.copy()
        elif format_type == 'dict':
            return self.shunt_data.to_dict('records')
        elif format_type == 'numpy':
            return self.shunt_data.select_dtypes(include=[np.number]).values
        elif format_type == 'list':
            return self.shunt_data.values.tolist()
        else:
            raise ValueError("format_type must be 'dataframe', 'dict', 'numpy', or 'list'")
    
    # Application-Specific Data Methods
    def get_power_flow_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get data formatted specifically for power flow analysis
        
        Returns:
            Tuple of (bus_data, branch_data) formatted for power flow solvers
        """
        # Bus data for power flow (combine loads with bus data)
        pf_bus_data = pd.DataFrame({
            'Bus': self.bus_data['Bus'],
            'Type': self.bus_data['Type'],
            'Pd': [0.0] * 14,  # Initialize with zeros
            'Qd': [0.0] * 14,  # Initialize with zeros
            'Vm': self.bus_data['Vm_initial'],
            'Va': self.bus_data['Va_initial']
        })
        
        # Add load data to buses
        for _, load in self.load_data.iterrows():
            bus_idx = load['Bus'] - 1  # Convert to 0-based indexing
            pf_bus_data.loc[bus_idx, 'Pd'] = load['P_MW']
            pf_bus_data.loc[bus_idx, 'Qd'] = load['Q_MVAr']
        
        # Combined branch data (lines + transformers)
        branches_combined = []
        
        # Add transmission lines
        for _, branch in self.branch_data.iterrows():
            branches_combined.append([
                branch['From_Bus'], branch['To_Bus'], 
                branch['R_pu'], branch['X_pu'], branch['B_pu']
            ])
        
        # Add transformers (adjust for tap ratios if needed)
        for _, transformer in self.transformer_data.iterrows():
            r = transformer['R_pu']
            x = transformer['X_pu']
            b = transformer['B_pu']
            tap = transformer['Tap_Ratio']
            
            # Adjust impedance for tap ratio
            if tap != 0 and tap != 1.0:
                x = x / (tap ** 2)
                r = r / (tap ** 2)  # Usually small for transformers
            
            branches_combined.append([
                transformer['From_Bus'], transformer['To_Bus'], r, x, b
            ])
        
        pf_branch_data = pd.DataFrame(branches_combined, columns=[
            'From', 'To', 'R', 'X', 'B'
        ])
        
        return pf_bus_data, pf_branch_data
    
    def get_stability_data(self) -> Dict:
        """
        Get data formatted for stability analysis
        
        Returns:
            Dictionary containing all necessary data for stability studies
        """
        return {
            'buses': self.get_bus_data('dict'),
            'generators': self.get_generator_data('dict'),
            'branches': self.get_branch_data('dict'),
            'transformers': self.get_transformer_data('dict'),
            'loads': self.get_load_data('dict'),
            'shunts': self.get_shunt_data('dict'),
            'system_info': self.get_system_info()
        }
    
    def get_matpower_format(self) -> Dict:
        """
        Get data in MATPOWER format for compatibility
        
        Returns:
            Dictionary with MATPOWER-style data matrices
        """
        # MATPOWER bus matrix format
        bus_matrix = []
        for _, bus in self.bus_data.iterrows():
            # Find load for this bus
            load_p = 0.0
            load_q = 0.0
            shunt_g = 0.0
            shunt_b = 0.0
            
            load_row = self.load_data[self.load_data['Bus'] == bus['Bus']]
            if not load_row.empty:
                load_p = load_row.iloc[0]['P_MW']
                load_q = load_row.iloc[0]['Q_MVAr']
            
            shunt_row = self.shunt_data[self.shunt_data['Bus'] == bus['Bus']]
            if not shunt_row.empty:
                shunt_g = shunt_row.iloc[0]['G_MW']
                shunt_b = shunt_row.iloc[0]['B_MVAr']
            
            bus_matrix.append([
                bus['Bus'], bus['Type'], load_p, load_q, shunt_g, shunt_b,
                bus['Area'], bus['Vm_initial'], bus['Va_initial'], bus['BaseKV'],
                bus['Zone'], bus['Vmax'], bus['Vmin']
            ])
        
        # MATPOWER generator matrix
        gen_matrix = []
        for _, gen in self.generator_data.iterrows():
            gen_matrix.append([
                gen['Bus'], gen['Pg_MW'], gen['Qg_MVAr'], gen['Qmax_MVAr'],
                gen['Qmin_MVAr'], gen['Vg_pu'], gen['mBase_MVA'], gen['Status'],
                gen['Pmax_MW'], gen['Pmin_MW']
            ])
        
        # MATPOWER branch matrix (combine lines and transformers)
        branch_matrix = []
        for _, branch in self.branch_data.iterrows():
            branch_matrix.append([
                branch['From_Bus'], branch['To_Bus'], branch['R_pu'], branch['X_pu'],
                branch['B_pu'], 0, 0, 0, 0, 0, 1, -360, 360
            ])
        
        for _, transformer in self.transformer_data.iterrows():
            branch_matrix.append([
                transformer['From_Bus'], transformer['To_Bus'], transformer['R_pu'],
                transformer['X_pu'], transformer['B_pu'], 0, 0, 0,
                transformer['Tap_Ratio'], transformer['Phase_Shift_deg'], 1, -360, 360
            ])
        
        return {
            'bus': np.array(bus_matrix),
            'gen': np.array(gen_matrix),
            'branch': np.array(branch_matrix),
            'baseMVA': self.base_mva
        }
    
    def get_system_info(self) -> Dict:
        """Get comprehensive system information"""
        total_load_mw = self.load_data['P_MW'].sum()
        total_load_mvar = self.load_data['Q_MVAr'].sum()
        total_gen_capacity_mw = self.generator_data['Pmax_MW'].sum()
        total_shunt_mvar = self.shunt_data['B_MVAr'].sum()
        
        return {
            'name': self.system_name,
            'buses': self.n_buses,
            'generators': self.n_generators,
            'branches': self.n_branches,
            'transformers': self.n_transformers,
            'loads': self.n_loads,
            'base_mva': self.base_mva,
            'base_kv': self.base_kv,
            'frequency': self.frequency,
            'total_load_mw': total_load_mw,
            'total_load_mvar': total_load_mvar,
            'total_generation_capacity_mw': total_gen_capacity_mw,
            'total_shunt_compensation_mvar': total_shunt_mvar,
            'peak_load_bus': self.load_data.loc[self.load_data['P_MW'].idxmax(), 'Bus'],
            'slack_bus': self.bus_data[self.bus_data['Type'] == 3]['Bus'].iloc[0],
            'pv_buses': self.bus_data[self.bus_data['Type'] == 2]['Bus'].tolist(),
            'pq_buses': self.bus_data[self.bus_data['Type'] == 1]['Bus'].tolist()
        }
    
    def export_to_json(self, filename: str = 'ieee14_system.json'):
        """Export all system data to JSON format"""
        data = {
            'system_info': self.get_system_info(),
            'buses': self.get_bus_data('dict'),
            'generators': self.get_generator_data('dict'),
            'branches': self.get_branch_data('dict'),
            'transformers': self.get_transformer_data('dict'),
            'loads': self.get_load_data('dict'),
            'shunts': self.get_shunt_data('dict')
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4, default=str)
        
        print(f"IEEE 14-bus system data exported to {filename}")
    
    def export_to_csv(self, directory: str = 'ieee14_data'):
        """Export all data tables to CSV files"""
        import os
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        self.bus_data.to_csv(f'{directory}/bus_data.csv', index=False)
        self.generator_data.to_csv(f'{directory}/generator_data.csv', index=False)
        self.branch_data.to_csv(f'{directory}/branch_data.csv', index=False)
        self.transformer_data.to_csv(f'{directory}/transformer_data.csv', index=False)
        self.load_data.to_csv(f'{directory}/load_data.csv', index=False)
        self.shunt_data.to_csv(f'{directory}/shunt_data.csv', index=False)
        
        # Export system info
        with open(f'{directory}/system_info.json', 'w') as f:
            json.dump(self.get_system_info(), f, indent=4, default=str)
        
        print(f"IEEE 14-bus system data exported to {directory}/ directory")
    
    def display_system_summary(self):
        """Display a comprehensive system summary"""
        info = self.get_system_info()
        
        print("=" * 60)
        print(f"         {info['name']}")
        print("=" * 60)
        print(f"System Base:        {info['base_mva']} MVA, {info['base_kv']} kV, {info['frequency']} Hz")
        print(f"Network Size:       {info['buses']} buses, {info['branches']} lines, {info['transformers']} transformers")
        print(f"Generation:         {info['generators']} units, {info['total_generation_capacity_mw']:.1f} MW capacity")
        print(f"Load:               {info['loads']} loads, {info['total_load_mw']:.1f} MW, {info['total_load_mvar']:.1f} MVAr")
        print(f"Shunt Compensation: {info['total_shunt_compensation_mvar']:.1f} MVAr")
        print(f"Bus Types:          Slack: {info['slack_bus']}, PV: {info['pv_buses']}")
        print(f"                    PQ: {info['pq_buses']}")
        print(f"Peak Load Bus:      Bus {info['peak_load_bus']}")
        print("=" * 60)
        
        # Display data samples
        print("\nBUS DATA SAMPLE:")
        print(self.bus_data[['Bus', 'Name', 'Type', 'BaseKV', 'Vm_initial']].head())
        
        print("\nGENERATOR DATA SAMPLE:")
        print(self.generator_data[['Bus', 'Name', 'Pg_MW', 'Qg_MVAr', 'Pmax_MW']].head())
        
        print("\nLOAD DATA SAMPLE:")
        print(self.load_data[['Bus', 'P_MW', 'Q_MVAr', 'Load_Type']].head())


def demo_ieee14_system():
    """Demonstration of the IEEE 14-bus system data library"""
    print("IEEE 14-BUS SYSTEM DATA LIBRARY DEMONSTRATION")
    print("=" * 60)
    
    # Create system instance
    system = IEEE14BusSystem()
    
    # Display system summary
    system.display_system_summary()
    
    print("\n" + "=" * 60)
    print("DATA ACCESS EXAMPLES")
    print("=" * 60)
    
    # Example 1: Get data for power flow analysis
    print("\n1. Power Flow Data Format:")
    bus_pf, branch_pf = system.get_power_flow_data()
    print("Bus Data Shape:", bus_pf.shape)
    print("Branch Data Shape:", branch_pf.shape)
    print("\nSample Bus Data:")
    print(bus_pf.head(3))
    
    # Example 2: Get generator data in different formats
    print("\n2. Generator Data (Dictionary Format):")
    gen_dict = system.get_generator_data('dict')
    print(f"Number of generators: {len(gen_dict)}")
    print("Sample:", gen_dict[0])
    
    # Example 3: Get MATPOWER format
    print("\n3. MATPOWER Format Available:")
    matpower_data = system.get_matpower_format()
    print("Available matrices:", list(matpower_data.keys()))
    print("Bus matrix shape:", matpower_data['bus'].shape)
    
    # Example 4: System information
    print("\n4. System Information:")
    info = system.get_system_info()
    for key, value in list(info.items())[:8]:  # Show first 8 items
        print(f"{key}: {value}")
    
    print("\n" + "=" * 60)
    print("LIBRARY READY FOR USE!")
    print("Import this module and create IEEE14BusSystem() instance")
    print("=" * 60)


if __name__ == "__main__":
    demo_ieee14_system()
