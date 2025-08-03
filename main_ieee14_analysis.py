"""
IEEE 14-Bus Power Flow Analysis - Main Program
==============================================

This program integrates the IEEE 14-bus system data with the Newton-Raphson
power flow solver to perform comprehensive power system analysis.

Features:
- Automatic data loading from IEEE 14-bus system
- Complete power flow solution using Newton-Raphson method
- Detailed results analysis and visualization
- Export capabilities for results
- Performance metrics and system health indicators

Author: Power Systems Analysis Tool
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
import time
from datetime import datetime
import json

# Import the power flow solver and IEEE 14-bus data
# Note: In actual implementation, these would be imported from separate files:
# from solve_power_flow import PowerFlowSolver
# from IEEE14bus import IEEE14BusSystem

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class IEEE14PowerFlowAnalysis:
    """
    Main analysis class that combines IEEE 14-bus data with power flow solver
    """
    
    def __init__(self, tolerance: float = 1e-6, max_iterations: int = 50):
        """
        Initialize the analysis system
        
        Args:
            tolerance: Convergence tolerance for power flow
            max_iterations: Maximum Newton-Raphson iterations
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.results = None
        self.analysis_time = None
        
        # Initialize system components
        print("Initializing IEEE 14-Bus Power Flow Analysis System...")
        self.system = IEEE14bus()
        self.solver = PowerFlowSolver(tolerance=tolerance, max_iterations=max_iterations)
        
        print(f"✓ System loaded: {self.system.system_name}")
        print(f"✓ Solver configured: tol={tolerance}, max_iter={max_iterations}")
    
    def run_power_flow_analysis(self, base_mva: float = None) -> Dict:
        """
        Execute complete power flow analysis
        
        Args:
            base_mva: System base MVA (uses system default if None)
            
        Returns:
            Dictionary containing all analysis results
        """
        if base_mva is None:
            base_mva = self.system.base_mva
        
        print("\n" + "="*70)
        print("EXECUTING POWER FLOW ANALYSIS")
        print("="*70)
        
        start_time = time.time()
        
        # Get data in power flow format
        print("Loading system data...")
        bus_data, branch_data = self.system.get_power_flow_data()
        
        print(f"✓ Loaded {len(bus_data)} buses and {len(branch_data)} branches")
        print(f"✓ System base MVA: {base_mva}")
        
        # Display system summary
        self._display_pre_analysis_summary(bus_data, branch_data, base_mva)
        
        # Solve power flow
        print("\nSolving power flow using Newton-Raphson method...")
        results = self.solver.solve_power_flow(bus_data, branch_data, base_mva)
        
        self.analysis_time = time.time() - start_time
        
        if results['success']:
            print(f"✓ Power flow converged in {results['iterations']} iterations")
            print(f"✓ Maximum mismatch: {results['max_mismatch']:.2e} p.u.")
            print(f"✓ Analysis completed in {self.analysis_time:.3f} seconds")
            
            # Enhance results with additional analysis
            enhanced_results = self._enhance_results(results, bus_data, branch_data, base_mva)
            self.results = enhanced_results
            
            return enhanced_results
        else:
            print(f"✗ Power flow failed: {results['error']}")
            return results
    
    def _display_pre_analysis_summary(self, bus_data: pd.DataFrame, 
                                     branch_data: pd.DataFrame, base_mva: float):
        """Display system summary before analysis"""
        total_load_mw = bus_data['Pd'].sum()
        total_load_mvar = bus_data['Qd'].sum()
        
        slack_buses = bus_data[bus_data['Type'] == 3]['Bus'].tolist()
        pv_buses = bus_data[bus_data['Type'] == 2]['Bus'].tolist()
        pq_buses = bus_data[bus_data['Type'] == 1]['Bus'].tolist()
        
        print(f"\nSystem Configuration:")
        print(f"  Total Load: {total_load_mw:.1f} MW, {total_load_mvar:.1f} MVAr")
        print(f"  Bus Types: {len(slack_buses)} Slack, {len(pv_buses)} PV, {len(pq_buses)} PQ")
        print(f"  Slack Bus: {slack_buses}")
        print(f"  PV Buses: {pv_buses}")
        print(f"  Network: {len(branch_data)} branches")
    
    def _enhance_results(self, results: Dict, bus_data: pd.DataFrame, 
                        branch_data: pd.DataFrame, base_mva: float) -> Dict:
        """Enhance basic results with additional analysis"""
        enhanced = results.copy()
        
        # Add system performance metrics
        enhanced['performance_metrics'] = self._calculate_performance_metrics(results)
        
        # Add voltage analysis
        enhanced['voltage_analysis'] = self._analyze_voltages(results['bus_results'])
        
        # Add loading analysis
        enhanced['loading_analysis'] = self._analyze_line_loading(results['line_results'])
        
        # Add loss analysis
        enhanced['loss_analysis'] = self._analyze_losses(results)
        
        # Add stability indicators
        enhanced['stability_indicators'] = self._calculate_stability_indicators(results)
        
        return enhanced
    
    def _calculate_performance_metrics(self, results: Dict) -> Dict:
        """Calculate system performance metrics"""
        summary = results['system_summary']
        
        # Efficiency metrics
        efficiency = (summary['total_load_MW'] / summary['total_generation_MW']) * 100
        loss_percentage = (summary['total_losses_MW'] / summary['total_generation_MW']) * 100
        
        # Power factor analysis
        bus_results = results['bus_results']
        total_p = bus_results['P_load_MW'].sum()
        total_q = bus_results['Q_load_MVAr'].sum()
        system_pf = total_p / np.sqrt(total_p**2 + total_q**2) if total_q != 0 else 1.0
        
        return {
            'system_efficiency_percent': efficiency,
            'loss_percentage': loss_percentage,
            'system_power_factor': system_pf,
            'convergence_rate': f"{results['iterations']} iterations",
            'solution_quality': 'Excellent' if results['max_mismatch'] < 1e-8 else 'Good'
        }
    
    def _analyze_voltages(self, bus_results: pd.DataFrame) -> Dict:
        """Analyze voltage profile"""
        voltages = bus_results['V_mag_pu'].values
        
        return {
            'min_voltage_pu': voltages.min(),
            'max_voltage_pu': voltages.max(),
            'avg_voltage_pu': voltages.mean(),
            'voltage_deviation_pu': voltages.std(),
            'min_voltage_bus': int(bus_results.loc[bus_results['V_mag_pu'].idxmin(), 'Bus']),
            'max_voltage_bus': int(bus_results.loc[bus_results['V_mag_pu'].idxmax(), 'Bus']),
            'buses_low_voltage': len(bus_results[bus_results['V_mag_pu'] < 0.95]),
            'buses_high_voltage': len(bus_results[bus_results['V_mag_pu'] > 1.05]),
            'voltage_profile_quality': 'Good' if 0.95 <= voltages.min() and voltages.max() <= 1.05 else 'Needs Attention'
        }
    
    def _analyze_line_loading(self, line_results: pd.DataFrame) -> Dict:
        """Analyze transmission line loading"""
        # For this analysis, we'll use current magnitude as loading indicator
        loadings = np.maximum(line_results['I_from_A'].values, line_results['I_to_A'].values)
        
        return {
            'max_loading_A': loadings.max(),
            'avg_loading_A': loadings.mean(),
            'most_loaded_line': f"Line {int(line_results.loc[loadings.argmax(), 'From'])}-{int(line_results.loc[loadings.argmax(), 'To'])}",
            'loading_distribution': {
                'light_loaded_lines': len(loadings[loadings < loadings.mean() * 0.5]),
                'moderately_loaded_lines': len(loadings[(loadings >= loadings.mean() * 0.5) & (loadings < loadings.mean() * 1.5)]),
                'heavily_loaded_lines': len(loadings[loadings >= loadings.mean() * 1.5])
            }
        }
    
    def _analyze_losses(self, results: Dict) -> Dict:
        """Analyze system losses"""
        line_results = results['line_results']
        total_losses_mw = line_results['P_loss_MW'].sum()
        total_losses_mvar = line_results['Q_loss_MVAr'].sum()
        
        # Find line with highest losses
        max_loss_idx = line_results['P_loss_MW'].idxmax()
        max_loss_line = f"Line {int(line_results.loc[max_loss_idx, 'From'])}-{int(line_results.loc[max_loss_idx, 'To'])}"
        max_loss_value = line_results.loc[max_loss_idx, 'P_loss_MW']
        
        return {
            'total_losses_MW': total_losses_mw,
            'total_losses_MVAr': total_losses_mvar,
            'highest_loss_line': max_loss_line,
            'highest_loss_MW': max_loss_value,
            'loss_distribution': line_results['P_loss_MW'].describe().to_dict(),
            'average_line_loss_MW': line_results['P_loss_MW'].mean()
        }
    
    def _calculate_stability_indicators(self, results: Dict) -> Dict:
        """Calculate basic stability indicators"""
        bus_results = results['bus_results']
        
        # Voltage stability margin (simplified)
        voltages = bus_results['V_mag_pu'].values
        voltage_margin = min(1.05 - voltages.max(), voltages.min() - 0.95)
        
        # Generation margin
        generators = bus_results[bus_results['P_gen_MW'] > 0]
        if not generators.empty:
            gen_utilization = generators['P_gen_MW'].sum() / 400  # Rough estimate of total capacity
        else:
            gen_utilization = 0
        
        return {
            'voltage_stability_margin_pu': voltage_margin,
            'generation_utilization_percent': gen_utilization * 100,
            'system_stability': 'Stable' if voltage_margin > 0.02 else 'Marginal',
            'critical_buses': bus_results[
                (bus_results['V_mag_pu'] < 0.95) | (bus_results['V_mag_pu'] > 1.05)
            ]['Bus'].tolist()
        }
    
    def display_comprehensive_results(self):
        """Display comprehensive analysis results"""
        if self.results is None:
            print("No results available. Run power flow analysis first.")
            return
        
        results = self.results
        
        print("\n" + "="*80)
        print("COMPREHENSIVE POWER FLOW ANALYSIS RESULTS")
        print("="*80)
        
        # Convergence Information
        print(f"\n📊 CONVERGENCE ANALYSIS")
        print(f"{'Status:':<25} {'✓ Converged' if results['converged'] else '✗ Failed'}")
        print(f"{'Iterations:':<25} {results['iterations']}")
        print(f"{'Final Mismatch:':<25} {results['max_mismatch']:.2e} p.u.")
        print(f"{'Solution Time:':<25} {self.analysis_time:.3f} seconds")
        
        # Performance Metrics
        print(f"\n⚡ SYSTEM PERFORMANCE")
        perf = results['performance_metrics']
        print(f"{'System Efficiency:':<25} {perf['system_efficiency_percent']:.2f}%")
        print(f"{'Loss Percentage:':<25} {perf['loss_percentage']:.3f}%")
        print(f"{'System Power Factor:':<25} {perf['system_power_factor']:.4f}")
        print(f"{'Solution Quality:':<25} {perf['solution_quality']}")
        
        # Voltage Analysis
        print(f"\n🔌 VOLTAGE PROFILE ANALYSIS")
        volt = results['voltage_analysis']
        print(f"{'Voltage Range:':<25} {volt['min_voltage_pu']:.4f} - {volt['max_voltage_pu']:.4f} p.u.")
        print(f"{'Average Voltage:':<25} {volt['avg_voltage_pu']:.4f} p.u.")
        print(f"{'Lowest Voltage Bus:':<25} Bus {volt['min_voltage_bus']} ({volt['min_voltage_pu']:.4f} p.u.)")
        print(f"{'Highest Voltage Bus:':<25} Bus {volt['max_voltage_bus']} ({volt['max_voltage_pu']:.4f} p.u.)")
        print(f"{'Profile Quality:':<25} {volt['voltage_profile_quality']}")
        
        # System Summary
        print(f"\n📈 SYSTEM SUMMARY")
        summary = results['system_summary']
        print(f"{'Total Generation:':<25} {summary['total_generation_MW']:.2f} MW")
        print(f"{'Total Load:':<25} {summary['total_load_MW']:.2f} MW")
        print(f"{'Total Losses:':<25} {summary['total_losses_MW']:.4f} MW")
        print(f"{'Base MVA:':<25} {summary['base_mva']:.0f} MVA")
        
        # Loss Analysis
        print(f"\n🔥 LOSS ANALYSIS")
        loss = results['loss_analysis']
        print(f"{'Total Real Losses:':<25} {loss['total_losses_MW']:.4f} MW")
        print(f"{'Total Reactive Losses:':<25} {loss['total_losses_MVAr']:.4f} MVAr")
        print(f"{'Highest Loss Line:':<25} {loss['highest_loss_line']} ({loss['highest_loss_MW']:.4f} MW)")
        print(f"{'Average Line Loss:':<25} {loss['average_line_loss_MW']:.4f} MW")
        
        # Stability Indicators
        print(f"\n🎯 STABILITY INDICATORS")
        stab = results['stability_indicators']
        print(f"{'System Status:':<25} {stab['system_stability']}")
        print(f"{'Voltage Margin:':<25} {stab['voltage_stability_margin_pu']:.4f} p.u.")
        print(f"{'Generation Utilization:':<25} {stab['generation_utilization_percent']:.1f}%")
        if stab['critical_buses']:
            print(f"{'Critical Buses:':<25} {stab['critical_buses']}")
        
    def display_detailed_bus_results(self, buses: List[int] = None):
        """Display detailed results for specific buses"""
        if self.results is None:
            print("No results available. Run power flow analysis first.")
            return
        
        bus_results = self.results['bus_results']
        
        if buses is None:
            display_buses = bus_results
            title = "ALL BUS RESULTS"
        else:
            display_buses = bus_results[bus_results['Bus'].isin(buses)]
            title = f"BUS RESULTS (Buses: {buses})"
        
        print(f"\n{title}")
        print("="*120)
        
        # Format and display
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.precision', 4)
        
        print(display_buses.to_string(index=False))
        
        # Reset pandas options
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
        pd.reset_option('display.precision')
    
    def display_detailed_line_results(self, lines: List[Tuple[int, int]] = None):
        """Display detailed results for specific lines"""
        if self.results is None:
            print("No results available. Run power flow analysis first.")
            return
        
        line_results = self.results['line_results']
        
        if lines is None:
            display_lines = line_results
            title = "ALL LINE FLOW RESULTS"
        else:
            # Filter for specific lines
            mask = line_results.apply(
                lambda row: (row['From'], row['To']) in lines or (row['To'], row['From']) in lines, 
                axis=1
            )
            display_lines = line_results[mask]
            title = f"LINE FLOW RESULTS (Lines: {lines})"
        
        print(f"\n{title}")
        print("="*140)
        
        # Format and display
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.precision', 4)
        
        print(display_lines.to_string(index=False))
        
        # Reset pandas options
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
        pd.reset_option('display.precision')
    
    def plot_results(self, save_plots: bool = False, plot_dir: str = 'plots'):
        """Generate comprehensive result plots"""
        if self.results is None:
            print("No results available. Run power flow analysis first.")
            return
        
        # Create plots directory if saving
        if save_plots:
            import os
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
        
        bus_results = self.results['bus_results']
        line_results = self.results['line_results']
        
        # Create subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('IEEE 14-Bus System Power Flow Analysis Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Voltage Profile
        axes[0, 0].bar(bus_results['Bus'], bus_results['V_mag_pu'], color='skyblue', alpha=0.7)
        axes[0, 0].axhline(y=1.05, color='red', linestyle='--', label='Upper Limit')
        axes[0, 0].axhline(y=0.95, color='red', linestyle='--', label='Lower Limit')
        axes[0, 0].axhline(y=1.0, color='green', linestyle='-', alpha=0.5, label='Nominal')
        axes[0, 0].set_xlabel('Bus Number')
        axes[0, 0].set_ylabel('Voltage Magnitude (p.u.)')
        axes[0, 0].set_title('Bus Voltage Profile')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Voltage Angles
        axes[0, 1].plot(bus_results['Bus'], bus_results['V_ang_deg'], 'o-', color='orange', markersize=6)
        axes[0, 1].set_xlabel('Bus Number')
        axes[0, 1].set_ylabel('Voltage Angle (degrees)')
        axes[0, 1].set_title('Bus Voltage Angles')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Power Generation vs Load
        x_pos = np.arange(len(bus_results))
        width = 0.35
        
        axes[1, 0].bar(x_pos - width/2, bus_results['P_gen_MW'], width, 
                      label='Generation', color='green', alpha=0.7)
        axes[1, 0].bar(x_pos + width/2, bus_results['P_load_MW'], width, 
                      label='Load', color='red', alpha=0.7)
        axes[1, 0].set_xlabel('Bus Number')
        axes[1, 0].set_ylabel('Power (MW)')
        axes[1, 0].set_title('Real Power Generation vs Load')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(bus_results['Bus'])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Line Losses
        line_labels = [f"{int(row['From'])}-{int(row['To'])}" for _, row in line_results.iterrows()]
        axes[1, 1].bar(range(len(line_results)), line_results['P_loss_MW'], 
                      color='purple', alpha=0.7)
        axes[1, 1].set_xlabel('Line')
        axes[1, 1].set_ylabel('Real Power Loss (MW)')
        axes[1, 1].set_title('Line Real Power Losses')
        axes[1, 1].set_xticks(range(len(line_results)))
        axes[1, 1].set_xticklabels(line_labels, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{plot_dir}/ieee14_power_flow_results.png', dpi=300, bbox_inches='tight')
            print(f"Plots saved to {plot_dir}/ieee14_power_flow_results.png")
        
        plt.show()
        
        # Convergence plot
        if len(self.results['convergence_history']) > 1:
            plt.figure(figsize=(10, 6))
            plt.semilogy(range(1, len(self.results['convergence_history']) + 1), 
                        self.results['convergence_history'], 'b-o', markersize=6, linewidth=2)
            plt.axhline(y=self.tolerance, color='red', linestyle='--', 
                       label=f'Tolerance ({self.tolerance})')
            plt.xlabel('Iteration')
            plt.ylabel('Maximum Mismatch (p.u.)')
            plt.title('Newton-Raphson Convergence')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if save_plots:
                plt.savefig(f'{plot_dir}/convergence_plot.png', dpi=300, bbox_inches='tight')
                print(f"Convergence plot saved to {plot_dir}/convergence_plot.png")
            
            plt.show()
    
    def export_results(self, filename: str = None, format_type: str = 'json'):
        """Export results to file"""
        if self.results is None:
            print("No results available. Run power flow analysis first.")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ieee14_powerflow_results_{timestamp}"
        
        if format_type.lower() == 'json':
            # Convert DataFrames to dictionaries for JSON serialization
            export_data = {
                'analysis_info': {
                    'timestamp': datetime.now().isoformat(),
                    'system': self.system.system_name,
                    'analysis_time_seconds': self.analysis_time,
                    'solver_tolerance': self.tolerance,
                    'max_iterations': self.max_iterations
                },
                'convergence': {
                    'converged': self.results['converged'],
                    'iterations': self.results['iterations'],
                    'max_mismatch': self.results['max_mismatch'],
                    'convergence_history': self.results['convergence_history']
                },
                'system_summary': self.results['system_summary'],
                'performance_metrics': self.results['performance_metrics'],
                'voltage_analysis': self.results['voltage_analysis'],
                'loading_analysis': self.results['loading_analysis'],
                'loss_analysis': self.results['loss_analysis'],
                'stability_indicators': self.results['stability_indicators'],
                'bus_results': self.results['bus_results'].to_dict('records'),
                'line_results': self.results['line_results'].to_dict('records')
            }
            
            with open(f"{filename}.json", 'w') as f:
                json.dump(export_data, f, indent=4, default=str)
            
            print(f"Results exported to {filename}.json")
        
        elif format_type.lower() == 'excel':
            with pd.ExcelWriter(f"{filename}.xlsx", engine='openpyxl') as writer:
                # Write different sheets
                self.results['bus_results'].to_excel(writer, sheet_name='Bus_Results', index=False)
                self.results['line_results'].to_excel(writer, sheet_name='Line_Results', index=False)
                
                # Summary sheet
                summary_data = {
                    'Metric': [],
                    'Value': []
                }
                
                # Add various metrics
                for key, value in self.results['system_summary'].items():
                    summary_data['Metric'].append(key)
                    summary_data['Value'].append(value)
                
                for key, value in self.results['performance_metrics'].items():
                    summary_data['Metric'].append(key)
                    summary_data['Value'].append(value)
                
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            print(f"Results exported to {filename}.xlsx")


def main():
    """Main execution function"""
    print("🔌 IEEE 14-BUS POWER FLOW ANALYSIS SYSTEM")
    print("="*70)
    print("Comprehensive power system analysis using Newton-Raphson method")
    print("="*70)
    
    try:
        # Initialize analysis system
        analyzer = IEEE14PowerFlowAnalysis(tolerance=1e-6, max_iterations=50)
        
        # Run power flow analysis
        results = analyzer.run_power_flow_analysis(base_mva=100.0)
        
        if results['success']:
            # Display comprehensive results
            analyzer.display_comprehensive_results()
            
            # Display detailed bus results (first 5 buses as example)
            print("\n" + "="*70)
            print("SAMPLE DETAILED RESULTS")
            print("="*70)
            analyzer.display_detailed_bus_results(buses=[1, 2, 3, 4, 5])
            
            # Display critical line results (highest loss lines)
            analyzer.display_detailed_line_results(lines=[(1, 2), (2, 3), (4, 9)])
            
            # Generate plots
            print("\nGenerating analysis plots...")
            analyzer.plot_results(save_plots=True, plot_dir='ieee14_analysis_plots')
            
            # Export results
            print("\nExporting analysis results...")
            analyzer.export_results(format_type='json')
            analyzer.export_results(format_type='excel')
            
            print("\n" + "="*70)
            print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
            print("="*70)
            print("📁 Check the generated files:")
            print("   • ieee14_analysis_plots/ - Visualization plots")
            print("   • ieee14_powerflow_results_*.json - Detailed results")
            print("   • ieee14_powerflow_results_*.xlsx - Excel summary")
            print("="*70)
            
        else:
            print(f"\n❌ Analysis failed: {results.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"\n❌ Critical error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
