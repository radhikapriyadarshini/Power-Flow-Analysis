# Power Flow Solver

Newton-Raphson based power flow solver for electrical power systems with IEEE test cases.

## Features

- Newton-Raphson power flow solution
- Support for PQ, PV, and Slack buses
- IEEE standard test systems (4, 5, 9, 14, 30, 57, 118 bus)
- Comprehensive results analysis

## Installation

```bash
pip install pandas numpy scipy pandapower
```

## Usage

### Run IEEE Test Cases
```bash
python main_pf.py
```
Select from available IEEE systems and get instant results.

### Custom Systems
```python
from solve_power_flow import PowerFlowSolver
import pandas as pd

# Define your system
bus_data = pd.DataFrame({...})  # Bus data
line_data = pd.DataFrame({...}) # Line data

# Solve
solver = PowerFlowSolver()
results = solver.solve_power_flow(bus_data, line_data)
```

## Bus Data Format
- **Bus**: Bus number
- **Type**: 1=PQ, 2=PV, 3=Slack
- **Pd, Qd**: Load (MW, MVAr)
- **Vm, Va**: Voltage magnitude (p.u.) and angle (deg)

## Line Data Format
- **From, To**: Connected buses
- **R, X, B**: Resistance, reactance, susceptance (p.u.)

## Results
- Bus voltages and power injections
- Line flows and losses
- System summary with convergence info

## Demo
```bash
python solve_power_flow.py
```
Runs built-in 5-bus demonstration system.
