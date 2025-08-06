
# IEEE 14-Bus Newton-Raphson Power Flow Analysis

This project performs **Newton-Raphson based power flow analysis** on the **IEEE 14-Bus Test System**. It is modular, well-structured, and designed for educational, research, or professional use in power system simulation and analysis.

---

## 📂 Project Structure

```

├── IEEE14bus.py          # IEEE 14-bus system data (buses, branches, transformers, generators, loads)
├── solve_power_flow.py   # Newton-Raphson power flow solver (modular, scalable)
├── main_ieee14_analysis.py               # Driver script to run power flow and print detailed results

````

---

## ⚡ How It Works

### 1. IEEE14bus.py
- Provides a class `IEEE14BusSystem` that contains all standard data for the IEEE 14-bus test case.
- Supports:
  - Bus, branch, transformer, generator, shunt, and load data
  - Exporting data in Pandas DataFrames or dictionaries
  - Ready-to-use method: `get_power_flow_data()` to return data formatted for a power flow solver

### 2. solve_power_flow.py
- Contains the `PowerFlowSolver` class implementing the **Newton-Raphson method**.
- Features:
  - Handles PQ, PV, and Slack buses
  - Builds `Ybus` matrix
  - Constructs and solves the Jacobian system
  - Returns convergence status, bus voltages, line flows, generation, mismatches, and system losses

### 3. main.py
- Loads data from `IEEE14bus.py`
- Runs Newton-Raphson power flow using `solve_power_flow.py`
- Prints:
  - Voltage magnitudes and angles
  - Generation results (P & Q)
  - Branch power flows (P & Q)
  - Mismatch history (iteration-wise)
  - System summary: total generation, load, and losses

---

## 🚀 How to Run

### ▶️ Requirements
- Python 3.7+
- `numpy`, `pandas`, `scipy`

### ▶️ Run the Power Flow
```bash
python main.py
````

---

## 📊 Example Output

```
✓ Power Flow Converged
Iterations: 6
Max Mismatch: 0.000002 p.u.

=== BUS VOLTAGES ===
 Bus  V_mag_pu  V_ang_deg
   1     1.060       0.000
   2     1.045      -4.981
   3     1.010      -12.72
   ...

=== GENERATION RESULTS ===
 Bus  P_gen_MW  Q_gen_MVAr
   1    232.39      -16.55
   2     40.00       43.56
   ...

=== SYSTEM LOSSES ===
Total Generation: 260.20 MW
Total Load:       259.42 MW
Total Losses:     0.7815 MW
```

---

## 📌 Notes

* All system data is per-unit and based on 100 MVA.
* Shunt compensation and transformer tap effects are included.
* Designed to be easily extended to IEEE 30, 57, or 118-bus systems.

---

## 📚 References

* Grainger & Stevenson, *Power System Analysis*
* Saadat, *Power System Analysis*
* MATPOWER and PSS/E standard cases

---

## 🛠️ Author

Developed with modularity and transparency in mind for power system enthusiasts, researchers, and engineers.
