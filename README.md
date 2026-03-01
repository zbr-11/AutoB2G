# AutoB2G

AutoB2G is a building–grid co-simulation framework that integrates building-level energy control (CityLearn) with power system analysis (Pandapower), and leverages the SOCIA framework to enable natural language–driven workflow orchestration. 
It supports coordinated evaluation of demand response strategies and their impact on distribution grid performance.

---

## 🔍 Overview

This project integrates:

- 🏢 **CityLearn** – Building-level energy simulation and control
- ⚡ **Pandapower** – Distribution grid power flow simulation
- 🤖 **Reinforcement Learning** – Data-driven control strategies
- 📈 **Grid Performance Optimization** – Voltage regulation
- 🔎 **Power System Analysis** – Grid robustness evaluation

---

## 🚀 Features

- Language–driven automated simulation
- Building dataset generation
- Multi-building control via reinforcement learning
- Power flow simulation
- Voltage regulation strategy optimization
- Power system analysis
- Visualization and CSV export

---

## ▶️ Usage

### 1️⃣ Configure API Key

Before running the framework, add your API key in:

```
keys.py
```

Example:

```python
OPENAI_API_KEY = "your_api_key_here"
```

---

### 2️⃣ Provide Natural Language Instruction

Inside `run.py`, provide a natural language instruction describing the experiment.  
For example:

```
Test the centralized building control policies from both the building-side and grid-side perspectives. 
Use the SAC model trained for 10 episodes, with a quadratic reward that minimizes voltage deviations.  
Test the resulting strategy on an IEEE 33-bus distribution network with 24 buildings connected at each node. 
Add a reactive power shunt element at bus 14 that injects -1.2 MVAr.  
Compute the grid-side metrics, including bus voltage magnitudes, line loadings, 
and an N-1 contingency analysis using a voltage tolerance of 0.05 p.u. and a line loading threshold of 70%.  
Output the results as both plots and CSV files.
```

The SOCIA framework will:

1. Parse the natural language instruction
2. Retrieve from the codebase
3. Generate the corresponding simulation workflow  
4. Execute CityLearn and grid-side simulations  
5. Compute evaluation metrics  
6. Output plots and CSV files  

---

### 3️⃣ Run the Framework

Execute:

```bash
python run.py
```

---

## 📤 Output

The framework generates outputs based on user-defined instructions. Depending on the experiment setup, results may include:

- Bus voltage magnitude time-series  
- Line loading time-series
- Building control performance metrics  
- Voltage regulation performance metrics  
- N-1 contingency analysis results  

Results are automatically saved to:

```
output_grid/
```

