# HyPowerFlow

HyPowerFlow algorithm accelerates power flow convergence across multiple grid scenarios by leveraging GPU parallelism, data compression, structured hyper-sparsity, and refined initial estimates. Efficient data formatting optimizes GPU utilization and eliminates redundant matrix re-creation. Structured hyper-sparsity exploits redundancy in the storage-intensive matrices to minimize memory overhead. Improved initial estimate of node voltage angles reduces iterations and accelerates convergence.

## 🏆 Award Winning Solution

HyPowerFlow is the winner of [Machine Learning for Physical Simulation Challenge - powergrid use case](https://www.codabench.org/competitions/2378/)

## 🚀 How to Run

1. **Clone the GitHub repository:**
   ```bash
   git clone https://github.com/IRT-SystemX/ml4physim_startingkit_powergrid.git
   ```

2. **Install the Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Edit `parameters.json`:**
   - Set `data_download` parameter (1 to download the data, 0 to skip)
   - Set `train_batch_size` and `eval_batch_size` as needed

4. **Set Dataset-Specific Parameters in `parameters.json`:**

   These parameters must be provided for each dataset before running:

   - `base_volt` — base voltages for buses  
   - `topo_vect_unique` — unique topologies in the dataset  
   - `Ybus_sparse_const` — base admittance matrix with no outages  
   - `PQ_unique` — PQ vector for each `topo_vect_unique`  
   - `PV_unique` — PV vector for each `topo_vect_unique`  
   - `bus_enable_flag`  
   - `bus_renumber`  

5. **Run the main script:**
   ```bash
   python main.py
   ```

## 📄 License

[Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/)
