# HyPowerFlow

HyPowerFlow algorithm accelerates power flow convergence across multiple grid scenarios by leveraging GPU parallelism, data compression, structured hyper-sparsity, and refined initial estimates. Efficient
data formatting optimizes GPU utilization and eliminates redundant matrix re-creation. Structured hyper-sparsity exploits redundancy in the
storage-intensive matrices to minimize memory overhead. Improved initial estimate of node voltage angles reduces iterations and accelerates convergence.is a Python library for dealing with word pluralization.

## Award Winning Solution

HyPowerFlow is the winner of Machine Learning for Physical Simulation Challenge - powergrid use case (https://www.codabench.org/competitions/2378/)

## How to run

- Clone the GitHub repository:
```bash
git clone https://github.com/IRT-SystemX/ml4physim_startingkit_powergrid.git
```
- install the Python dependencies:
```bash
pip install -r requirements.txt
```
- set data_download paramter in paramters.json (1 to download the data and 0 to avoid downloading)
- set train_batch_size and eval_batch_size in paramters.json as desired.
- run main.py

## License

[Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License] (https://creativecommons.org/licenses/by-nc-nd/4.0/)
