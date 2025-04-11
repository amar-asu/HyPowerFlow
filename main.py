"""
Copyright (c) 2025,

Dr. Amarsagar Matavalam, Arizona State University (amar.sagar@asu.edu)
Shaban Ghias Satti, Arizona State University (ishabansatti@gmail.com)

This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 
International License. To view a copy of this license, visit:

    https://creativecommons.org/licenses/by-nc-nd/4.0/

You are free to share this work (copy and redistribute it in any medium or format) 
under the following terms:
- Attribution: You must give appropriate credit, provide a link to the license, 
  and indicate if changes were made.
- NonCommercial: You may not use the material for commercial purposes.
- NoDerivatives: If you remix, transform, or build upon the material, 
  you may not distribute the modified material.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR 
OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.
"""


import torch
import os
import json
from pprint import pprint
from lips.config import ConfigManager
from lips.benchmark.powergridBenchmark import PowerGridBenchmark
from lips.evaluation.powergrid_evaluation import PowerGridEvaluation
from lips.dataset.powergridDataSet import downloadPowergridDataset
from utils.compute_score import evaluate_model, compute_global_score
from augmented_simulator import *
from lips.dataset.scaler import StandardScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")

metrics_all = dict()

PARAMETERS_PATH = "parameters.json"
with open(PARAMETERS_PATH) as f:
    d = json.load(f)

data_download = d["data_download"]

if(data_download == 1):
   downloadPowergridDataset("input_data_local", "lips_idf_2023")

BENCH_CONFIG_PATH = os.path.join("configs", "benchmarks", "lips_idf_2023.ini")
DATA_PATH = os.path.join("input_data_local", "lips_idf_2023")
LOG_PATH = "lips_idf_2023_log.log"
SIM_CONFIG_PATH  = "config.ini"


sim_config_name = "DEFAULT"
config = ConfigManager(section_name=sim_config_name, path=SIM_CONFIG_PATH)


base_volt =  d["simulator_extra_parameters"]["base_volt"]
bus_enable_flag = d["simulator_extra_parameters"]["bus_enable_flag"]
bus_renumber = d["simulator_extra_parameters"]["bus_renumber"]
topo_vect_unique        = d["simulator_extra_parameters"]['topo_vect_unique']
Ybus_sparse_const = d["simulator_extra_parameters"]["Ybus_sparse_const"]
PQ_unique       = d["simulator_extra_parameters"]['PQ_unique']
PV_unique       = d["simulator_extra_parameters"]['PV_unique']

lr = d["training_config"]["lr"]
epochs =  d["training_config"]["epochs"]
train_batch_size = d["training_config"]["train_batch_size"]
eval_batch_size = d["evaluation_config"]["eval_batch_size"]
benchmark_name="Benchmark_competition"


#It is assumed that the data is already available on the device for code execution.

benchmark = PowerGridBenchmark(benchmark_name=benchmark_name,
                               benchmark_path=DATA_PATH,
                               load_data_set=True,
                               log_path=None,
                               config_path=BENCH_CONFIG_PATH,
                                load_ybus_as_sparse=True  # Ybus is registered as sparse
                              )

evaluator = PowerGridEvaluation(benchmark.config)

torch_simulator = TorchSimulator(benchmark, config, StandardScaler, device, base_volt = base_volt,\
                                 bus_enable_flag = bus_enable_flag, bus_renumber = bus_renumber,\
                                    topo_vect_unique =  topo_vect_unique, Ybus_sparse_const = Ybus_sparse_const,\
                                       PQ_unique = PQ_unique, PV_unique = PV_unique)



_ = torch_simulator.train(benchmark.train_dataset, benchmark.val_dataset, lr = lr, epochs = epochs, train_batch_size = train_batch_size,  eval_batch_size = eval_batch_size)

metrics_test = evaluate_model(benchmark, model = torch_simulator, dataset_type = 'test', batch_size = eval_batch_size)
pprint(metrics_test)
metrics_all["test"] = metrics_test["test"]

metrics_ood = evaluate_model(benchmark, model = torch_simulator, dataset_type = 'test_ood_topo', batch_size = eval_batch_size)
pprint(metrics_ood)
metrics_all["test_ood_topo"] = metrics_ood['test_ood_topo']

compute_global_score(metrics_all, benchmark.config)