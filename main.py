# Copyright (c) 2024 Amarsagar Reddy Ramapuram Matavalam and Shaban Satti , Arizona State University
# 
# Licensed under the creative commons Attribution-NonCommercial-NoDerivatives 4.0 International license
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-nd/4.0/
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import torch
import os
import json
from pprint import pprint
from lips.config import ConfigManager
from lips.benchmark.powergridBenchmark import PowerGridBenchmark
from lips.evaluation.powergrid_evaluation import PowerGridEvaluation
from utils.compute_score import evaluate_model, compute_global_score
from augmented_simulator import *
from lips.dataset.scaler import StandardScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")

metrics_all = dict()

BENCH_CONFIG_PATH = os.path.join("configs", "benchmarks", "lips_idf_2023.ini")
DATA_PATH = os.path.join("input_data_local", "lips_idf_2023")
LOG_PATH = "lips_idf_2023_log.log"
SIM_CONFIG_PATH  = "config.ini"
PARAMETERS_PATH = "parameters.json"

sim_config_name = "DEFAULT"
config = ConfigManager(section_name=sim_config_name, path=SIM_CONFIG_PATH)

with open(PARAMETERS_PATH) as f:
    d = json.load(f)
base_volt =  d["simulator_extra_parameters"]["base_volt"]
bus_enable_flag = d["simulator_extra_parameters"]["bus_enable_flag"]
bus_renumber = d["simulator_extra_parameters"]["bus_renumber"]
topo_vect_unique        = d["simulator_extra_parameters"]['topo_vect_unique']
Ybus_sparse_const = d["simulator_extra_parameters"]["Ybus_sparse_const"]
PQ_unique       = d["simulator_extra_parameters"]['PQ_unique']
PV_unique       = d["simulator_extra_parameters"]['PV_unique']

lr = d["training_config"]["lr"]
epochs =  d["training_config"]["epochs"]
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



_ = torch_simulator.train(benchmark.train_dataset, benchmark.val_dataset, lr = lr, epochs = epochs)

metrics_test = evaluate_model(benchmark, model = torch_simulator, dataset_type = 'test', batch_size = 100000)
pprint(metrics_test)
metrics_all["test"] = metrics_test["test"]

metrics_ood = evaluate_model(benchmark, model = torch_simulator, dataset_type = 'test_ood_topo', batch_size = 100000)
pprint(metrics_ood)
metrics_all["test_ood_topo"] = metrics_ood['test_ood_topo']

compute_global_score(metrics_all, benchmark.config)