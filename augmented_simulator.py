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

import numpy as np
import torch
from torch import optim
from lips.benchmark.powergridBenchmark import get_env
from lips.dataset.utils.powergrid_utils import get_kwargs_simulator_scenario
import torch.nn.functional as F
from lips.augmented_simulators import AugmentedSimulator
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import torch_geometric as pytg
import torch.nn.functional as F
from scipy.sparse.sparsetools import csr_minus_csr
from torch import nn
from functools import *
import numba as nb
##################################################################################################################


class TorchSimulator(AugmentedSimulator):
    """Simulator class that allows to train and evalute your custom model

        Parameters
        ----------
        benchmark : PowerGridBenchmark
            A benchmark object passed inside the ingestion program
        config : ConfigManager
            A lips ConfigManager object allowing the access to all the parameters in `config.ini`
        scaler : StandardScaler
            A scaler class already implemented in LIPS and selected in parameters.json file
        device : torch.device
            the device on which the execution should be executed, controlled by ingestion program
        **kwargs
            The set of supplementary parameters passed through the parameters.json file and `simulator_extra_parameters` key
        
        """        
    def __init__(self,
                 benchmark = None,
                 config = None,
                 scaler = None,
                 device = None,
                 **kwargs):
        
        self.benchmark = benchmark
        self.params = config.get_options_dict() # Load parameters from config.ini file
        
        self.params.update(kwargs) # update parameters with user defined `simulator_extra_parameters` parameters
        
        self.scaler = scaler
        self.device = device
        env = get_env(get_kwargs_simulator_scenario(benchmark.config))  
        z_base = torch.tensor(np.power(env.backend.lines_or_pu_to_kv, 2) / env.backend._grid.get_sn_mva()).to(self.device)
        grid_model = env.backend._grid
        
        self.base_volt         = torch.tensor(self.params["base_volt"]).to(self.device)
        bus_enable_flag        = self.params["bus_enable_flag"]
        self.bus_enable_flag   = np.array(bus_enable_flag[0:236])!=0
        self.bus_renumber      = np.array(self.params["bus_renumber"])
        Ybus_sparse_const        = self.params["Ybus_sparse_const"]
        self.topo_vect_unique        = torch.tensor(self.params["topo_vect_unique"], dtype=torch.float32).to(self.device)
        self.PQ_unique = torch.tensor(self.params["PQ_unique"], dtype=torch.bool).to(self.device)
        self.PV_unique = torch.tensor(self.params["PV_unique"], dtype=torch.bool).to(self.device)
        layers = self.params["layers"]
        
        self.Ybus_sprase_const = []
        for i in range(0, len(Ybus_sparse_const["indptr"])):
            data =  np.array(Ybus_sparse_const["data_real"][i]) + 1j * np.array(Ybus_sparse_const["data_imag"][i])
            indices = np.array(Ybus_sparse_const["indices"][i])
            indptr = np.array(Ybus_sparse_const["indptr"][i])
            shape = Ybus_sparse_const["shape"][i]
            self.Ybus_sprase_const.append(csr_matrix((data, indices, indptr), shape=shape))
        
        self.line_params = get_line_params(grid_model, z_base[:173], self.device)
        self.traf_params= get_traf_params(grid_model, z_base[173:], self.device)

        self.line_topo_data = (torch.tensor(env.line_or_pos_topo_vect).to(self.device),\
                               torch.tensor(env.line_ex_pos_topo_vect).to(self.device),\
                               torch.tensor(env.line_or_to_subid).to(self.device),\
                                 torch.tensor(env.line_ex_to_subid).to(self.device))
        
        self.P_mis_max = torch.tensor([3000.0], device = self.device)  # 3000.0 is a random number
        self.Q_mis_max = torch.tensor([3000.0], device = self.device)
        self.P_mis_mae = torch.tensor([3000.0], device = self.device)  # 3000.0 is a random number
        self.Q_mis_mae = torch.tensor([3000.0], device = self.device)

        temp = 1.0 / np.sqrt(3.)
        self.inv_sqrt_3 = torch.tensor(temp).to(device)

        self.one = torch.tensor(1.0).to(device)

        self.YBus_Base_flat_csr, self.YBus_Base_torch, self.YBus_imag_Base_torch,\
              self.Ybus_imag_base_diag_torch, self.row_sum_Ybus_imag_base_torch = YBus_base_variables(self.Ybus_sprase_const, self.device)
        
        model_PF = TorchPF(name = "TorchPF_Model",
                        n_RNN_iter = 25, # TODO change to 2, 10 and check, change to 40 , note the time
                        n_P_iter = 4,
                        n_Q_iter = 2
                             )
        input_size, output_size = infer_input_output_size(benchmark.train_dataset)
        model_theta = Model(input_size, 
                           layers, 
                           output_size)
        
        ## It is not used in this simple implementation
        if scaler is not None:
            self.scaler = scaler()
        else:
            self.scaler = None
        self._model = None

        super().__init__(model_PF)
        self.model_PF     = model_PF
        self.model_theta  = model_theta


        self.build_model()
        self.model_PF.to(self.device)
        self.model_theta.to(self.device)
        self.torch_get_line_vals_vmap = torch.vmap(torch_get_line_vals_sample, in_dims= (0,) * 6 + (None,) * 17, out_dims=(0,) * 6)
    
    def build_model(self):
        self.model_PF.build_model()

    def train(self, train_dataset, val_dataset = None, **kwargs):
        
        params = kwargs
        
        in_data, target, train_loader = process_dataset(dataset = train_dataset,\
                                                          batch_size=params["train_batch_size"], training=True, shuffle=False)
        train_losses = []
        
        optimizer = optim.Adam(self.model_theta.parameters(), lr=params["lr"], weight_decay=5e-4)
        loss_function = nn.MSELoss()
        
        for epoch in range(params["epochs"]):
            self.model_theta.train()
            total_loss = 0
            
            for batch in train_loader:
                input_batch  = in_data[batch, :].to(self.device)
                target_batch = target[batch, :].to(self.device)

                # reset the gradient
                optimizer.zero_grad()
                # predict using your model on the current batch of data
                prediction = self.model_theta(input_batch)
                # compute the loss between prediction and real target
                loss = loss_function(prediction, target_batch)
                # compute the gradient (backward pass of back propagation algorithm)
                loss.backward()
                # update the parameters of your model
                optimizer.step()
                total_loss += loss.item() * len(input_batch)
                
            
            mean_loss = total_loss / sum(map(np.size, train_loader))

            print(f"Train Epoch: {epoch}   Avg_Loss: {mean_loss:.5f}")
            train_losses.append(mean_loss)
            
        self.predict(val_dataset, eval_batch_size=params["eval_batch_size"]) 
        print("warm_up, done")
        return train_losses
        

    def validate(self, val_loader):           
        pass    

    #@line_profiler.profile
    def predict(self, dataset, eval_batch_size=128, shuffle=False, **kwargs):
        self.dataset = dataset
        self.model_PF.eval()
        self.model_theta.eval()

        predictions_list = []
        data, YBus_flat, Islanded_buses, data_loader =\
                  process_dataset(dataset, bus_enable_flag = self.bus_enable_flag, topo_vect_unique = self.topo_vect_unique,\
                                   PQ_unique = self.PQ_unique, PV_unique = self.PV_unique, device = self.device, \
                                   batch_size = eval_batch_size, training = False, shuffle = shuffle)
                

        prod_p, prod_v, load_p, load_q, topo_vect,\
              line_status, PQ_nodes, PV_nodes, SBus_124 = data
        
        self.init_volt  = torch.zeros([prod_p.shape[0], 237], dtype=torch.float32)
        self.volt_236   = torch.zeros([prod_p.shape[0], 236], dtype=torch.complex64)
        
        with torch.no_grad():
            for batch in data_loader:
                prod_p_batch    = prod_p[batch, :].to(self.device)
                prod_v_batch    = prod_v[batch, :].to(self.device)
                load_p_batch    = load_p[batch, :].to(self.device)
                load_q_batch    = load_q[batch, :].to(self.device)

                P_mis_max = self.P_mis_max
                Q_mis_max = self.Q_mis_max
                P_mis_mae = self.P_mis_mae
                Q_mis_mae = self.Q_mis_mae

                temp = prod_p_batch.shape[0] * 124
                X0          = torch.zeros((temp,1), device = self.device)
                W_k         = torch.zeros((temp,1), device = self.device)
                A_times_Wk  = torch.zeros((temp,1), device = self.device)
                Vtheta      = torch.zeros((temp,1), device = self.device)
                Vmag        = torch.ones((temp,1), device = self.device) * 1.03
    
                topo_vect_batch      = topo_vect[batch, :]
                line_status_batch    = line_status[batch, :].to(self.device)
                PQ_nodes_batch       = PQ_nodes[batch, :].reshape(-1,1)
                PV_nodes_batch       = PV_nodes[batch, :].reshape(-1,1)
                Islanded_buses_batch = Islanded_buses[batch, :].to(self.device).reshape(-1,1)

                Vmag[Islanded_buses_batch] *= 0.0

                SBus_batch      = SBus_124[batch, :].to(self.device).reshape(-1,1) / 100
                init_volt_batch = self.init_volt[batch, :].to(self.device)
                volt_236_batch  = self.volt_236[batch, :].to(self.device)
               
                model_theta_input  = torch.cat((prod_p_batch/100, prod_v_batch/100, load_p_batch/100, load_q_batch/100, topo_vect_batch, line_status_batch), 1)
                model_theta_output = self.model_theta(model_theta_input)

                Y_blk_diag, Y_imag_blk_diag, row_sum_Ybus_delta_imag, Ybus_imag_delta_diag = \
                    get_fast_Ybus_Ydelta_spdiag_imag(self.YBus_Base_flat_csr, YBus_flat[batch, :])
                
                Y_blk_diag              = Y_blk_diag.to(self.device)
                Y_imag_blk_diag         = Y_imag_blk_diag.to(self.device)
                row_sum_Ybus_delta_imag = row_sum_Ybus_delta_imag.to(self.device)
                Ybus_imag_delta_diag    = Ybus_imag_delta_diag.to(self.device)

                prediction, P_mis_norm, Q_mis_norm, P_mis_mae, Q_mis_mae, P_mis_max_scenario, Q_mis_max_scenario\
                      = self.model_PF(Y_blk_diag, Y_imag_blk_diag, row_sum_Ybus_delta_imag, SBus_batch, PQ_nodes_batch, PV_nodes_batch,\
                                   Islanded_buses_batch, topo_vect_batch, line_status_batch, Ybus_imag_delta_diag, init_volt_batch,\
                                      model_theta_output,\
                                    self.YBus_Base_torch, self.YBus_imag_Base_torch, self.Ybus_imag_base_diag_torch,\
                                          self.row_sum_Ybus_imag_base_torch, self.bus_renumber, self.line_topo_data,\
                                              P_mis_max, Q_mis_max, P_mis_mae, Q_mis_mae, X0, W_k, A_times_Wk, Vmag, Vtheta)
                

                volt_236_batch[:,np.where(self.bus_renumber>-1)[0]] = prediction.reshape(-1,124)

                predictions = self.torch_get_line_vals_vmap(prod_p_batch, load_p_batch, torch.abs(volt_236_batch),torch.angle(volt_236_batch),topo_vect_batch,\
                                                             line_status_batch.long(), self.inv_sqrt_3, self.one, self.base_volt,\
                                                                  *self.line_topo_data, *self.line_params, *self.traf_params)
                
                predictions = torch.cat(predictions, dim=1)
                predictions_list.append(predictions)

                
        predictions = torch.cat(predictions_list, dim=0)
        
        predictions = predictions.detach().cpu().numpy()
        predictions = {"v_or": predictions[:, 0: 186], "v_ex":predictions[:, 186:372] ,"p_or": predictions[:, 372: 558], "p_ex": predictions[:, 558: 744],\
                       "a_or": predictions[:, 744: 930], "a_ex": predictions[:, 930: 1116]}
        
        adjust_GC_CPU(prod_p.numpy(), load_p.numpy(), predictions["p_ex"], predictions["p_or"])
        
        return predictions

#@line_profiler.profile
def process_dataset(dataset, bus_enable_flag = None, topo_vect_unique = None, PQ_unique = None, PV_unique = None, device = None,
                    batch_size: int=512, training: bool=False, shuffle: bool=False, normalize = None, dtype=torch.float32):
    
    if training:
        print("train")
        train_dataset  = LIPSDataset(dataset, device)
        in_data, out_data = train_dataset.get_training_data()
        samples = np.arange(dataset.size)
        try:
            train_loader = np.array_split(samples, dataset.size // batch_size)
        except ValueError:
            print("Error: Batch size is greater than data size.")
            exit()

        return in_data, out_data, train_loader
    else:
        inf_dataset  = LIPSDataset(dataset, bus_enable_flag, topo_vect_unique, PQ_unique, PV_unique, None, device)
        data            = inf_dataset.data_variables()
        YBus            = inf_dataset.get_reduced_Ybus_flattened_BATCH_enabled_124_buses()
        Islanded_buses  = inf_dataset.get_bus_islanded_flag_BATCH_124()
        samples = np.arange(dataset.size)
        inf_loader = np.array_split(samples, dataset.size // batch_size)

        return data, YBus, Islanded_buses, inf_loader


def infer_input_output_size(dataset):
    input_size = dataset.env_data["_size_x"] + dataset.env_data["_size_tau"]
    output_size = dataset.env_data["n_line"] * 2
    return input_size, output_size

##########################################################################################################################
####################### Defining Initialization model


        
class LIPSDataset():
    
    def __init__(self, dataset = None, bus_enable_flag = None, topo_vect_unique = None,\
                  PQ_unique = None, PV_unique = None, normalize = None, device = None):
        
        self.dataset   = dataset
        self.device = device
        self.bus_enable_flag = bus_enable_flag
        self.topo_vect_unique = topo_vect_unique
        self.PQ_unique = PQ_unique
        self.PV_unique = PV_unique
        self.normalize = normalize
        
    def __len__(self):       
        return self.prod_p.shape[0]
    
    #@line_profiler.profile
    def get_training_data(self):

        prod_p      = self.dataset.data['prod_p'] / 100
        prod_v      = self.dataset.data['prod_v'] / 100
        load_p      = self.dataset.data['load_p'] / 100
        load_q      = self.dataset.data['load_q'] / 100
        theta_ex    = self.dataset.data['theta_ex']
        theta_or    = self.dataset.data['theta_or']
        topo_vect   = self.dataset.data['topo_vect']
        line_status = self.dataset.data['line_status']

        input_data  = np.concatenate((prod_p, prod_v, load_p, load_q, topo_vect, line_status), axis=1)
        output_data = np.concatenate((theta_or, theta_ex), axis=1)

        return torch.from_numpy(input_data.astype(np.float32)), torch.from_numpy(output_data.astype(np.float32))
    

    def data_variables(self):
        
        self.prod_p        = torch.tensor(self.dataset.data['prod_p'], dtype = torch.float32)
        self.prod_v        = torch.tensor(self.dataset.data['prod_v'] )
        self.load_p        = torch.tensor(self.dataset.data['load_p'])
        self.load_q        = torch.tensor(self.dataset.data['load_q'], dtype = torch.float32)
        self.line_status   = torch.tensor(self.dataset.data['line_status']).to(torch.long)
        self.SBus_124      = torch.tensor(self.dataset.data['SBus'][:, self.bus_enable_flag].astype(np.complex64))
        self.topo_vect          = torch.tensor(self.dataset.data['topo_vect'], dtype=torch.float32).to(self.device)
        def distance(x, y):
            return torch.norm(x - y, p=1)

        def distances_from_a_to_all_b(a):
            return torch.vmap(lambda b: distance(a, b))(self.topo_vect_unique)

        topo_vect_differences_from_uniq = torch.vmap(distances_from_a_to_all_b)(self.topo_vect)
        closest_uniq_topo = torch.argmin(topo_vect_differences_from_uniq, dim=1)

        self.PQ_nodes = self.PQ_unique[closest_uniq_topo][:, self.bus_enable_flag]
        self.PV_nodes = self.PV_unique[closest_uniq_topo][:, self.bus_enable_flag]
    
        return self.prod_p, self.prod_v, self.load_p, self.load_q, self.topo_vect,\
              self.line_status, self.PQ_nodes, self.PV_nodes, self.SBus_124,

    #@line_profiler.profile
    def get_reduced_Ybus_flattened_BATCH_enabled_124_buses(self):
        
        self.Ybus_flat = self.dataset.data['YBus'] / 100
        bus_enable_flag_matrix_flattened = np.einsum('i,j->ij', self.bus_enable_flag, self.bus_enable_flag).flatten()
        Ybus_Batch_reduced_124 = self.Ybus_flat[:,bus_enable_flag_matrix_flattened] # assume each row is a scenario in Ybus_batch
        self.Ybus_Batch_reduced_124 = Ybus_Batch_reduced_124.tocsr()
        return Ybus_Batch_reduced_124
    #@line_profiler.profile
    def get_bus_islanded_flag_BATCH_124(self):
        Ybus_Batch_diag = get_diag_Ybus_flattened_BATCH_124_buses(self.Ybus_Batch_reduced_124)
        bus_islanded_flag_batch_124 = (np.abs(Ybus_Batch_diag.toarray())==0)
        return torch.tensor(bus_islanded_flag_batch_124)
    

class TorchFC(torch.nn.Module):
    def __init__(self, in_size, fc_hc, out_size):
        super().__init__()
        self.fc1 = nn.Linear(in_size, fc_hc[0])
        self.fc2 = nn.Linear(fc_hc[0], fc_hc[1])
        self.fc3 = nn.Linear(fc_hc[1], fc_hc[2])
        self.fc4 = nn.Linear(fc_hc[2], out_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


class Model(torch.nn.Module):
    def __init__(self, in_size, fc_hc, out_size):
        super().__init__()
        self.encoder = TorchFC(in_size, fc_hc, out_size)

    def forward(self, x_dict):
        return self.encoder(x_dict)


def YBus_base_variables(Ybus_sparse_list, device):

    YBus_Base_flat = Ybus_sparse_list[0].reshape(1, -1)
    YBus_Base_flat_csr = YBus_Base_flat.tocsr()

    Y_pytg = pytg.utils.from_scipy_sparse_matrix(
        YBus_Base_flat.reshape(124, 124).astype(np.complex64))
    YBus_Base_torch = pytg.utils.to_torch_sparse_tensor(edge_index=Y_pytg[0], edge_attr=Y_pytg[1],
                                                        size=(124, 124), is_coalesced=True)

    Y_imag_pytg = pytg.utils.from_scipy_sparse_matrix(
        YBus_Base_flat.reshape(124, 124).imag.astype(np.float32))
    YBus_imag_Base_torch = pytg.utils.to_torch_sparse_tensor(edge_index=Y_imag_pytg[0], edge_attr=Y_imag_pytg[1],
                                                             size=(124, 124), is_coalesced=True)

    row_sum_Ybus_imag_base = np.array(
        YBus_Base_flat.reshape(124, 124).sum(1).imag, dtype=np.float32)
    row_sum_Ybus_imag_base_torch = torch.tensor(row_sum_Ybus_imag_base)

    Ybus_imag_base_diag = YBus_Base_flat.reshape(
        124, 124).diagonal(0).reshape(-1, 1).imag.astype(np.float32)
    Ybus_imag_base_diag_torch = torch.tensor(Ybus_imag_base_diag)

    YBus_Base_torch = YBus_Base_torch.to(device).to_dense()
    YBus_imag_Base_torch = YBus_imag_Base_torch.to(device).to_dense()
    Ybus_imag_base_diag_torch = Ybus_imag_base_diag_torch.to(device)
    row_sum_Ybus_imag_base_torch = row_sum_Ybus_imag_base_torch.to(device)

    return YBus_Base_flat_csr, YBus_Base_torch, YBus_imag_Base_torch, Ybus_imag_base_diag_torch, row_sum_Ybus_imag_base_torch

# @line_profiler.profile


def get_J11_J22_diags(Ybus_imag_base_diag, Ybus_imag_delta_diag, row_sum_Ybus_imag_base, row_sum_Ybus_delta_imag,
                      J11_enable_flag, J22_enable_flag):
    V = 1.03
    batch_size = len(Ybus_imag_delta_diag)//124
    J11_diag = Ybus_imag_delta_diag - row_sum_Ybus_delta_imag + \
        (Ybus_imag_base_diag-row_sum_Ybus_imag_base).repeat(batch_size, 1)
    J11_diag *= -V*V
    # J11_diag = J11_diag*J11_enable_flag + (1-J11_enable_flag) = (J11_diag-1)*J11_enable_flag + 1
    J11_diag -= 1
    J11_diag *= J11_enable_flag
    J11_diag += 1

    J22_diag = Ybus_imag_delta_diag + row_sum_Ybus_delta_imag + \
        (Ybus_imag_base_diag+row_sum_Ybus_imag_base).repeat(batch_size, 1)
    J22_diag *= -V
    # J22_diag = J22_diag*J22_enable_flag + (1-J22_enable_flag) = (J22_diag-1)*J22_enable_flag + 1
    J22_diag -= 1
    J22_diag *= J22_enable_flag
    J22_diag += 1

    return J11_diag, J22_diag


# @line_profiler.profile
def get_fast_Ybus_Ydelta_spdiag_imag(YBus_base_flat_csr, batch_sparse_flat):

    batch_size, num_cols = batch_sparse_flat.shape
    YBus_base_flat_repeat_indptr = np.arange(batch_size+1)*476
    YBus_base_flat_repeat_indices = np.repeat(
        [YBus_base_flat_csr.indices], batch_size, axis=0).ravel()
    YBus_base_flat_repeat_data = np.repeat(
        [YBus_base_flat_csr.data], batch_size, axis=0).ravel()

    # ------------------- MOVE code until above to neural network model intiialization ? if batch_size is known.. --------------

    minus_res_max_nnz = 100*batch_size  # 100 is based on analysis
    idx_dtype = batch_sparse_flat._get_index_dtype((batch_sparse_flat.indptr, batch_sparse_flat.indices,
                                                    YBus_base_flat_repeat_indptr, YBus_base_flat_repeat_indices),
                                                   maxval=minus_res_max_nnz)

    minus_res_indptr = np.empty(
        batch_sparse_flat.indptr.shape, dtype=idx_dtype)
    minus_res_indices = np.empty(minus_res_max_nnz, dtype=idx_dtype)
    minus_res_data = np.empty(
        minus_res_max_nnz, dtype=batch_sparse_flat.data.dtype)

    # https://github.com/scipy/scipy/blob/afeab37e1a02a0cf09d9507fddab05795eea14fa/scipy/sparse/_compressed.py#L1367
    csr_minus_csr(batch_size, num_cols,
                  batch_sparse_flat.indptr,
                  batch_sparse_flat.indices,
                  batch_sparse_flat.data,
                  YBus_base_flat_repeat_indptr,
                  YBus_base_flat_repeat_indices,
                  YBus_base_flat_repeat_data,
                  minus_res_indptr, minus_res_indices, minus_res_data)

    minus_res_indices = minus_res_indices[:minus_res_indptr[-1]]
    minus_res_data = minus_res_data[:minus_res_indptr[-1]]

    n_matrix = np.sqrt(batch_sparse_flat.shape[1]).astype(np.int32)
    # nnz = batch_sparse_flat.data.size

    minus_res_data = minus_res_data.astype(np.complex64)
    # get_3D_indices_from_batch_sparse_flat(batch_vec, row_vec, col_vec, batch_sparse_flat, n=n_matrix)

    row_vec, col_vec = np.divmod(minus_res_indices, n_matrix)
    batch_vec = np.repeat(np.arange(batch_size), np.diff(minus_res_indptr))
    Ybus_delta_scipy_batch_sparse_blk_diag = sp.coo_array((minus_res_data, (row_vec + batch_vec * n_matrix, col_vec + batch_vec * n_matrix)),
                                                          shape=(batch_size*n_matrix, batch_size*n_matrix))
    Ybus_delta_scipy_batch_sparse_blk_diag.eliminate_zeros()

    # TODO - GPU will be better
    # row_sum_Ybus_delta = np.zeros((batch_size*n_matrix,1), dtype=np.complex128) # not really allocated
    # row_sum_Ybus_delta = row_sum_Ybus_delta*0 # will take some time and memory here..
    # np.add.at(row_sum_Ybus_delta[:,0],row_vec+batch_vec*n_matrix,minus_res_data) # TODO - GPU - slow becase no multi threading.. use GPUs??

    # this is slow.. not sure why
    row_sum_Ybus_delta_v2 = np.ones((batch_size*n_matrix, 1), dtype=np.float32)
    row_sum_Ybus_delta_v2_imag = Ybus_delta_scipy_batch_sparse_blk_diag.imag @ row_sum_Ybus_delta_v2

    Ybus_imag_delta_diag = Ybus_delta_scipy_batch_sparse_blk_diag.diagonal(
        0).reshape(-1, 1).imag

    Y_pytg = pytg.utils.from_scipy_sparse_matrix(
        Ybus_delta_scipy_batch_sparse_blk_diag)

    Y_torch_batch_sparse_blk_diag = pytg.utils.to_torch_sparse_tensor(edge_index=Y_pytg[0], edge_attr=Y_pytg[1],
                                                                      size=(
                                                                          batch_size*n_matrix, batch_size*n_matrix),
                                                                      is_coalesced=True)

    Y_pytg = pytg.utils.from_scipy_sparse_matrix(
        Ybus_delta_scipy_batch_sparse_blk_diag.imag)
    Y_imag_torch_batch_sparse_blk_diag = pytg.utils.to_torch_sparse_tensor(edge_index=Y_pytg[0], edge_attr=Y_pytg[1],
                                                                           size=(
                                                                               batch_size*n_matrix, batch_size*n_matrix),
                                                                           is_coalesced=True)

    # torch_batch_sparse_blk_diag = torch.sparse_coo_tensor(indices=torch.tensor(np.array([row_vec+batch_vec*n_matrix, col_vec+batch_vec*n_matrix])),
    #                                          values=data_vec,
    #                                          size=(batch_size*n_matrix, batch_size*n_matrix))
    return Y_torch_batch_sparse_blk_diag, Y_imag_torch_batch_sparse_blk_diag, torch.tensor(row_sum_Ybus_delta_v2_imag), torch.tensor(Ybus_imag_delta_diag)


def get_diag_Ybus_flattened_BATCH_124_buses(Ybus_Batch_reduced_124):
    Ybus_diag_index_flattened_124 = np.where(np.eye(124, 124).flatten())[0]
    Ybus_Batch_diag = Ybus_Batch_reduced_124[:, Ybus_diag_index_flattened_124]
    return Ybus_Batch_diag


class TorchFC(torch.nn.Module):
    def __init__(self, in_size, fc_hc, out_size):
        super().__init__()
        self.fc1 = nn.Linear(in_size, fc_hc[0])
        self.fc2 = nn.Linear(fc_hc[0], fc_hc[1])
        self.fc3 = nn.Linear(fc_hc[1], fc_hc[2])
        self.fc4 = nn.Linear(fc_hc[2], out_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


class Model(torch.nn.Module):

    def __init__(self, in_size, hc_size, out_size):
        super().__init__()
        self.encoder = TorchFC(in_size, hc_size, out_size)

    def forward(self, x_dict):
        return self.encoder(x_dict)


def is_positive_semidefinite(matrix):
    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return False

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(matrix)

    # Check if all eigenvalues are non-negative
    return np.all(eigenvalues >= 0)


def is_positive_symmetric_semidefinite(matrix):
    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return False

    # Check if the matrix is symmetric
    if not np.allclose(matrix, matrix.T):
        return False

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(matrix)

    # Check if all eigenvalues are non-negative
    return np.all(eigenvalues >= 0)


def get_3D_indices_from_batch_sparse_flat_test_MiniBATCH_Delta_Y(YBus_base_flat, batch_sparse_flat, Sample_select_flag):
    Delta_Y_flat = batch_sparse_flat[Sample_select_flag] - \
        sp.coo_array(np.ones((len(Sample_select_flag), 1)))@YBus_base_flat
    Delta_Y_flat.eliminate_zeros()
    if not (Delta_Y_flat.has_sorted_indices):  # VERY IMPORTANT STEP..
        Delta_Y_flat.sort_indices()
    return Delta_Y_flat


def get_3D_indices_from_batch_sparse_flat_test_MiniBATCH(batch_sparse_flat, Sample_select_flag):
    return get_3D_indices_from_batch_sparse_flat_test_v2(batch_sparse_flat[Sample_select_flag])

# @line_profiler.profile


def get_3D_indices_from_batch_sparse_flat_test(batch_sparse_flat, sparse_output_format="torch_blk_diag"):
    if not (batch_sparse_flat.has_sorted_indices):  # VERY IMPORTANT STEP..
        batch_sparse_flat.sort_indices()
    batch_size = batch_sparse_flat.shape[0]
    n_matrix = np.sqrt(batch_sparse_flat.shape[1]).astype(np.int32)
    nnz = batch_sparse_flat.data.size

    data_vec = batch_sparse_flat.data.astype(np.complex64)
    # get_3D_indices_from_batch_sparse_flat(batch_vec, row_vec, col_vec, batch_sparse_flat, n=n_matrix)

    indptr = batch_sparse_flat.indptr
    indices = batch_sparse_flat.indices

    row_vec, col_vec = np.divmod(indices, n_matrix)
    batch_vec = np.repeat(np.arange(batch_size), np.diff(indptr))

    if sparse_output_format == "torch_3D":
        torch_batch_sparse_3D = torch.sparse_coo_tensor(indices=torch.tensor(np.array([batch_vec, row_vec, col_vec])),
                                                        values=data_vec,
                                                        size=(batch_size, n_matrix, n_matrix))
        return torch_batch_sparse_3D
    if sparse_output_format == "torch_blk_diag":
        torch_batch_sparse_blk_diag = torch.sparse_coo_tensor(indices=torch.tensor(np.array([row_vec+batch_vec*n_matrix, col_vec+batch_vec*n_matrix])),
                                                              values=data_vec,
                                                              size=(batch_size*n_matrix, batch_size*n_matrix))
        return torch_batch_sparse_blk_diag
    if sparse_output_format == "pytg":
        scipy_batch_sparse_blk_diag = sp.coo_array((data_vec, (row_vec+batch_vec*n_matrix, col_vec+batch_vec*n_matrix)),
                                                   shape=(batch_size*n_matrix, batch_size*n_matrix))
        return pytg.utils.from_scipy_sparse_matrix(scipy_batch_sparse_blk_diag)
    if sparse_output_format == "scipy":
        scipy_batch_sparse_blk_diag = sp.coo_array((data_vec, (row_vec+batch_vec*n_matrix, col_vec+batch_vec*n_matrix)),
                                                   shape=(batch_size*n_matrix, batch_size*n_matrix))
        return scipy_batch_sparse_blk_diag

# @line_profiler.profile

# @torch.jit.script


def get_3D_indices_from_batch_sparse_flat_test_v2(data_vec, indptr, indices, batch_vector, batch_size, n_matrix):

    row_vec = torch.div(indices, n_matrix, rounding_mode='floor')
    col_vec = torch.remainder(indices, n_matrix)
    B = indptr[1:] - indptr[:-1]
    batch_vec = batch_vector.repeat_interleave(B)

    indices = torch.vstack(
        (row_vec+batch_vec*n_matrix, col_vec+batch_vec*n_matrix))
    torch_batch_sparse_blk_diag = torch.sparse_coo_tensor(indices=indices, values=data_vec, size=[batch_size*n_matrix, batch_size*n_matrix], dtype=torch.complex64,
                                                          device=torch.device('cuda:0'), is_coalesced=True)

    return torch_batch_sparse_blk_diag


def get_Jac_Neighbour_Diag_from_flattened_Ybus_MiniBATCH(batch_sparse_flat_Ybus,
                                                         PQ_flags,
                                                         PV_flags,
                                                         Islanded_bus_flag,
                                                         Sample_select_flag):
    return get_Jac_Neighbour_Diag_from_flattend_Ybus_BATCH_v2(batch_sparse_flat_Ybus[Sample_select_flag],
                                                              PQ_flags,
                                                              PV_flags,
                                                              Islanded_bus_flag)


@nb.njit()
def zero_out_data_vec_TRUE_buses_SINGLE(row_vec, col_vec, data_vec, row_col_to_zero_flag_vector):
    row_flag = row_col_to_zero_flag_vector[row_vec]
    col_flag = row_col_to_zero_flag_vector[col_vec]
    data_vec[np.logical_or(row_flag, col_flag)] = 0.0
    # diag_indx = np.where(row_vec == col_vec)[0]
    # data_vec[row_vec == col_vec] += 1.0


# @line_profiler.profile
def get_Ybus_imag_flattened_vectors(batch_sparse_flat):
    if not (batch_sparse_flat.has_sorted_indices):  # VERY IMPORTANT STEP..
        batch_sparse_flat.sort_indices()
    batch_size = batch_sparse_flat.shape[0]
    n_matrix = np.sqrt(batch_sparse_flat.shape[1]).astype(np.int32)
    nnz = batch_sparse_flat.data.size

    indptr = batch_sparse_flat.indptr
    indices = batch_sparse_flat.indices

    batch_vec = np.zeros(nnz, dtype=np.int32)
    row_vec = np.zeros(nnz, dtype=np.int32)
    col_vec = np.zeros(nnz, dtype=np.int32)

    data_vec = batch_sparse_flat.imag.data.astype(np.float32)
    row_vec, col_vec = np.divmod(indices, n_matrix)
    batch_vec = np.repeat(np.arange(batch_size), np.diff(indptr))

    return batch_vec, row_vec, col_vec, data_vec, indptr, n_matrix, batch_size


@nb.njit()
def get_batch_sum_row_sparse_matrices(row_vec, data_vec, indptr, n_matrix, batch_size):
    row_sum_batch = np.zeros((batch_size, n_matrix))
    for k in nb.prange(batch_size):
        row_sum_batch[k, :] = np.bincount(
            row_vec[indptr[k]: indptr[k + 1]], weights=data_vec[indptr[k]: indptr[k + 1]], minlength=n_matrix)
    return row_sum_batch


@nb.njit()
def zero_out_data_vec_TRUE_buses_BATCH(row_vec, col_vec, data_vec, row_col_to_zero_flag_vector, indptr):
    for k in nb.prange(row_col_to_zero_flag_vector.shape[0]):
        zero_out_data_vec_TRUE_buses_SINGLE(row_vec[indptr[k]:indptr[k + 1]],
                                            col_vec[indptr[k]:indptr[k + 1]],
                                            data_vec[indptr[k]:indptr[k + 1]],
                                            row_col_to_zero_flag_vector[k])


@nb.njit()
def extract_diag_into_vec_and_zero_diag_SINGLE(row_vec, col_vec, data_vec, diag_vec):
    diag_indx = np.where(row_vec == col_vec)[0]
    diag_vec[row_vec[diag_indx]] = data_vec[diag_indx].copy()
    data_vec[diag_indx] *= 0


@nb.njit()
def extract_diag_into_vec_and_zero_diag_BATCH(row_vec, col_vec, data_vec, diag_vec, indptr):
    for k in nb.prange(diag_vec.shape[0]):
        extract_diag_into_vec_and_zero_diag_SINGLE(row_vec[indptr[k]:indptr[k + 1]],
                                                   col_vec[indptr[k]:indptr[k + 1]],
                                                   data_vec[indptr[k]:indptr[k + 1]],
                                                   diag_vec[k])


@nb.njit()
# @line_profiler.profile
def custom_extract_diag_into_vec_and_zero_diag_SINGLE(row_vec, col_vec, data_vec, diag_vec, row_sum, n_matrix):
    row_sum[:] = np.bincount(row_vec, weights=data_vec, minlength=n_matrix)
    diag_indx = np.where(row_vec == col_vec)[0]
    diag_vec[row_vec[diag_indx]] = data_vec[diag_indx].copy()
    data_vec[diag_indx] *= 0

# @nb.njit()


def custom_extract_diag_into_vec_and_zero_diag_BATCH(row_vec, col_vec, data_vec, diag_vec, row_sum, indptr, batch_size, n_matrix):
    for k in nb.prange(batch_size):
        custom_extract_diag_into_vec_and_zero_diag_SINGLE(row_vec[indptr[k]:indptr[k + 1]],
                                                          col_vec[indptr[k]:indptr[k + 1]],
                                                          data_vec[indptr[k]:indptr[k + 1]],
                                                          diag_vec[k],
                                                          row_sum[k],
                                                          n_matrix)


def diag_vec_replace_by_1_TRUE_buses_BATCH_SINGLE(diag_vec, row_col_to_zero_flag_vector):
    # THIS IS THE SIMPLE # -1 is there as we are estimating the negative of the
    diag_vec[row_col_to_zero_flag_vector] = 1.0


def get_Jac_Neighbour_Diag_from_flattend_Ybus_BATCH(batch_sparse_flat_Ybus,
                                                    PQ_flags,
                                                    PV_flags,
                                                    Islanded_bus_flag, sparse_output_format):
    batch_vec, row_vec, col_vec, data_vec, indptr, n_matrix, batch_size = get_Ybus_imag_flattened_vectors(
        batch_sparse_flat_Ybus)

    J11_D_diag_vec = np.zeros((batch_size, n_matrix))
    J22_D_diag_vec = np.zeros((batch_size, n_matrix))

    row_sum_batch = get_batch_sum_row_sparse_matrices(
        row_vec, data_vec, indptr, n_matrix, batch_size)

    extract_diag_into_vec_and_zero_diag_BATCH(
        row_vec, col_vec, data_vec, J11_D_diag_vec, indptr)
    J22_D_diag_vec = J11_D_diag_vec.copy()

    J11_D_diag_vec -= row_sum_batch
    J22_D_diag_vec += row_sum_batch

    # negative sign is important
    J11_N_data_vec = data_vec * -1.03 * 1.03
    J22_N_data_vec = data_vec * -1.03
    J11_D_diag_vec *= -1.03 * 1.03
    J22_D_diag_vec *= -1.03

    # Islanded buses
    zero_out_data_vec_TRUE_buses_BATCH(
        row_vec, col_vec, J11_N_data_vec, Islanded_bus_flag, indptr)
    zero_out_data_vec_TRUE_buses_BATCH(
        row_vec, col_vec, J22_N_data_vec, Islanded_bus_flag, indptr)
    J11_D_diag_vec[Islanded_bus_flag] = 1.0
    J22_D_diag_vec[Islanded_bus_flag] = 1.0

    # slack bus in J11
    zero_out_data_vec_TRUE_buses_BATCH(row_vec, col_vec, J11_N_data_vec,
                                       np.logical_not(np.logical_or(PQ_flags, PV_flags)), indptr)
    diag_vec_replace_by_1_TRUE_buses_BATCH_SINGLE(
        J11_D_diag_vec, np.logical_not(np.logical_or(PQ_flags, PV_flags)))

    # slack and PV buses in J22
    zero_out_data_vec_TRUE_buses_BATCH(row_vec, col_vec, J22_N_data_vec,
                                       np.logical_not(PQ_flags), indptr)
    diag_vec_replace_by_1_TRUE_buses_BATCH_SINGLE(
        J22_D_diag_vec, np.logical_not(PQ_flags))

    if sparse_output_format == "torch_3D":

        # constuct sparse batched matrices
        J11_N_torch_sparse_3D = torch.sparse_coo_tensor(indices=torch.tensor(np.array([batch_vec, row_vec, col_vec])),
                                                        values=J11_N_data_vec,
                                                        size=(batch_size, n_matrix, n_matrix))
        J22_N_torch_sparse_3D = torch.sparse_coo_tensor(indices=torch.tensor(np.array([batch_vec, row_vec, col_vec])),
                                                        values=J22_N_data_vec,
                                                        size=(batch_size, n_matrix, n_matrix))

        return J11_N_torch_sparse_3D, J22_N_torch_sparse_3D, J11_D_diag_vec, J22_D_diag_vec

    # construct sparse blk diag
    if sparse_output_format == "torch_blk_diag":
        J11_N_torch_sparse = torch.sparse_coo_tensor(indices=torch.tensor(np.array([row_vec+batch_vec*n_matrix, col_vec+batch_vec*n_matrix])),
                                                     values=J11_N_data_vec,
                                                     size=(batch_size*n_matrix, batch_size*n_matrix)).to_sparse_csr()
        # J11_N_torch_sparse = J11_N_torch_sparse.coa
        J22_N_torch_sparse = torch.sparse_coo_tensor(indices=torch.tensor(np.array([row_vec+batch_vec*n_matrix, col_vec+batch_vec*n_matrix])),
                                                     values=J22_N_data_vec,
                                                     size=(batch_size*n_matrix, batch_size*n_matrix)).to_sparse_csr()

        return J11_N_torch_sparse, J22_N_torch_sparse, J11_D_diag_vec, J22_D_diag_vec
    if sparse_output_format == "pytg":
        J11_N_scipy_sparse = sp.coo_array((J11_N_data_vec, (row_vec+batch_vec*n_matrix, col_vec+batch_vec*n_matrix)),
                                          shape=(batch_size*n_matrix, batch_size*n_matrix), dtype=np.float32)
        J22_N_scipy_sparse = sp.coo_array((J22_N_data_vec, (row_vec+batch_vec*n_matrix, col_vec+batch_vec*n_matrix)),
                                          shape=(batch_size*n_matrix, batch_size*n_matrix), dtype=np.float32)

        return pytg.utils.from_scipy_sparse_matrix(J11_N_scipy_sparse), pytg.utils.from_scipy_sparse_matrix(J22_N_scipy_sparse), \
            J11_D_diag_vec, J22_D_diag_vec

    if sparse_output_format == "scipy":
        J11_N_scipy_sparse = sp.coo_array((J11_N_data_vec, (row_vec+batch_vec*n_matrix, col_vec+batch_vec*n_matrix)),
                                          shape=(batch_size*n_matrix, batch_size*n_matrix), dtype=np.float32)
        J22_N_scipy_sparse = sp.coo_array((J22_N_data_vec, (row_vec+batch_vec*n_matrix, col_vec+batch_vec*n_matrix)),
                                          shape=(batch_size*n_matrix, batch_size*n_matrix), dtype=np.float32)

        return J11_N_scipy_sparse, J22_N_scipy_sparse, J11_D_diag_vec, J22_D_diag_vec


# @line_profiler.profile
def get_Jac_Neighbour_Diag_from_flattend_Ybus_BATCH_v2(batch_sparse_flat_Ybus,
                                                       PQ_flags,
                                                       PV_flags,
                                                       Islanded_bus_flag):
    batch_vec, row_vec, col_vec, data_vec, indptr, n_matrix, batch_size = get_Ybus_imag_flattened_vectors(
        batch_sparse_flat_Ybus)

    J11_D_diag_vec = np.zeros((batch_size, n_matrix))
    J22_D_diag_vec = np.zeros((batch_size, n_matrix))

    row_sum_batch = get_batch_sum_row_sparse_matrices(
        row_vec, data_vec, indptr, n_matrix, batch_size)

    extract_diag_into_vec_and_zero_diag_BATCH(
        row_vec, col_vec, data_vec, J11_D_diag_vec, indptr)
    J22_D_diag_vec = J11_D_diag_vec.copy()

    J11_D_diag_vec -= row_sum_batch
    J22_D_diag_vec += row_sum_batch

    # negative sign is important
    J11_N_data_vec = data_vec * -1.03 * 1.03
    J22_N_data_vec = data_vec * -1.03
    J11_D_diag_vec *= -1.03 * 1.03
    J22_D_diag_vec *= -1.03

    # Islanded buses
    zero_out_data_vec_TRUE_buses_BATCH(
        row_vec, col_vec, J11_N_data_vec, Islanded_bus_flag, indptr)
    zero_out_data_vec_TRUE_buses_BATCH(
        row_vec, col_vec, J22_N_data_vec, Islanded_bus_flag, indptr)
    J11_D_diag_vec[Islanded_bus_flag] = 1.0
    J22_D_diag_vec[Islanded_bus_flag] = 1.0

    # slack bus in J11
    zero_out_data_vec_TRUE_buses_BATCH(row_vec, col_vec, J11_N_data_vec,
                                       np.logical_not(np.logical_or(PQ_flags, PV_flags)), indptr)
    diag_vec_replace_by_1_TRUE_buses_BATCH_SINGLE(
        J11_D_diag_vec, np.logical_not(np.logical_or(PQ_flags, PV_flags)))

    # slack and PV buses in J22
    zero_out_data_vec_TRUE_buses_BATCH(row_vec, col_vec, J22_N_data_vec,
                                       np.logical_not(PQ_flags), indptr)
    diag_vec_replace_by_1_TRUE_buses_BATCH_SINGLE(
        J22_D_diag_vec, np.logical_not(PQ_flags))

    """
    J11_N_scipy_sparse = sp.coo_array((J11_N_data_vec, (row_vec+batch_vec*n_matrix, col_vec+batch_vec*n_matrix)),
                                      shape=(batch_size*n_matrix, batch_size*n_matrix), dtype=np.float32)
    J22_N_scipy_sparse = sp.coo_array((J22_N_data_vec, (row_vec+batch_vec*n_matrix, col_vec+batch_vec*n_matrix)),
                                      shape=(batch_size*n_matrix, batch_size*n_matrix), dtype=np.float32)
    J11_N_scipy_sparse.eliminate_zeros()
    J22_N_scipy_sparse.eliminate_zeros()

    # construct sparse blk diag
    J_temp = pytg.utils.from_scipy_sparse_matrix(J11_N_scipy_sparse)
    J11_N_torch_sparse = pytg.utils.to_torch_sparse_tensor(edge_index=J_temp[0], edge_attr=J_temp[1],
                                                               size=(batch_size*n_matrix, batch_size*n_matrix),
                                                               is_coalesced=True)
        # TODO - coalesce to increase speed ??
    J_temp = pytg.utils.from_scipy_sparse_matrix(J22_N_scipy_sparse)
    J22_N_torch_sparse = pytg.utils.to_torch_sparse_tensor(edge_index=J_temp[0], edge_attr=J_temp[1],
                                                               size=(batch_size*n_matrix, batch_size*n_matrix),
                                                               is_coalesced=True)
    """
    J11_N_torch_sparse = 0
    J22_N_torch_sparse = 0
    return J11_N_torch_sparse, J22_N_torch_sparse, J11_D_diag_vec, J22_D_diag_vec


def solve_jacobi_weighted_for_powerflow(
        # device: torch.device,
        Off, Diag_inv, b, n=torch.tensor(25)):
    device = 'cuda:0'
    x = torch.zeros_like(b, device=device)
    omega = 0.9
    for _ in torch.arange(n):
        x = omega * Diag_inv * (b - Off @ x) + (1 - omega) * x
    return x


@torch.jit.script
# @line_profiler.profile
def get_PQ_mismatch_Delta_Y_v2(YBus_Base, Delta_YBus_blk_diag, SBus_batch_124, J11_enable_flag, J22_enable_flag,
                               Vtheta, Vmag):
    VBus = torch.complex(torch.cos(Vtheta) * Vmag, torch.sin(Vtheta) * Vmag)
    # IBus = Delta_YBus_blk_diag @ VBus + (VBus.view(-1,124)@YBus_Base).view(-1,1)
    # IBus = Delta_YBus_blk_diag @ VBus
    IBus = F.linear(Delta_YBus_blk_diag, VBus.view(1, -1))
    temp1 = VBus.view(-1, 124)@YBus_Base
    temp2 = temp1.reshape(-1, 1)
    IBus += temp2  # TODO - remove reshape as it could be slow...
    # SBus_calc = VBus*torch.conj(IBus)
    SBus_mis = VBus * torch.conj(IBus) - SBus_batch_124
    # SBus_mis[New_Islanded_buses_124] = 0.0
    P_mis = SBus_mis.real
    Q_mis = SBus_mis.imag
    P_mis *= J11_enable_flag
    Q_mis *= J22_enable_flag
    return P_mis, Q_mis

# @line_profiler.profile


@torch.jit.script
# in-use
def get_Jac_times_vector(Y_base, Y_delta, J_flag_Y_base_row_sum, J_flag_Y_delta_row_sum, enable_flag, disable_flag, V_scale, en_scale, J11_J22_flag, x):

    res_0 = disable_flag * x
    res_1 = en_scale * x

    res_1_0 = J_flag_Y_delta_row_sum * res_1

    # res_1_1 = F.linear(Y_delta, res_1.view(1, -1))
    res_1_1 = torch.sparse.mm(Y_delta, res_1)
    temp1 = res_1.reshape(-1, 124)

    temp2 = torch.matmul(temp1, Y_base)
    res_1_2 = temp2.reshape(-1, 1)
    res_1_3 = (J_flag_Y_base_row_sum*res_1.reshape(-1, 124)).reshape(-1, 1)

    res_1 = res_1_3 + res_1_2 + res_1_1 + res_1_0
    res_1 = enable_flag * res_1

    return res_0 + res_1


@torch.jit.script
# @line_profiler.profile
# in-use
def pcg_solve_delta_J_implicit_J(YBus_imag_Base, Delta_YBus_imag_blk_diag, row_sum_Ybus_imag_base, row_sum_Ybus_delta_imag,
                                 enable_flag, disable_flag, scale, en_scale, flag, A_Diag_vec_inv, X0, W_k, A_times_Wk, b, n=torch.tensor(15)):
    X_k = X0
    R_k = b

    Z_k = A_Diag_vec_inv*R_k
    X_k1 = X_k
    R_k1 = R_k
    Z_k1 = Z_k
    for k in torch.arange(n):
        Z_k = A_Diag_vec_inv*R_k

        if k == 0:
            W_k += Z_k  # TODO - ASSUMES initial W_k is zero
            R_k1 = R_k
            X_k1 = X_k
            Z_k1 = Z_k
            ab_numerator = torch.matmul(R_k1.T, Z_k1)
        else:
            R_k2 = R_k1
            Z_k2 = Z_k1
            R_k1 = R_k
            Z_k1 = Z_k
            X_k1 = X_k
            denominator = torch.matmul(R_k2.T, Z_k2)
            denominator.masked_fill_(denominator == 0, 1e-8)
            ab_numerator = torch.matmul(R_k1.T, Z_k1)
            # beta = torch.matmul(R_k1.T, Z_k1) / denominator
            beta = ab_numerator / denominator
            W_k *= beta
            W_k += Z_k1
        A_times_Wk *= 0.0

        A_times_Wk = get_Jac_times_vector(YBus_imag_Base, Delta_YBus_imag_blk_diag, row_sum_Ybus_imag_base, row_sum_Ybus_delta_imag,
                                          enable_flag, disable_flag, scale, en_scale, flag, W_k)
        denominator = torch.matmul(W_k.T, A_times_Wk)
        denominator.masked_fill_(denominator == 0, 1e-8)
        # alpha = torch.matmul(R_k1.T, Z_k1) / denominator
        alpha = ab_numerator / denominator
        X_k = X_k1 + alpha * W_k
        R_k = R_k1 - alpha * A_times_Wk

    return X_k


@torch.jit.script
# @line_profiler.profile
# in-use
def RNN_block_Delta_Y_Delta_J_implicit_J(YBus_Base, Delta_YBus_blk_diag, YBus_imag_Base, Delta_YBus_imag_blk_diag, row_sum_Ybus_imag_base, row_sum_Ybus_delta_imag, SBus_batch_124,
                                         J11_D_diag_vec_inv, J22_D_diag_vec_inv, J11_enable_flag, J22_enable_flag, J11_disable_flag, J22_disable_flag, scale_J11, en_scale_J11,
                                         scale_J22, en_scale_J22, Vtheta, Vmag, X0, W_k, A_times_Wk, n_P=torch.tensor(5), n_Q=torch.tensor(4)):

    P_mis, Q_mis = get_PQ_mismatch_Delta_Y_v2(YBus_Base, Delta_YBus_blk_diag, SBus_batch_124, J11_enable_flag, J22_enable_flag,
                                              Vtheta, Vmag)
    X0 *= 0
    W_k *= 0
    A_times_Wk *= 0
    delta_Vtheta = pcg_solve_delta_J_implicit_J(YBus_imag_Base, Delta_YBus_imag_blk_diag, -row_sum_Ybus_imag_base,
                                                -row_sum_Ybus_delta_imag, J11_enable_flag, J11_disable_flag, scale_J11, en_scale_J11,
                                                torch.tensor(-1), J11_D_diag_vec_inv, X0, W_k, A_times_Wk, P_mis, n=n_P)
    X0 *= 0
    W_k *= 0
    A_times_Wk *= 0
    delta_Vmag = pcg_solve_delta_J_implicit_J(YBus_imag_Base, Delta_YBus_imag_blk_diag, row_sum_Ybus_imag_base,
                                              row_sum_Ybus_delta_imag, J22_enable_flag, J22_disable_flag, scale_J22, en_scale_J22,
                                              torch.tensor(1), J22_D_diag_vec_inv, X0, W_k, A_times_Wk, Q_mis, n=n_Q)

    # max mismatch
    return Vtheta - delta_Vtheta, Vmag - delta_Vmag, torch.norm(P_mis, float('inf')), torch.norm(Q_mis, float('inf')), torch.mean(torch.abs(P_mis)), torch.mean(torch.abs(Q_mis))

# @line_profiler.profile
# in-use


def power_flow_RNN_approach_Delta_Y_Delta_J_v2(Vmag, Vtheta, nn_VBus_theta_124, P_mis_max, Q_mis_max, P_mis_mae, Q_mis_mae,
                                               YBus_Base, Delta_YBus_blk_diag, YBus_imag_Base, Delta_YBus_imag_blk_diag,
                                               row_sum_Ybus_imag_base, row_sum_Ybus_delta_imag, SBus_batch_124,
                                               J11_D_diag_vec_inv, J22_D_diag_vec_inv, J11_enable_flag, J22_enable_flag, J11_disable_flag, J22_disable_flag, New_Islanded_buses_124,
                                               X0, W_k, A_times_Wk, n_RNN_iter=torch.tensor(25), n_P=torch.tensor(5), n_Q=torch.tensor(5)):
    Vtheta += nn_VBus_theta_124

    scale_J11 = torch.tensor(1.03 * 1.03)
    en_scale_J11 = J11_enable_flag * scale_J11 * -1
    scale_J22 = torch.tensor(1.03)
    en_scale_J22 = J22_enable_flag * scale_J22 * -1

    for _ in torch.arange(n_RNN_iter):
        Vtheta, Vmag, P_mis_max, Q_mis_max, P_mis_mae, Q_mis_mae \
            = RNN_block_Delta_Y_Delta_J_implicit_J(YBus_Base, Delta_YBus_blk_diag, YBus_imag_Base, Delta_YBus_imag_blk_diag, row_sum_Ybus_imag_base, row_sum_Ybus_delta_imag, SBus_batch_124,
                                                   J11_D_diag_vec_inv, J22_D_diag_vec_inv, J11_enable_flag, J22_enable_flag, J11_disable_flag, J22_disable_flag,
                                                   scale_J11, en_scale_J11, scale_J22, en_scale_J22, Vtheta, Vmag, X0, W_k, A_times_Wk, n_P, n_Q)

    P_mis, Q_mis = get_PQ_mismatch_Delta_Y_v2(YBus_Base, Delta_YBus_blk_diag, SBus_batch_124, J11_enable_flag, J22_enable_flag,
                                              Vtheta, Vmag)
    P_mis = P_mis.reshape(-1, 124)
    Q_mis = Q_mis.reshape(-1, 124)

    P_mis_max_scenario = torch.abs(P_mis)
    Q_mis_max_scenario = torch.abs(Q_mis)

    VBus = (Vmag * torch.exp(1j*Vtheta)).reshape(-1, 124)
    # P_mis_max_scenario = torch.max(torch.abs(P_mis), dim=1)[0]
    # Q_mis_max_scenario = torch.max(torch.abs(Q_mis), dim=1)[0]

    return VBus, P_mis_max, Q_mis_max, P_mis_mae, Q_mis_mae, P_mis_max_scenario, Q_mis_max_scenario  # max mismatch


# @line_profiler.profile
# in-use
def f2(batch_vec, row_vec, col_vec, data_vec, n_matrix, batch_size, J_base_flat):
    # flattened construction
    flattened_row = batch_vec
    flattened_col = row_vec + n_matrix*col_vec
    flattened_shape = (batch_size, n_matrix*n_matrix)
    J_flat = sp.coo_matrix(
        (data_vec, (flattened_row, flattened_col)), shape=flattened_shape).tocsr()
    # J_flat.eliminate_zeros()

    Delta_J = J_flat.astype(np.float32) - sp.csr_matrix(
        np.ones((batch_size, 1)))@J_base_flat.astype(np.float32).reshape(1, -1)
    Delta_J.data[np.abs(Delta_J.data) < 1e-5] = 0
    Delta_J.eliminate_zeros()

    batch_vec_delta, row_vec_delta, col_vec_delta, data_vec_delta, _, _, _ = get_Ybus_imag_flattened_vectors(
        Delta_J*1j)

    # sp diag construction
    sp_diag_row = row_vec_delta+batch_vec_delta*n_matrix
    sp_diag_col = col_vec_delta+batch_vec_delta*n_matrix
    sp_diag_shape = (batch_size*n_matrix, batch_size*n_matrix)

    sp_diag_torch_Delta_J = torch.sparse_coo_tensor(indices=torch.tensor(np.array([sp_diag_row, sp_diag_col])),
                                                    values=data_vec_delta,
                                                    size=sp_diag_shape).to_sparse_csr()  # no csr as it is memory inefficient

    return sp_diag_torch_Delta_J


# @line_profiler.profile
# in-use
def get_Delta_Jac_Neighbour_Diag_from_flattend_Ybus_BATCH(J11_N_base_flat, J22_N_base_flat,
                                                          batch_sparse_flat_Ybus,
                                                          PQ_flags,
                                                          PV_flags,
                                                          Islanded_bus_flag):
    batch_vec, row_vec, col_vec, data_vec, indptr, n_matrix, batch_size = get_Ybus_imag_flattened_vectors(
        batch_sparse_flat_Ybus)

    J11_D_diag_vec = np.zeros((batch_size, n_matrix))
    J22_D_diag_vec = np.zeros((batch_size, n_matrix))
    row_sum_batch = np.zeros((batch_size, n_matrix))
    # row_sum_batch = get_batch_sum_row_sparse_matrices(row_vec, data_vec, indptr, n_matrix, batch_size)
    custom_extract_diag_into_vec_and_zero_diag_BATCH(
        row_vec, col_vec, data_vec, J11_D_diag_vec, row_sum_batch,  indptr, batch_size, n_matrix)
    J22_D_diag_vec = J11_D_diag_vec.copy()

    J11_D_diag_vec -= row_sum_batch
    J22_D_diag_vec += row_sum_batch

    # negative sign is important
    J11_N_data_vec = data_vec * -1.03 * 1.03
    J22_N_data_vec = data_vec * -1.03
    J11_D_diag_vec *= -1.03 * 1.03
    J22_D_diag_vec *= -1.03

    # Islanded buses
    zero_out_data_vec_TRUE_buses_BATCH(
        row_vec, col_vec, J11_N_data_vec, Islanded_bus_flag, indptr)
    zero_out_data_vec_TRUE_buses_BATCH(
        row_vec, col_vec, J22_N_data_vec, Islanded_bus_flag, indptr)
    J11_D_diag_vec[Islanded_bus_flag] = 1.0
    J22_D_diag_vec[Islanded_bus_flag] = 1.0

    # slack bus in J11
    zero_out_data_vec_TRUE_buses_BATCH(row_vec, col_vec, J11_N_data_vec,
                                       np.logical_not(np.logical_or(PQ_flags, PV_flags)), indptr)
    diag_vec_replace_by_1_TRUE_buses_BATCH_SINGLE(
        J11_D_diag_vec, np.logical_not(np.logical_or(PQ_flags, PV_flags)))

    # slack and PV buses in J22
    zero_out_data_vec_TRUE_buses_BATCH(row_vec, col_vec, J22_N_data_vec,
                                       np.logical_not(PQ_flags), indptr)
    diag_vec_replace_by_1_TRUE_buses_BATCH_SINGLE(
        J22_D_diag_vec, np.logical_not(PQ_flags))

    J11_N_delta_torch = f2(batch_vec, row_vec, col_vec,
                           J11_N_data_vec, n_matrix, batch_size, J11_N_base_flat)
    J22_N_delta_torch = f2(batch_vec, row_vec, col_vec,
                           J22_N_data_vec, n_matrix, batch_size, J22_N_base_flat)

    return J11_N_delta_torch, J22_N_delta_torch, J11_D_diag_vec, J22_D_diag_vec


def get_Delta_Jac_Neighbour_Diag_from_flattened_Ybus_MiniBATCH(J11_N_base_flat, J22_N_base_flat, batch_sparse_flat_Ybus,
                                                               PQ_flags,
                                                               PV_flags,
                                                               Islanded_bus_flag,
                                                               Sample_select_flag):
    return get_Delta_Jac_Neighbour_Diag_from_flattend_Ybus_BATCH(J11_N_base_flat, J22_N_base_flat, batch_sparse_flat_Ybus[Sample_select_flag],
                                                                 PQ_flags,
                                                                 PV_flags,
                                                                 Islanded_bus_flag)


def get_YBus_base(YBus_base_list):
    Y_pytg = pytg.utils.from_scipy_sparse_matrix(
        YBus_base_list.reshape(124, 124).astype(np.complex64))
    Y_torch_sparse = pytg.utils.to_torch_sparse_tensor(edge_index=Y_pytg[0], edge_attr=Y_pytg[1],
                                                       size=(124, 124), is_coalesced=True)
    return Y_torch_sparse


def get_J_base(J_base):

    J_pytg = pytg.utils.from_scipy_sparse_matrix(
        J_base.astype(np.float32).reshape(124, 124))

    torch_J = pytg.utils.to_torch_sparse_tensor(edge_index=J_pytg[0], edge_attr=J_pytg[1],
                                                size=(124, 124)).to_sparse_csc()

    return torch_J


def torch_get_v_or_ex_from_Vbus_sample_v2(Vbus_kv_mag_sample, Vbus_pu_mag_sample, Vbus_theta_sample, topo_vect_sample, line_status_sample, line_or_pos_topo_vect, line_ex_pos_topo_vect, line_or_subid, line_ex_subid):
    line_or_bus_sample = torch.zeros_like(
        line_status_sample)  # -1 if disconnected
    line_ex_bus_sample = torch.zeros_like(line_status_sample)
    # line_ex_bus_sample = np.ones_like(line_or_subid) * np.nan

    topo_vect_or_expanded = topo_vect_sample[line_or_pos_topo_vect]
    topo_vect_ex_expanded = topo_vect_sample[line_ex_pos_topo_vect]
    mask = (topo_vect_or_expanded == 1).int()
    line_or_bus_sample += line_or_subid*mask
    mask = (topo_vect_or_expanded == 2).int()
    line_or_bus_sample += line_or_subid*mask + 118*mask
    mask = (topo_vect_ex_expanded == 1).int()
    line_ex_bus_sample += line_ex_subid*mask
    mask = (topo_vect_ex_expanded == 2).int()
    line_ex_bus_sample += line_ex_subid*mask + 118*mask

    # TODO - is the -1 case being properly estimated as the -1 index is just looping backwards
    v_or_sample = Vbus_kv_mag_sample[line_or_bus_sample]*line_status_sample
    v_ex_sample = Vbus_kv_mag_sample[line_ex_bus_sample]*line_status_sample
    # TODO - is the -1 case being properly estimated as the -1 index is just looping backwards
    v_or_pu_sample = Vbus_pu_mag_sample[line_or_bus_sample]*line_status_sample
    v_ex_pu_sample = Vbus_pu_mag_sample[line_ex_bus_sample]*line_status_sample
    # TODO - is the -1 case being properly estimated as the -1 index is just looping backwards
    theta_or_rad_sample = Vbus_theta_sample[line_or_bus_sample]
    theta_ex_rad_sample = Vbus_theta_sample[line_ex_bus_sample]

    return v_or_sample, v_ex_sample, v_or_pu_sample, v_ex_pu_sample, theta_or_rad_sample, theta_ex_rad_sample


# assumes parallel = 1 and length = 1km
def get_P_I_line(v_pu_or, v_pu_ex, theta_or_rad, theta_ex_rad, line_params):

    yac_ff_, yac_tt_, yac_tf_, yac_ft_ = line_params

    E_or = v_pu_or * torch.exp(1j*theta_or_rad)
    E_ex = v_pu_ex * torch.exp(1j*theta_ex_rad)

    I_orex = yac_ff_ * E_or + yac_ft_ * E_ex
    I_exor = yac_tt_ * E_ex + yac_tf_ * E_or

    I_orex = torch.conj(I_orex)
    I_exor = torch.conj(I_exor)

    s_orex = E_or * I_orex
    s_exor = E_ex * I_exor

    return s_orex, s_exor


def get_P_I_traf(v_pu_or, v_pu_ex, theta_or_rad, theta_ex_rad, traf_params):

    yac_ff_, yac_tt_, yac_tf_, yac_ft_ = traf_params

    E_hv = v_pu_or * torch.exp(1j*theta_or_rad)
    E_lv = v_pu_ex * torch.exp(1j*theta_ex_rad)

    I_hvlv = yac_ff_ * E_hv + yac_ft_ * E_lv
    I_lvhv = yac_tt_ * E_lv + yac_tf_ * E_hv

    I_hvlv = torch.conj(I_hvlv)
    I_lvhv = torch.conj(I_lvhv)

    s_hvlv = E_hv * I_hvlv
    s_lvhv = E_lv * I_lvhv

    return s_hvlv, s_lvhv


def get_amps(p, q, v, const1, const2):

    p2q2 = torch.square(p) + torch.square(q)
    p2q2 = torch.sqrt(p2q2)

    v_tmp = torch.abs(v)
    mask = (v_tmp == 0.).float()
    # v_tmp[v_tmp == 0.] = const2
    # v_tmp.masked_fill_(v_tmp == 0, const2)
    v_tmp += mask

    a = 1e3 * p2q2 * const1 / v_tmp

    return a


def get_line_params(grid_model, z_line_base, device):
    # r_pu_line = torch.tensor([i.r_pu for i in grid_model.get_lines()]).to(device)
    # x_pu_line = torch.tensor([i.x_pu for i in grid_model.get_lines()]).to(device)
    # h_pu_line = torch.tensor([i.h_pu for i in grid_model.get_lines()]).to(device)

    lines = grid_model.get_lines()
    r_pu_line, x_pu_line, h_pu_line = map(lambda attr: torch.tensor([getattr(i, attr) for i in lines]).to(device),
                                          ['r_pu', 'x_pu', 'h_pu'])
    z_line_pu = r_pu_line + 1j*x_pu_line
    ys = 1/z_line_pu

    h_or = 1j * 0.5 * h_pu_line
    h_ex = 1j * 0.5 * h_pu_line

    yac_ff_ = (ys + h_or)
    yac_tt_ = (ys + h_ex)
    yac_tf_ = -ys
    yac_ft_ = -ys
    z_line = r_pu_line * z_line_base
    return yac_ff_, yac_tt_, yac_tf_, yac_ft_, z_line


def get_traf_params(grid_model, z_base,  device):
    # r_pu_traf = torch.tensor([i.r_pu for i in grid_model.get_trafos()]).to(device)
    # x_pu_traf = torch.tensor([i.x_pu for i in grid_model.get_trafos()]).to(device)
    # h_pu_traf = torch.tensor([i.h_pu for i in grid_model.get_trafos()]).to(device)
    # ratio = torch.tensor([i.ratio for i in grid_model.get_trafos()]).to(device)
    # is_tap_hv_side = torch.tensor([i.is_tap_hv_side for i in grid_model.get_trafos()]).to(device)
    # shift = torch.tensor([i.shift_rad for i in grid_model.get_trafos()]).to(device)

    trafos = grid_model.get_trafos()
    r_pu_traf, x_pu_traf, h_pu_traf, ratio, is_tap_hv_side, shift = map(lambda attr: torch.tensor([getattr(i, attr) for i in trafos]).to(device),
                                                                        ['r_pu', 'x_pu', 'h_pu', 'ratio', 'is_tap_hv_side', 'shift_rad'])

    ys = 1. / (r_pu_traf + 1j * x_pu_traf)
    h = 1j * h_pu_traf * 0.5
    tau = ratio
    tau[~is_tap_hv_side] = 1/tau[~is_tap_hv_side]
    theta_shift = shift

    shape = (13,)

    real_part = 1.0
    imag_part = 0.0

    real_tensor = torch.full(shape, real_part)
    imag_tensor = torch.full(shape, imag_part)

    eitheta_shift = torch.complex(real_tensor, imag_tensor).to(device)
    emitheta_shift = torch.complex(real_tensor, imag_tensor).to(device)

    temp = theta_shift != 0.
    cos_theta = torch.cos(theta_shift[temp])
    sin_theta = torch.sin(theta_shift[temp])
    eitheta_shift[temp] = torch.complex(cos_theta, sin_theta)
    emitheta_shift[temp] = torch.complex(cos_theta, -sin_theta)

    yac_ff_ = (ys + h) / (tau * tau)
    yac_tt_ = (ys + h)
    yac_tf_ = -ys / (tau * emitheta_shift)
    yac_ft_ = -ys / (tau * eitheta_shift)

    z_traf = r_pu_traf * z_base
    return yac_ff_, yac_tt_, yac_tf_, yac_ft_, z_traf


def torch_get_line_vals_sample(prod_p, load_p, vbus_pu, vbus_theta, topo_vect, line_status, inv_sqrt_3, one, bus_kv,  line_or_pos_topo_vect, line_ex_pos_topo_vect, line_or_subid, line_ex_subid,
                               yac_ff_line, yac_tt_line, yac_tf_line, yac_ft_line, r_line,
                               yac_ff_traf, yac_tt_traf, yac_tf_traf, yac_ft_traf, r_traf):

    n_line = 173

    line_params = [yac_ff_line, yac_tt_line, yac_tf_line, yac_ft_line]
    traf_params = [yac_ff_traf, yac_tt_traf, yac_tf_traf, yac_ft_traf]

    vbus_kv = vbus_pu*bus_kv
    v_or, v_ex, v_or_pu, v_ex_pu, theta_or, theta_ex = torch_get_v_or_ex_from_Vbus_sample_v2(vbus_kv, vbus_pu, vbus_theta, topo_vect,
                                                                                             line_status, line_or_pos_topo_vect, line_ex_pos_topo_vect,
                                                                                             line_or_subid, line_ex_subid)

    v_or_line = v_or[:n_line]
    v_ex_line = v_ex[:n_line]
    theta_or_line = theta_or[:n_line]
    theta_ex_line = theta_ex[:n_line]

    v_or_traf = v_or[n_line:]
    v_ex_traf = v_ex[n_line:]
    theta_or_traf = theta_or[n_line:]
    theta_ex_traf = theta_ex[n_line:]

    v_pu_or_line = v_or_pu[:n_line]
    v_pu_ex_line = v_ex_pu[:n_line]
    v_pu_or_traf = v_or_pu[n_line:]
    v_pu_ex_traf = v_ex_pu[n_line:]
    s_orex, s_exor = get_P_I_line(
        v_pu_or_line, v_pu_ex_line, theta_or_line, theta_ex_line, line_params)

    s_hvlv, s_lvhv = get_P_I_traf(
        v_pu_or_traf, v_pu_ex_traf, theta_or_traf, theta_ex_traf, traf_params)

    p_or_line = torch.real(s_orex)
    q_or_line = torch.imag(s_orex)
    p_ex_line = torch.real(s_exor)
    q_ex_line = torch.imag(s_exor)

    p_or_traf = torch.real(s_hvlv)
    mask = (torch.abs(p_or_traf) > 0).float()
    p_or_traf = p_or_traf + mask * 1e-4
    q_or_traf = torch.imag(s_hvlv)
    p_ex_traf = torch.real(s_lvhv)
    q_ex_traf = torch.imag(s_lvhv)

    p_or = torch.cat((p_or_line, p_or_traf))
    p_ex = torch.cat((p_ex_line, p_ex_traf))

    gc_error = torch.sum(prod_p) - torch.sum(load_p) - torch.sum(p_ex + p_or)

    mask_gc = (torch.abs(p_ex_line) > 0).float()

    scale = 0.5 * gc_error / torch.sum(mask_gc)

    p_ex_line += mask_gc * scale
    p_or_line += mask_gc * scale

    p_or = torch.cat((p_or_line, p_or_traf))
    p_ex = torch.cat((p_ex_line, p_ex_traf))

    pl_mw_line = torch.abs(p_or_line + p_ex_line) / 3

    pl_line_r = torch.sqrt(pl_mw_line / r_line) * 2 * 1000

    a_ex_line = get_amps(p_ex_line, q_ex_line, v_ex_line, inv_sqrt_3, one)
    a_or_line = get_amps(p_or_line, q_or_line, v_or_line, inv_sqrt_3, one)
    jl_error = pl_line_r - a_or_line - a_ex_line

    a_ex_line += 0.5 * jl_error
    a_or_line += 0.5 * jl_error

    a_ex_traf = get_amps(p_ex_traf, q_ex_traf, v_ex_traf, inv_sqrt_3, one)
    a_or_traf = get_amps(p_or_traf, q_or_traf, v_or_traf, inv_sqrt_3, one)

    a_or = torch.cat((a_or_line, a_or_traf))
    a_ex = torch.cat((a_ex_line, a_ex_traf))

    return v_or, v_ex, p_or, p_ex, a_or, a_ex

# @nb.njit()


def adjust_GC_CPU(prod_p, load_p, p_ex_pred, p_or_pred):
    gc_cpu_error = np.sum(prod_p, axis=1) - np.sum(load_p,
                                                   axis=1) - np.sum(p_ex_pred + p_or_pred, axis=1)
    mask_gc = (np.abs(p_or_pred[:, :173]) > 0).astype(np.float32)

    scale = 0.5 * gc_cpu_error / np.sum(mask_gc, axis=1)

    p_ex_pred[:, :173] += mask_gc * scale[:, np.newaxis]
    p_or_pred[:, :173] += mask_gc * scale[:, np.newaxis]


def torch_get_Vbus_theta_from_theta_or_ex_sample(theta_bus_init, theta_or, theta_ex, topo_vect_sample, line_status_sample, line_or_pos_topo_vect, line_ex_pos_topo_vect, line_or_subid, line_ex_subid):
    # the ones are inportant here to increase the size to 237..
    line_or_bus_sample = torch.ones_like(
        line_status_sample)  # -1 if disconnected
    line_ex_bus_sample = torch.ones_like(line_status_sample)

    # line_ex_bus_sample = np.ones_like(line_or_subid) * np.nan
    Vbus_theta_sample = torch.zeros_like(theta_bus_init)

    topo_vect_or_expanded = topo_vect_sample[line_or_pos_topo_vect]
    topo_vect_ex_expanded = topo_vect_sample[line_ex_pos_topo_vect]
    mask = (topo_vect_or_expanded == 1).int()
    line_or_bus_sample += line_or_subid*mask
    mask = (topo_vect_or_expanded == 2).int()
    line_or_bus_sample += line_or_subid*mask + 118*mask
    mask = (topo_vect_ex_expanded == 1).int()
    line_ex_bus_sample += line_ex_subid*mask
    mask = (topo_vect_ex_expanded == 2).int()
    line_ex_bus_sample += line_ex_subid*mask + 118*mask

    line_or_bus_sample = line_or_bus_sample*line_status_sample
    line_ex_bus_sample = line_ex_bus_sample*line_status_sample

    # TODO - is the -1 case being properly estimated as the -1 index is just looping backwards
    Vbus_theta_sample[line_or_bus_sample] = theta_or
    Vbus_theta_sample[line_ex_bus_sample] = theta_ex

    # discarding zeroth element as it is islanding..
    Vbus_theta_sample = Vbus_theta_sample[1:]

    return Vbus_theta_sample


class TorchPF(nn.Module):
    def __init__(self,
                 name=None,
                 n_RNN_iter=None,
                 n_P_iter=None,
                 n_Q_iter=None,
                 get_PF_method=power_flow_RNN_approach_Delta_Y_Delta_J_v2,
                 get_node_volt=torch_get_Vbus_theta_from_theta_or_ex_sample,
                 get_J11_J22_diags=get_J11_J22_diags
                 ):
        super().__init__()
        self.name = name
        self._n_RNN_iter = torch.tensor(n_RNN_iter, dtype=torch.int8)
        self._n_P_iter = torch.tensor(n_P_iter, dtype=torch.int8)
        self._n_Q_iter = torch.tensor(n_Q_iter, dtype=torch.int8)
        self.get_PF_method_jit = torch.jit.script(get_PF_method)
        self.get_J11_J22_diags_jit = torch.jit.script(get_J11_J22_diags)

        # get_node_volt_jit = torch.jit.script(get_node_volt)
        # torch_get_Vbus_theta = get_node_volt
        self.torch_get_Vbus_theta = torch.vmap(get_node_volt,
                                               in_dims=(
                                                   0, 0, 0, 0, 0, None, None, None, None),
                                               out_dims=(0,))

    def build_model(self):
        pass
    # @line_profiler.profile

    def forward(self, _Delta_YBus, _Delta_YBus_imag, _row_sum_Ybus_delta_imag, _SBus, _PQ_nodes, _PV_nodes, _Islanded_buses,
                _topo_vect, _line_status, _Ybus_imag_delta_diag, _init_volt, _init_model_output, _YBus_Base,
                _YBus_imag_Base, _Ybus_imag_base_diag, _row_sum_Ybus_imag_base, _bus_renumber, line_topo_data,
                _P_mis_max, _Q_mis_max, _P_mis_mae, _Q_mis_mae, X0, W_k, A_times_Wk, _Vmag, _Vtheta):

        _J11_enable_flag = torch.logical_and(torch.logical_or(
            _PQ_nodes, _PV_nodes), ~_Islanded_buses).to(torch.int32).reshape(-1, 1)
        _J11_disable_flag = 1 - _J11_enable_flag

        _J22_enable_flag = torch.logical_and(
            _PQ_nodes, ~_Islanded_buses).to(torch.int32).reshape(-1, 1)
        _J22_disable_flag = 1 - _J22_enable_flag

        J11_D_diag_vec, J22_D_diag_vec = self.get_J11_J22_diags_jit(_Ybus_imag_base_diag, _Ybus_imag_delta_diag, _row_sum_Ybus_imag_base,
                                                                    _row_sum_Ybus_delta_imag, _J11_enable_flag, _J22_enable_flag)

        _J11_D_diag_inv = 1.0 / J11_D_diag_vec.reshape(-1, 1)
        _J22_D_diag_inv = 1.0 / J22_D_diag_vec.reshape(-1, 1)

        Vbus_theta_deg = self.torch_get_Vbus_theta(_init_volt, _init_model_output[:, 0:186],
                                                   _init_model_output[:, 186:], _topo_vect, _line_status.long(
        ),
            *line_topo_data)

        _VBus_theta = torch.deg2rad(
            Vbus_theta_deg[:, _bus_renumber > -1]).reshape(-1, 1)

        _VBus, _P_mis_norm, _Q_mis_norm, _P_mis_mae, _Q_mis_mae, _P_mis_max_scenario, _Q_mis_max_scenario\
            = self.get_PF_method_jit(_Vmag, _Vtheta, _VBus_theta, _P_mis_max, _Q_mis_max, _P_mis_mae, _Q_mis_mae, _YBus_Base,
                                     _Delta_YBus, _YBus_imag_Base,
                                     _Delta_YBus_imag, _row_sum_Ybus_imag_base.T, _row_sum_Ybus_delta_imag,
                                     _SBus, _J11_D_diag_inv, _J22_D_diag_inv, _J11_enable_flag, _J22_enable_flag,
                                     _J11_disable_flag, _J22_disable_flag, _Islanded_buses,
                                     X0, W_k, A_times_Wk, self._n_RNN_iter, self._n_P_iter, self._n_Q_iter)

        return _VBus, _P_mis_norm, _Q_mis_norm, _P_mis_mae, _Q_mis_mae, _P_mis_max_scenario, _Q_mis_max_scenario
