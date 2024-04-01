from typing import List, Tuple, Optional, Dict, Union
import glob
import os

import flwr as fl
from flwr.common import *
from flwr.server.client_proxy import ClientProxy
import numpy as np
import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F

import flwr.server.client_manager as ClientManager
import json

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


net = Net().to(DEVICE)


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# Define strategy
# strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def _set_initial_parameters(self):
        if not os.listdir('checkpoints'):  # Check if the directory is empty
            print("No checkpoints found.")
            return None
        param, _ = load_parameters_from_disk(net)
        return param

    def _filter(self, obj):
        result = {}
        for key, value in obj.items():
            result[key] = {
                'status': {
                    'code': value.status.code.name,
                    'message': value.status.message
                },
                'properties': value.properties
            }
        with open('properties.json', 'w') as f:
            json.dump(result, f)

    def initialize_parameters(self, client_manager: ClientManager):
        client_manager.wait_for(num_clients=2)
        client_properties = {}
        for cid, client in client_manager.all().items():
            ins = GetPropertiesIns({})
            client_properties[cid] = client.get_properties(ins, timeout=30)
        print(f"Client properties before filter: {client_properties}")
        print(self._filter(client_properties))
        # for loading saved model checkpoints
        initial_parameters = self._set_initial_parameters()
        return initial_parameters  # Always return initial_parameters, which could be None

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(net.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v)
                                     for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

            # Save the model
            torch.save(net.state_dict(), f"checkpoints/model_round_{server_round}.pth")

            # aggregated_parameters = fl.common.ndarrays_to_parameters(aggregated_parameters)

        return aggregated_parameters, aggregated_metrics


def load_parameters_from_disk(net):
    list_of_files = [fname for fname in glob.glob('checkpoints/model_round_*')]
    if not list_of_files:  # Check if the list is empty
        print("No checkpoints found.")
        return None, None
    else:    
        # latest_round_file = max(list_of_files, key=os.path.getctime)
        latest_round_file = max(list_of_files, key=lambda fname: int(
            fname.split('_')[-1].split('.')[0]))
        latest_round_number = int(latest_round_file.split('_')[-1].split('.')[0])
        print(f"Loaded {latest_round_file} from disk")
        state_dict = torch.load(latest_round_file, map_location=DEVICE)
        net.load_state_dict(state_dict, strict=True)
        state_dict_ndarrays = [v.cpu().numpy() for v in net.state_dict().values()]
        parameters = fl.common.ndarrays_to_parameters(state_dict_ndarrays)
        # Convert the list of numpy ndarrays to Parameters
        return parameters, latest_round_number


# _, latest_round_number = load_parameters_from_disk(net)
_, latest_round_number = load_parameters_from_disk(net)

# strategy = SaveModelStrategy(
#         evaluate_metrics_aggregation_fn=weighted_average,
# )

strategy = SaveModelStrategy(
    initial_parameters=None, evaluate_metrics_aggregation_fn=weighted_average)

if latest_round_number is None:
    latest_round_number = 0

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10 - latest_round_number),
    # config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)
