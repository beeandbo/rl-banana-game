import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepQModel(nn.Module):
    "Nueral Network used to drive a DQN agent"

    def __init__(self, state_size, action_size, seed):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.input_layer = nn.Linear(state_size, 128)
        self.hl1 = nn.Linear(128, 64)
        self.hl2 = nn.Linear(64, 64)
        # self.hidden_layers = []
        # for i, (in_size, out_size) in enumerate(zip(hidden_layers[:-1], hidden_layers[1:])):
        #     self.hidden_layers.append(nn.Linear(in_size, out_size))
        #     self.add_module("hidden_layer_%i" % (i,), self.hidden_layers[-1])
        self.output_layer = nn.Linear(64, action_size)

    def forward(self, input):
        x = F.relu(self.input_layer(input))
        x = F.relu(self.hl1(x))
        x = F.relu(self.hl2(x))
        return self.output_layer(x)
