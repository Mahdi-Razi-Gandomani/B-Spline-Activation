import torch.nn as nn
from activations import get_activation



class MLP(nn.Module):
    def __init__(self, in_size, hidden_sizes, out_size, act, act_cfg=None, shared_act=False):
        super().__init__()
        if act_cfg is None:
            act_cfg = {}

        layers = []
        acts = []
        prev = in_size
        for i, h in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev, h))
            if shared_act and i > 0:
                acts.append(acts[0])
            else:
                acts.append(get_activation(act, **act_cfg))
            
            prev = h

        layers.append(nn.Linear(prev, out_size))

        self.layers = nn.ModuleList(layers)
        self.acts = nn.ModuleList(acts)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        act_idx = 0
        for layer in self.layers[ : -1]:
            if isinstance(layer, nn.Linear):
                x = layer(x)
                x = self.acts[act_idx](x)
                act_idx += 1
            else:
                x = layer(x)
        x = self.layers[-1](x)
        return x


class CNN(nn.Module):
    def __init__(self, in_ch=3, num_classes=10, act='relu', act_cfg=None, chans=[32,64,128], fc=[256,128]):
        super().__init__()
        if act_cfg is None:
            act_cfg = {}

        conv_layers = []
        conv_acts = []
        prev_ch = in_ch
        for out_ch in chans:
            conv_layers.append(nn.Conv2d(prev_ch, out_ch, 3, padding = 1))
            conv_layers.append(nn.BatchNorm2d(out_ch))
            conv_acts.append(get_activation(act, **act_cfg))
            conv_layers.append(nn.MaxPool2d(2, 2))
            prev_ch = out_ch

        self.conv_layers = nn.ModuleList(conv_layers)
        self.conv_acts = nn.ModuleList(conv_acts)

        # calc fc input size
        s = 32 // (2 ** len(chans))
        fc_in = chans[-1] * s * s

        fc_layers = []
        fc_acts = []
        prev = fc_in
        for fc_size in fc:
            fc_layers.append(nn.Linear(prev, fc_size))
            fc_acts.append(get_activation(act, **act_cfg))
            prev = fc_size
        fc_layers.append(nn.Linear(prev, num_classes))

        self.fc_layers = nn.ModuleList(fc_layers)
        self.fc_acts = nn.ModuleList(fc_acts)

    def forward(self, x):
        act_idx = 0
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv2d):
                x = layer(x)
                x = self.conv_acts[act_idx](x)
                act_idx += 1
            else:
                x = layer(x)

        x = x.view(x.size(0), -1)

        act_idx = 0
        for layer in self.fc_layers[:-1]:
            x = layer(x)
            x = self.fc_acts[act_idx](x)
            act_idx += 1
        x = self.fc_layers[-1](x)
        return x





def create_model(model_type, in_size, out_size, act, **kwargs):

    if model_type == 'mlp':
        return MLP(in_size,kwargs.get('hidden_sizes', [256, 256]),out_size, act, kwargs.get('act_cfg', {}), kwargs.get('shared_act', False))
        
    elif model_type == 'cnn':
        return CNN(in_size, out_size, act, kwargs.get('act_cfg', {}), kwargs.get('chans', [32, 64, 128]), kwargs.get('fc', [256]))
    else:
        raise ValueError('Unknown model')
