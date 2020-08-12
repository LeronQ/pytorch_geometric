import os.path as osp
import sys

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T


import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.nn import Sequential, ReLU, Linear, LogSoftmax
from torch_geometric.utils import degree
from torch_geometric.nn import PNAConv
import argparse, numpy as np, time

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, help='Number of epochs.', default=1000)
parser.add_argument('--print_every', type=int, help='Print every.', default=100)
parser.add_argument('--weight_decay', type=float, help="Please give a value for weight_decay", default=5e-3)
parser.add_argument('--lr', type=float, help="Please give a value for init_lr", default=0.001)
parser.add_argument('--hidden', type=int, help='Size of hidden conv', default=100)
parser.add_argument('--n_conv_layers', type=int, help='number of conv layers (after first)', default=3)
parser.add_argument('--pretrans_layers', type=int, help='pretrans_layers.', default=1)
parser.add_argument('--posttrans_layers', type=int, help='posttrans_layers.', default=1)
parser.add_argument('--towers', type=int, help='towers.', default=5)
parser.add_argument('--mlp_layers', type=int, help='mlp_layers.', default=3)
args = parser.parse_args()

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.TargetIndegree())
data = dataset[0]

data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.train_mask[:data.num_nodes - 1000] = 1
data.val_mask = None
data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.test_mask[data.num_nodes - 500:] = 1

deg = torch.zeros(5, dtype=torch.long)

d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
deg = torch.bincount(d, minlength=deg.numel())


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = ModuleList()
        self.convs.append(PNAConv(in_channels=dataset.num_features, out_channels=args.hidden,
                                  aggregators=aggregators, scalers=scalers, deg=deg,
                                  edge_dim=dataset.num_edge_features, towers=1, pre_layers=args.pretrans_layers,
                                  post_layers=args.posttrans_layers,
                                  divide_input=False))
        for _ in range(args.n_conv_layers):
            conv = PNAConv(in_channels=args.hidden, out_channels=args.hidden,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=dataset.num_edge_features, towers=args.towers, pre_layers=args.pretrans_layers,
                           post_layers=args.posttrans_layers,
                           divide_input=False)
            self.convs.append(conv)

        gr = []
        g = list(map(int, np.ceil(np.geomspace(args.hidden, dataset.num_classes, args.mlp_layers + 1))))
        g[0] = args.hidden
        g[-1] = dataset.num_classes
        for i in range(args.mlp_layers):
            gr.append(Linear(g[i], g[i + 1]))
            gr.append(LogSoftmax() if i == args.mlp_layers-1 else ReLU())

        self.mlp = Sequential(*gr)

    def forward(self, x, edge_index, edge_attr):

        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_attr))
        return self.mlp(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device', device)
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(data.x, data.edge_index, data.edge_attr)[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def test():
    model.eval()
    logits, accs = model(data.x, data.edge_index, data.edge_attr), []
    for _, mask in data('train_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        accs.append(loss)
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc * 100)
    return accs

sys.stdout.flush()

tic = time.time()
for epoch in range(1, args.epochs + 1):
    train()
    if epoch%args.print_every == 0:
        log = 'Epoch: {:04d} , Train: loss: {:.4f} acc: {:02.2f}%, Test: loss: {:.4f} acc: {:02.2f}% {:01.2}s/epoch'
        print(log.format(epoch, *test(), (time.time()-tic)/epoch))
        sys.stdout.flush()

