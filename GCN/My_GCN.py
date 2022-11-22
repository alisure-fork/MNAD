import os
import sys
import dgl
import glob
import torch
import warnings
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from alisuretool.Tools import Tools
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset, DataLoader


warnings.filterwarnings("ignore")
sys.path.append("./benchmarking-gnns")


from layers.gat_layer import GATLayer
from layers.gcn_layer import GCNLayer
from layers.gated_gcn_layer import GatedGCNLayer
from layers.graphsage_layer import GraphSageLayer


def gpu_setup(use_gpu, gpu_id):
    if torch.cuda.is_available() and use_gpu:
        Tools.print()
        Tools.print('Cuda available with GPU: {}'.format(torch.cuda.get_device_name(0)))
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device("cuda:{}".format(gpu_id))
    else:
        Tools.print()
        Tools.print('Cuda not available')
        device = torch.device("cpu")
    return device


SketchFlowGraphNode = namedtuple("SketchFlowGraphNode",
                                 "index, point, length, angle, displacement, direction")


class SketchFlowGraph(object):

    def __init__(self):
        self.node_data = [SketchFlowGraphNode(index=i,
                                              point=[np.random.randint(1, 100) / 100, np.random.randint(1, 100) / 100],
                                              length=np.random.randint(0, 100) / 100,
                                              angle=np.random.randint(0, 180) / 180,
                                              displacement=np.random.randint(0, 100) / 100,
                                              direction=np.random.randint(0, 8) / 8)
                          for i in range(np.random.randint(20, 100))]
        self.edge_index, self.edge_w = self.set_edge()

        self.target = sum([one.length for one in self.node_data]) / sum([one for one in self.edge_w])
        pass

    def __len__(self):
        return len(self.node_data)

    def set_edge(self):
        # check
        assert sum([0<one.point[0]<1 and 0<one.point[1]<1 for one in self.node_data]) == len(self.node_data)

        edge_index, edge_w = [], []
        for one in self.node_data:
            _edge_w = []
            for two in self.node_data:
                dis_2 = 1/(np.sqrt((one.point[0] - two.point[0]) ** 2 + (one.point[1] - two.point[1]) ** 2) + 1)
                _edge_w.append(dis_2)

                edge_index.append([one.index, two.index])
                pass
            edge_w.extend(_edge_w / np.sum(_edge_w, axis=0))
            pass
        return np.asarray(edge_index), np.asarray(edge_w)

    def merge_node_data(self):
        np_data = [np.asarray([one.length, one.angle, one.displacement, one.direction]) for one in self.node_data]
        return np.asarray(np_data)

    pass


class MyDataset(Dataset):

    def __init__(self, data_root_path='D:\data\CIFAR', is_train=True):
        super().__init__()
        self.is_train = is_train
        self.data_root_path = data_root_path

        data_size = 1000 if self.is_train else 500
        self.data_set = [SketchFlowGraph() for i in range(data_size)]
        pass

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        my_graph = self.data_set[idx]

        target = my_graph.target
        node_data = my_graph.merge_node_data()
        edge_index, edge_w = my_graph.edge_index, my_graph.edge_w

        graph = dgl.DGLGraph()
        # graph = dgl.graph(data)
        graph.add_nodes(len(my_graph))
        graph.add_edges(edge_index[:, 0], edge_index[:, 1])
        graph.edata['feat'] = torch.from_numpy(edge_w).unsqueeze(1).float()
        graph.ndata['feat'] = torch.from_numpy(node_data).float()

        return graph, target

    @staticmethod
    def collate_fn(samples):
        graphs, targets = map(list, zip(*samples))

        targets = torch.tensor(np.array(targets))

        _nodes_num = [graph.number_of_nodes() for graph in graphs]
        _edges_num = [graph.number_of_edges() for graph in graphs]
        nodes_num_norm_sqrt = torch.cat([torch.zeros((num, 1)).fill_(1./num) for num in _nodes_num]).sqrt()
        edges_num_norm_sqrt = torch.cat([torch.zeros((num, 1)).fill_(1./num) for num in _edges_num]).sqrt()
        batched_graph = dgl.batch(graphs)

        return targets, batched_graph, nodes_num_norm_sqrt, edges_num_norm_sqrt

    pass


class GCNNet(nn.Module):

    def __init__(self, node_dim, in_dim, hidden_dims, out_dim=128):
        super().__init__()
        self.embedding_h = nn.Linear(node_dim, in_dim)
        self.gcn_list = nn.ModuleList()
        for hidden_dim in hidden_dims:
            self.gcn_list.append(GCNLayer(in_dim, hidden_dim, F.relu, 0.0, True, True, True))
            in_dim = hidden_dim
            pass
        self.readout_mlp = nn.Linear(hidden_dims[-1], out_dim, bias=False)
        pass

    def forward(self, graphs, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt):
        hidden_nodes_feat = self.embedding_h(nodes_feat)
        for gcn in self.gcn_list:
            hidden_nodes_feat = gcn(graphs, hidden_nodes_feat, nodes_num_norm_sqrt)
            pass
        graphs.ndata['h'] = hidden_nodes_feat
        hg = dgl.mean_nodes(graphs, 'h')
        logits = self.readout_mlp(hg)
        return logits

    pass


class GraphSageNet(nn.Module):

    def __init__(self, node_dim, in_dim, hidden_dims, out_dim=128):
        super().__init__()
        self.embedding_h = nn.Linear(node_dim, in_dim)
        self.gcn_list = nn.ModuleList()
        for hidden_dim in hidden_dims:
            self.gcn_list.append(GraphSageLayer(in_dim, hidden_dim, F.relu, 0.0, "meanpool", True))
            in_dim = hidden_dim
            pass
        self.readout_mlp = nn.Linear(hidden_dims[-1], out_dim, bias=False)
        pass

    def forward(self, graphs, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt):
        hidden_nodes_feat = self.embedding_h(nodes_feat)
        for gcn in self.gcn_list:
            hidden_nodes_feat = gcn(graphs, hidden_nodes_feat, nodes_num_norm_sqrt)
            pass

        graphs.ndata['h'] = hidden_nodes_feat
        hg = dgl.mean_nodes(graphs, 'h')
        logits = self.readout_mlp(hg)
        return logits

    pass


class GatedGCNNet(nn.Module):

    def __init__(self, node_dim, in_dim, hidden_dims, out_dim=128):
        super().__init__()

        self.in_dim_edge = 1
        self.embedding_h = nn.Linear(node_dim, in_dim)
        self.embedding_e = nn.Linear(self.in_dim_edge, in_dim)

        self.gcn_list = nn.ModuleList()
        for hidden_dim in hidden_dims:
            self.gcn_list.append(GatedGCNLayer(in_dim, hidden_dim, 0.0, True, True, True))
            in_dim = hidden_dim
            pass

        self.readout_mlp = nn.Linear(hidden_dims[-1], out_dim, bias=False)
        pass

    def forward(self, graphs, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt):
        h = self.embedding_h(nodes_feat)
        e = self.embedding_e(edges_feat)

        for gcn in self.gcn_list:
            h, e = gcn(graphs, h, e, nodes_num_norm_sqrt, edges_num_norm_sqrt)
            pass

        graphs.ndata['h'] = h
        hg = dgl.mean_nodes(graphs, 'h')
        logits = self.readout_mlp(hg)
        return logits

    pass


class MyGCNNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.model_gnn = GCNNet(node_dim=4, in_dim=128, hidden_dims=[128, 128, 128, 128], out_dim=1)
        # self.model_gnn = GraphSageNet(node_dim=4, in_dim=128, hidden_dims=[128, 128, 128, 128], out_dim=1)
        # self.model_gnn = GatedGCNNet(node_dim=4, in_dim=128, hidden_dims=[128, 128, 128, 128], out_dim=1)
        pass

    def forward(self, batched_graph, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt):
        logits = self.model_gnn.forward(batched_graph, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt)
        return logits

    pass


class Runner(object):

    def __init__(self, data_root_path='/mnt/4T/Data/cifar/cifar-10', batch_size=64,
                 train_print_freq=100, test_print_freq=50, is_sgd=True,
                 root_ckpt_dir="./ckpt2/norm3", num_workers=8, use_gpu=True, gpu_id="0"):
        self.train_print_freq = train_print_freq
        self.test_print_freq = test_print_freq

        self.device = gpu_setup(use_gpu=use_gpu, gpu_id=gpu_id)
        self.root_ckpt_dir = Tools.new_dir(root_ckpt_dir)

        self.train_dataset = MyDataset(data_root_path=data_root_path, is_train=True)
        self.test_dataset = MyDataset(data_root_path=data_root_path, is_train=False)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers, collate_fn=self.train_dataset.collate_fn)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False,
                                      num_workers=num_workers, collate_fn=self.test_dataset.collate_fn)

        self.model = MyGCNNet().to(self.device)
        if is_sgd:
            self.lr_s = [[0, 0.001], [100, 0.0001]]
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr_s[0][0], momentum=0.9, weight_decay=5e-4)
        else:
            self.lr_s = [[0, 0.001], [100, 0.0001]]
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_s[0][0], weight_decay=0.0)

        self.loss_class = nn.MSELoss().to(self.device)

        Tools.print("Total param: {}".format(self._view_model_param(self.model)))
        pass

    def load_model(self, model_file_name):
        ckpt = torch.load(model_file_name, map_location=self.device)
        self.model.load_state_dict(ckpt, strict=False)
        Tools.print('Load Model: {}'.format(model_file_name))
        pass

    def train(self, epochs, start_epoch=0):
        for epoch in range(start_epoch, epochs):
            Tools.print()
            Tools.print("Start Epoch {}".format(epoch))

            self._lr(epoch)
            Tools.print('Epoch:{:02d},lr={:.4f}'.format(epoch, self.optimizer.param_groups[0]['lr']))

            epoch_loss, epoch_train_acc = self._train_epoch()
            self._save_checkpoint(self.model, self.root_ckpt_dir, epoch)
            epoch_test_loss, epoch_test_acc = self.test()

            Tools.print('Epoch: {:02d}, Train: {:.4f}/{:.4f} Test: {:.4f}/{:.4f}'.format(
                epoch, epoch_train_acc, epoch_loss, epoch_test_acc, epoch_test_loss))
            pass
        pass

    def _train_epoch(self):
        self.model.train()
        epoch_loss, epoch_train_acc, nb_data = 0, 0, 0
        for i, (labels, batched_graph, nodes_num_norm_sqrt, edges_num_norm_sqrt) in enumerate(self.train_loader):
            # Data
            labels = labels.float().to(self.device)
            batched_graph = batched_graph.to(self.device)
            nodes_feat = batched_graph.ndata['feat'].to(self.device)
            edges_feat = batched_graph.edata['feat'].to(self.device)
            nodes_num_norm_sqrt = nodes_num_norm_sqrt.to(self.device)
            edges_num_norm_sqrt = edges_num_norm_sqrt.to(self.device)

            # Run
            self.optimizer.zero_grad()
            logits = self.model.forward(batched_graph, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt)
            loss = self.loss_class(logits, labels.unsqueeze(-1))
            loss.backward()
            self.optimizer.step()

            # Stat
            nb_data += labels.size(0)
            epoch_loss += loss.detach().item()
            epoch_train_acc += self._accuracy(logits.detach().cpu(), labels.cpu())

            # Print
            if i % self.train_print_freq == 0:
                Tools.print("{}-{} loss={:4f}/{:4f} acc={:4f}".format(
                    i, len(self.train_loader), epoch_loss/(i+1), loss.detach().item(), epoch_train_acc/nb_data))
                pass
            pass

        epoch_train_acc /= nb_data
        epoch_loss /= (len(self.train_loader) + 1)
        return epoch_loss, epoch_train_acc

    def test(self):
        self.model.eval()

        Tools.print()
        epoch_test_loss, epoch_test_acc, nb_data = 0, 0, 0
        with torch.no_grad():
            for i, (labels, batched_graph, nodes_num_norm_sqrt, edges_num_norm_sqrt) in enumerate(self.test_loader):
                # Data
                labels = labels.long().to(self.device)
                batched_graph = batched_graph.to(self.device)
                nodes_feat = batched_graph.ndata['feat'].to(self.device)
                edges_feat = batched_graph.edata['feat'].to(self.device)
                nodes_num_norm_sqrt = nodes_num_norm_sqrt.to(self.device)
                edges_num_norm_sqrt = edges_num_norm_sqrt.to(self.device)

                # Run
                logits = self.model.forward(batched_graph, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt)
                loss = self.loss_class(logits, labels.unsqueeze(-1))

                # Stat
                nb_data += labels.size(0)
                epoch_test_loss += loss.detach().item()
                epoch_test_acc += self._accuracy(logits.detach().cpu(), labels.cpu())

                # Print
                if i % self.test_print_freq == 0:
                    Tools.print("{}-{} loss={:4f}/{:4f} acc={:4f}".format(
                        i, len(self.test_loader), epoch_test_loss/(i+1), loss.detach().item(), epoch_test_acc/nb_data))
                    pass
                pass
            pass

        return epoch_test_loss / (len(self.test_loader) + 1), epoch_test_acc / nb_data

    def _lr(self, epoch):
        # [[0, 0.001], [100, 0.001]
        for lr in self.lr_s:
            if lr[0] == epoch:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr[1]
                pass
            pass
        pass

    @staticmethod
    def _save_checkpoint(model, root_ckpt_dir, epoch):
        torch.save(model.state_dict(), os.path.join(root_ckpt_dir, 'epoch_{}.pkl'.format(epoch)))
        for file in glob.glob(root_ckpt_dir + '/*.pkl'):
            if int(file.split('_')[-1].split('.')[0]) < epoch - 1:
                os.remove(file)
                pass
            pass
        pass

    @staticmethod
    def _accuracy(scores, targets):
        return mean_squared_error(targets, scores)

    @staticmethod
    def _view_model_param(model):
        total_param = 0
        for param in model.parameters():
            total_param += np.prod(list(param.data.size()))
        return total_param

    pass


if __name__ == '__main__':
    _root_ckpt_dir = "./log/gcn/{}".format("GCNNet")
    _batch_size = 8
    _epochs = 200
    _train_print_freq = 200
    _test_print_freq = 100
    _num_workers = 8
    _use_gpu = True
    _gpu_id = "0"

    Tools.print("ckpt:{} batch size:{} workers:{} gpu:{}".format(_root_ckpt_dir, _batch_size, _num_workers, _gpu_id))

    runner = Runner(root_ckpt_dir=_root_ckpt_dir, batch_size=_batch_size,
                       train_print_freq=_train_print_freq, test_print_freq=_test_print_freq,
                       num_workers=_num_workers, use_gpu=_use_gpu, gpu_id=_gpu_id)
    runner.train(_epochs, start_epoch=0)

    pass
