import sys
import dgl
import torch.nn as nn
import torch.nn.functional as F
from .memory_final_spatial_sumonly_weight_ranking_top1 import *


sys.path.append("./GCN/benchmarking-gnns")
sys.path.append("../GCN/benchmarking-gnns")


from layers.gat_layer import GATLayer
from layers.gcn_layer import GCNLayer
from layers.gated_gcn_layer import GatedGCNLayer
from layers.graphsage_layer import GraphSageLayer


class Encoder(torch.nn.Module):
    def __init__(self, t_length = 5, n_channel =3):
        super(Encoder, self).__init__()
        
        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput), torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput), torch.nn.ReLU(inplace=False))
        
        def Basic_(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput), torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1))
        
        self.moduleConv1 = Basic(n_channel*(t_length-1), 64)
        self.modulePool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(64, 128)
        self.modulePool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.moduleConv3 = Basic(128, 256)
        self.modulePool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic_(256, 512)
        self.moduleBatchNorm = torch.nn.BatchNorm2d(512)
        self.moduleReLU = torch.nn.ReLU(inplace=False)
        pass
        
    def forward(self, x):
        tensorConv1 = self.moduleConv1(x)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)

        tensorConv4 = self.moduleConv4(tensorPool3)
        return tensorConv4, tensorConv1, tensorConv2, tensorConv3

    pass


class Decoder(torch.nn.Module):

    def __init__(self, t_length = 5, n_channel =3):
        super(Decoder, self).__init__()
        
        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput), torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput), torch.nn.ReLU(inplace=False))

        def Gen(intInput, intOutput, nc):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc), torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc), torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh())
        
        def Upsample(nc, intOutput):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=nc, out_channels=intOutput, kernel_size=3,
                                         stride=2, padding=1, output_padding=1),
                torch.nn.BatchNorm2d(intOutput), torch.nn.ReLU(inplace=False))
      
        self.moduleConv = Basic(1024, 512)
        self.moduleUpsample4 = Upsample(512, 256)

        self.moduleDeconv3 = Basic(512, 256)
        self.moduleUpsample3 = Upsample(256, 128)

        self.moduleDeconv2 = Basic(256, 128)
        self.moduleUpsample2 = Upsample(128, 64)

        self.moduleDeconv1 = Gen(128,n_channel,64)
        pass
        
    def forward(self, x, skip1, skip2, skip3):
        tensorConv = self.moduleConv(x)

        tensorUpsample4 = self.moduleUpsample4(tensorConv)
        cat4 = torch.cat((skip3, tensorUpsample4), dim = 1)
        
        tensorDeconv3 = self.moduleDeconv3(cat4)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)
        cat3 = torch.cat((skip2, tensorUpsample3), dim = 1)
        
        tensorDeconv2 = self.moduleDeconv2(cat3)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)
        cat2 = torch.cat((skip1, tensorUpsample2), dim = 1)
        
        output = self.moduleDeconv1(cat2)
        return output

    pass


class convAE(torch.nn.Module):

    def __init__(self, n_channel=3, t_length=5, memory_size=10, feature_dim=512,
                 key_dim=512, temp_update=0.1, temp_gather=0.1):
        super(convAE, self).__init__()
        self.encoder = Encoder(t_length, n_channel)
        self.decoder = Decoder(t_length, n_channel)
        self.memory = Memory(memory_size,feature_dim, key_dim, temp_update, temp_gather)
        pass

    def forward(self, x, keys,train=True):
        fea, skip1, skip2, skip3 = self.encoder(x)
        if train:
            updated_fea, keys, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = self.memory(fea, keys, train)
            output = self.decoder(updated_fea, skip1, skip2, skip3)
            return output, fea, updated_fea, keys, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss
        else:  # test
            updated_fea, keys, softmax_score_query, softmax_score_memory,query, top1_keys, keys_ind, compactness_loss = self.memory(fea, keys, train)
            output = self.decoder(updated_fea, skip1, skip2, skip3)
            return output, fea, updated_fea, keys, softmax_score_query, softmax_score_memory, query, top1_keys, keys_ind, compactness_loss
        pass

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

    def __init__(self, node_dim=4, in_dim=128, hidden_dims=[128, 128, 128, 128], out_dim=1, gnn=GCNNet):
        super().__init__()
        self.model_gnn = gnn(node_dim=node_dim, in_dim=in_dim, hidden_dims=hidden_dims, out_dim=out_dim)
        pass

    def forward(self, batched_graph, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt):
        logits = self.model_gnn.forward(batched_graph, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt)
        return logits

    pass


class ConvAESketchFlow(torch.nn.Module):

    def __init__(self, n_channel=3, t_length=5, memory_size=10, feature_dim=512,
                 key_dim=512, temp_update=0.1, temp_gather=0.1, gcn_nets=None):
        super(ConvAESketchFlow, self).__init__()

        self.encoder = Encoder(t_length, n_channel)
        self.decoder = Decoder(t_length, n_channel)
        self.gcns = gcn_nets
        self.memory = Memory(memory_size,feature_dim, key_dim, temp_update, temp_gather)
        pass

    def forward(self, x, keys, batched_graph, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt, train=True,
                batched_graph_2=None, nodes_feat_2=None, edges_feat_2=None, nodes_num_norm_sqrt_2=None, edges_num_norm_sqrt_2=None):
        fea, skip1, skip2, skip3 = self.encoder(x)

        gcn_feature = self.gcns[0].forward(batched_graph, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt)
        gcn_feature_softmax = torch.softmax(gcn_feature, dim=1)

        if batched_graph_2 is None:
            fea = fea + fea * gcn_feature_softmax.unsqueeze(-1).unsqueeze(-1)
        else:
            gcn_feature_2 = self.gcns[1].forward(batched_graph_2, nodes_feat_2, edges_feat_2, nodes_num_norm_sqrt_2, edges_num_norm_sqrt_2)
            gcn_feature_softmax_2 = torch.softmax(gcn_feature_2, dim=1)
            fea = fea + (fea * gcn_feature_softmax.unsqueeze(-1).unsqueeze(-1) + fea * gcn_feature_softmax_2.unsqueeze(-1).unsqueeze(-1)) / 2
            pass

        if train:
            updated_fea, keys, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = self.memory(fea, keys, train)
            output = self.decoder(updated_fea, skip1, skip2, skip3)
            return output, fea, updated_fea, keys, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss
        else:  # test
            updated_fea, keys, softmax_score_query, softmax_score_memory,query, top1_keys, keys_ind, compactness_loss = self.memory(fea, keys, train)
            output = self.decoder(updated_fea, skip1, skip2, skip3)
            return output, fea, updated_fea, keys, softmax_score_query, softmax_score_memory, query, top1_keys, keys_ind, compactness_loss
        pass

    pass

