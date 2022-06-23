import os
import glob
import random
import argparse
import warnings
from utils import *
import torch.optim as optim
import torch.utils.data as data
from collections import OrderedDict
from torch.autograd import Variable
from alisuretool.Tools import Tools
from torch.nn import functional as F
import torchvision.transforms as transforms
from model.Reconstruction import convAE as ConvAERecon
from model.utils import DataLoader, DataLoaderSketchFlow
from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import convAE as ConvAEPred
from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import GCNNet, GraphSageNet, GatedGCNNet, MyGCNNet, ConvAESketchFlow


warnings.filterwarnings("ignore")


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


def seed_setup(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance
    pass


class Runner(object):

    def __init__(self, args):
        self.args = args

        self.log_dir = Tools.new_dir(os.path.join('./result/exp', self.args.dataset_type,
                                                  self.args.method, self.args.exp_dir))

        # Loading dataset
        self.train_folder = os.path.join(self.args.dataset_path, self.args.dataset_type, "training/frames")
        self.test_folder = os.path.join(self.args.dataset_path, self.args.dataset_type, "testing/frames")

        if self.args.has_sketch_flow:
            self.sketch_flow_train_folder = os.path.join(self.args.dataset_path, "sketch", self.args.dataset_type, "training/sketch_flow")
            self.sketch_flow_test_folder = os.path.join(self.args.dataset_path, "sketch", self.args.dataset_type, "testing/sketch_flow")
            self.train_dataset = DataLoaderSketchFlow(self.train_folder, self.sketch_flow_train_folder,
                                                      transforms.Compose([transforms.ToTensor()]), resize_height=self.args.h,
                                                      resize_width=self.args.w, time_step=self.args.t_length-1,
                                                      which_sketch_flow=self.args.which_sketch_flow)
            self.test_dataset = DataLoaderSketchFlow(self.test_folder, self.sketch_flow_test_folder,
                                                     transforms.Compose([transforms.ToTensor()]), resize_height=self.args.h,
                                                     resize_width=self.args.w, time_step=self.args.t_length - 1,
                                                      which_sketch_flow=self.args.which_sketch_flow)
            self.train_batch = data.DataLoader(self.train_dataset, batch_size=self.args.batch_size,
                                               shuffle=True, num_workers=self.args.num_workers,
                                               drop_last=True, collate_fn=self.train_dataset.collate_fn)
            self.test_batch = data.DataLoader(self.test_dataset, batch_size=self.args.test_batch_size,
                                              shuffle=False, num_workers=self.args.num_workers_test,
                                              drop_last=False, collate_fn=self.train_dataset.collate_fn)
        else:
            self.train_dataset = DataLoader(self.train_folder, transforms.Compose([transforms.ToTensor()]),
                                            resize_height=self.args.h, resize_width=self.args.w,
                                            time_step=self.args.t_length - 1)
            self.test_dataset = DataLoader(self.test_folder, transforms.Compose([transforms.ToTensor()]),
                                           resize_height=self.args.h, resize_width=self.args.w,
                                           time_step=self.args.t_length - 1)
            self.train_batch = data.DataLoader(self.train_dataset, batch_size=self.args.batch_size,
                                               shuffle=True, num_workers=self.args.num_workers, drop_last=True)
            self.test_batch = data.DataLoader(self.test_dataset, batch_size=self.args.test_batch_size,
                                              shuffle=False, num_workers=self.args.num_workers_test, drop_last=False)
            pass
        self.train_size = len(self.train_dataset)
        self.test_size = len(self.test_dataset)

        # Model setting
        if self.args.method == 'pred':
            if self.args.has_sketch_flow:
                gcn_nets = nn.ModuleList([MyGCNNet(node_dim=4, in_dim=128, hidden_dims=self.args.hidden_dims, out_dim=512,
                                     gnn=self.args.which_gnn) for one in self.args.which_sketch_flow])
                self.model = ConvAESketchFlow(self.args.c, self.args.t_length, self.args.msize,
                                              self.args.fdim, self.args.mdim, gcn_nets=gcn_nets)
            else:
                self.model = ConvAEPred(self.args.c, self.args.t_length, self.args.msize, self.args.fdim, self.args.mdim)
                pass
        else:
            self.model = ConvAERecon(self.args.c, memory_size=self.args.msize, feature_dim=self.args.fdim, key_dim=self.args.mdim)
            pass

        self.params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters())
        if self.args.has_sketch_flow:
            for gcn in self.model.gcns:
                self.params += list(gcn.parameters())
            pass

        self.optimizer = torch.optim.Adam(self.params, lr=self.args.lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs)
        self.model.to(args.device)

        # Training
        self.loss_func_mse = nn.MSELoss(reduction='none')
        self.m_items = F.normalize(torch.rand((self.args.msize, self.args.mdim), dtype=torch.float), dim=1).to(args.device)
        pass

    def train(self):
        # self.test(epoch=0)

        max_acc = 0.0
        m_items = self.m_items
        for epoch in range(self.args.epochs):
            self.model.train()
            for j, now_data in enumerate(self.train_batch):

                if self.args.method == 'pred':
                    if self.args.has_sketch_flow:
                        imgs = Variable(now_data[0]).to(self.args.device)
                        batched_graph, nodes_num_norm_sqrt, edges_num_norm_sqrt = now_data[1:4]
                        batched_graph = batched_graph.to(self.args.device)
                        nodes_feat = batched_graph.ndata['feat'].to(self.args.device)
                        edges_feat = batched_graph.edata['feat'].to(self.args.device)
                        nodes_num_norm_sqrt = nodes_num_norm_sqrt.to(self.args.device)
                        edges_num_norm_sqrt = edges_num_norm_sqrt.to(self.args.device)

                        if not self.train_dataset.is_two_sketch_flow:
                            (outputs, _, _, m_items, _, _, separateness_loss, compactness_loss) = self.model.forward(
                                imgs[:, 0:12], m_items, batched_graph, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt, True)
                        else:
                            batched_graph_2, nodes_num_norm_sqrt_2, edges_num_norm_sqrt_2 = now_data[4:]
                            batched_graph_2 = batched_graph_2.to(self.args.device)
                            nodes_feat_2 = batched_graph_2.ndata['feat'].to(self.args.device)
                            edges_feat_2 = batched_graph_2.edata['feat'].to(self.args.device)
                            nodes_num_norm_sqrt_2 = nodes_num_norm_sqrt_2.to(self.args.device)
                            edges_num_norm_sqrt_2 = edges_num_norm_sqrt_2.to(self.args.device)
                            (outputs, _, _, m_items, _, _, separateness_loss, compactness_loss) = self.model.forward(
                                imgs[:, 0:12], m_items, batched_graph, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt, True,
                                batched_graph_2, nodes_feat_2, edges_feat_2, nodes_num_norm_sqrt_2, edges_num_norm_sqrt_2)
                            pass
                    else:
                        imgs = Variable(now_data).to(self.args.device)
                        (outputs, _, _, m_items, _, _, separateness_loss, compactness_loss) = self.model.forward(imgs[:, 0:12], m_items, True)
                        pass
                else:
                    imgs = Variable(now_data).to(self.args.device)
                    (outputs, _, _, m_items, _, _, separateness_loss, compactness_loss) = self.model.forward(imgs, m_items, True)

                self.optimizer.zero_grad()
                if self.args.method == 'pred':
                    loss_pixel = torch.mean(self.loss_func_mse(outputs, imgs[:, 12:]))
                else:
                    loss_pixel = torch.mean(self.loss_func_mse(outputs, imgs))

                loss = loss_pixel + self.args.loss_compact * compactness_loss + self.args.loss_separate * separateness_loss
                loss.backward(retain_graph=True)
                self.optimizer.step()
                pass

            self.scheduler.step()

            Tools.print('----------------------------------------')
            Tools.print('Epoch: {}'.format(epoch + 1))

            # Save the model and the memory items
            # self.save_model(epoch=epoch)

            if self.args.method == 'pred':
                Tools.print('Loss: Prediction {:.6f}/ Compactness {:.6f}/ Separateness {:.6f}'.format(
                    loss_pixel.item(), compactness_loss.item(), separateness_loss.item()))
            else:
                Tools.print('Loss: Reconstruction {:.6f}/ Compactness {:.6f}/ Separateness {:.6f}'.format(
                    loss_pixel.item(), compactness_loss.item(), separateness_loss.item()))
                pass

            if (epoch + 1) % 2 == 0:
                acc = self.test(epoch=epoch)
                if acc > max_acc:
                    max_acc = acc
                pass

            Tools.print('----------------------------------------')
            pass

        # Save and test the final model
        # self.save_model()
        self.test()
        Tools.print(max_acc)
        pass

    def test(self, epoch=-1):
        labels = np.load('./data/frame_labels_' + self.args.dataset_type + '.npy')

        videos = OrderedDict()
        videos_list = sorted(glob.glob(os.path.join(self.test_folder, '*')))
        for video in videos_list:
            video_name = video.split('/')[-1]
            videos[video_name] = {}
            videos[video_name]['path'] = video
            videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
            videos[video_name]['frame'].sort()
            videos[video_name]['length'] = len(videos[video_name]['frame'])
            pass

        labels_list = []
        label_length = 0
        psnr_list = {}
        feature_distance_list = {}

        Tools.print('Evaluation of {} in epoch {}'.format(self.args.dataset_type, epoch + 1))
        # Setting for video anomaly detection
        for video in sorted(videos_list):
            video_name = video.split('/')[-1]
            if self.args.method == 'pred':
                labels_list = np.append(labels_list, labels[0][4 + label_length:videos[video_name]['length'] + label_length])
            else:
                labels_list = np.append(labels_list, labels[0][label_length:videos[video_name]['length'] + label_length])
            label_length += videos[video_name]['length']
            psnr_list[video_name] = []
            feature_distance_list[video_name] = []
            pass

        label_length = 0
        video_num = 0
        label_length += videos[videos_list[video_num].split('/')[-1]]['length']

        self.model.eval()
        m_items_test = self.m_items.clone()
        for k, now_data in enumerate(self.test_batch):
            if self.args.method == 'pred':
                if k == label_length - 4 * (video_num + 1):
                    video_num += 1
                    label_length += videos[videos_list[video_num].split('/')[-1]]['length']
            else:
                if k == label_length:
                    video_num += 1
                    label_length += videos[videos_list[video_num].split('/')[-1]]['length']
                pass

            if self.args.method == 'pred':
                if self.args.has_sketch_flow:

                    imgs = Variable(now_data[0]).to(self.args.device)
                    batched_graph, nodes_num_norm_sqrt, edges_num_norm_sqrt = now_data[1:4]
                    batched_graph = batched_graph.to(self.args.device)
                    nodes_feat = batched_graph.ndata['feat'].to(self.args.device)
                    edges_feat = batched_graph.edata['feat'].to(self.args.device)
                    nodes_num_norm_sqrt = nodes_num_norm_sqrt.to(self.args.device)
                    edges_num_norm_sqrt = edges_num_norm_sqrt.to(self.args.device)

                    if not self.train_dataset.is_two_sketch_flow:
                        (outputs, feas, _, m_items_test, _, _, _, _, _, compactness_loss) = self.model.forward(
                            imgs[:, 0:12], m_items_test, batched_graph, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt, False)
                    else:
                        batched_graph_2, nodes_num_norm_sqrt_2, edges_num_norm_sqrt_2 = now_data[4:]
                        batched_graph_2 = batched_graph_2.to(self.args.device)
                        nodes_feat_2 = batched_graph_2.ndata['feat'].to(self.args.device)
                        edges_feat_2 = batched_graph_2.edata['feat'].to(self.args.device)
                        nodes_num_norm_sqrt_2 = nodes_num_norm_sqrt_2.to(self.args.device)
                        edges_num_norm_sqrt_2 = edges_num_norm_sqrt_2.to(self.args.device)
                        (outputs, feas, _, m_items_test, _, _, _, _, _, compactness_loss) = self.model.forward(
                            imgs[:, 0:12], m_items_test, batched_graph, nodes_feat, edges_feat, nodes_num_norm_sqrt, edges_num_norm_sqrt, False,
                            batched_graph_2, nodes_feat_2, edges_feat_2, nodes_num_norm_sqrt_2, edges_num_norm_sqrt_2)
                        pass
                else:
                    imgs = Variable(now_data).to(self.args.device)
                    (outputs, feas, _, m_items_test, _, _, _, _, _, compactness_loss) = self.model.forward(imgs[:, 0:3 * 4], m_items_test, False)
                    pass
                mse_imgs = torch.mean(self.loss_func_mse((outputs[0] + 1) / 2, (imgs[0, 3 * 4:] + 1) / 2)).item()
                mse_feas = compactness_loss.item()
                point_sc = point_score(outputs, imgs[:, 3 * 4:])
            else:
                imgs = Variable(now_data).to(self.args.device)
                (outputs, feas, _, m_items_test, _, _, compactness_loss) = self.model.forward(imgs, m_items_test, False)
                mse_imgs = torch.mean(self.loss_func_mse((outputs[0] + 1) / 2, (imgs[0] + 1) / 2)).item()
                mse_feas = compactness_loss.item()
                point_sc = point_score(outputs, imgs)
                pass

            if point_sc < self.args.th:
                query = F.normalize(feas, dim=1)
                query = query.permute(0, 2, 3, 1)  # b X h X w X d
                m_items_test = self.model.memory.update(query, m_items_test, False)
                pass

            psnr_list[videos_list[video_num].split('/')[-1]].append(psnr(mse_imgs))
            feature_distance_list[videos_list[video_num].split('/')[-1]].append(mse_feas)
            pass

        # Measuring the abnormality score and the AUC
        anomaly_score_total_list = []
        for video in sorted(videos_list):
            video_name = video.split('/')[-1]
            anomaly_score_total_list += score_sum(anomaly_score_list(psnr_list[video_name]),
                                                  anomaly_score_list_inv(feature_distance_list[video_name]),
                                                  self.args.alpha)
            pass

        anomaly_score_total_list = np.asarray(anomaly_score_total_list)
        accuracy = AUC(anomaly_score_total_list, np.expand_dims(1-labels_list, 0)) * 100

        Tools.print('The result of {} in epoch {}'.format(self.args.dataset_type, epoch + 1))
        Tools.print('AUC: {} %'.format(accuracy))
        return accuracy

    def save_model(self, epoch=-1):
        Tools.print('Training of Epoch {} is finished'.format(epoch + 1))
        if (epoch + 1) % 5 == 0:
            torch.save(self.model, os.path.join(self.log_dir, 'model_{}.pth'.format(epoch + 1)))
            torch.save(self.m_items, os.path.join(self.log_dir, 'keys_{}.pt'.format(epoch + 1)))
            Tools.print('Saving model of {} in {}'.format(epoch + 1, self.log_dir))
            pass
        if epoch < 0:
            Tools.print('Training is finished')
            torch.save(self.model, os.path.join(self.log_dir, 'model.pth'))
            torch.save(self.m_items, os.path.join(self.log_dir, 'keys.pt'))
            Tools.print('Saving final model in {}'.format(self.log_dir))
            pass
        pass

    pass


def get_arg(gpu_id=0, run_name="demo", has_sketch_flow=True, which_sketch_flow=["cluster"], which_gnn=GCNNet, hidden_dims=None):
    parser = argparse.ArgumentParser(description="MNAD")
    parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
    parser.add_argument('--epochs', type=int, default=60, help='number of epochs for training')
    parser.add_argument('--loss_compact', type=float, default=0.1, help='weight of the feature compactness loss')
    parser.add_argument('--loss_separate', type=float, default=0.1, help='weight of the feature separateness loss')
    parser.add_argument('--h', type=int, default=256, help='height of input images')
    parser.add_argument('--w', type=int, default=256, help='width of input images')
    parser.add_argument('--c', type=int, default=3, help='channel of input images')
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--method', type=str, default='pred', help='The target task for anoamly detection')
    parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
    parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
    parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
    parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
    parser.add_argument('--alpha', type=float, default=0.6, help='weight for the anomality score')
    parser.add_argument('--th', type=float, default=0.01, help='threshold for test updating')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
    parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
    parser.add_argument('--dataset_path', type=str, default='./data', help='directory of data')
    parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai')
    parser.add_argument('--exp_dir', type=str, default='{}_{}'.format(gpu_id, run_name), help='directory of log')
    args = parser.parse_args()

    args.device = gpu_setup(use_gpu=True, gpu_id=str(gpu_id))
    args.which_sketch_flow = which_sketch_flow
    args.which_gnn = which_gnn
    args.hidden_dims = hidden_dims
    args.has_sketch_flow = has_sketch_flow
    assert args.method == 'pred' or args.method == 'recon', 'Wrong task name'
    return args


"""
No Sketch Flow, AUC=93.5%
   Sketch Flow, AUC=94.7%
          Half, AUC=95.3%
          Half, AUC=96.6%
         Param, AUC=94.4%
         Param, AUC=96.6%
      Original, AUC=96.3%

   Sketch Flow, AUC=96.1% 4layer GCN track_line
   Sketch Flow, AUC=96.7% 4layer GCN cluster
   Sketch Flow, AUC=96.6% 4layer GraphSage cluster
   Sketch Flow, AUC=95.3% 4layer GatedGCN cluster
   
   Sketch Flow, AUC=96.6% 6layer GCN cluster
   Sketch Flow, AUC=97.0% 6layer GCN cluster
"""


"""
cluster_sage_6layer 96.45, 96.78
cluster_gcn_6layer  93.96, 91.46
seed2022 cluster_GraphSageNet_4layer 95.53
seed2022 cluster_GraphSageNet_6layer 95.28

seed1 No sketch_flow              94.84
seed1 cluster_GraphSageNet_4layer 93.83
seed1 cluster_GraphSageNet_6layer 94.42

seed2 No sketch_flow              95.97
seed2 cluster_GraphSageNet_4layer 96.51
seed2 cluster_GraphSageNet_6layer 96.68
seed2     all_GraphSageNet_4layer 94.13

seed3 No sketch_flow              94.18
seed3 cluster_GraphSageNet_4layer 95.28
seed3 cluster_GraphSageNet_6layer 95.32

seed4 No sketch_flow              94.93
seed4 cluster_GraphSageNet_4layer 94.99
seed4 cluster_GraphSageNet_6layer 95.13
"""


"""
seed2 No sketch_flow              95.15
seed2 cluster_GraphSageNet_4layer 95.53
seed2 cluster_GraphSageNet_6layer 94.08
"""


"""
cd /media/ubuntu/4T2/ubuntu/4T/ALISURE/MNAD
nohup python Runner_SketchFlow.py > ./result/log/ped2/pred/2_seed2_cluster_GraphSageNet_4layer.log 2>&1 &
"""


if __name__ == '__main__':
    seed = 2
    gpu_id = 0
    has_sketch_flow = False
    # which_sketch_flow = ["cluster", "track_line"]
    which_sketch_flow = ["cluster"]  # cluster, track_line
    # which_sketch_flow = "[track_line" ] # cluster, track_line
    which_gnn = GraphSageNet  # GCNNet, GraphSageNet, GatedGCNNet
    hidden_dims = [128, 128, 256, 256]  # [128, 128, 256, 256, 512, 512], [128, 128, 256, 256]
    # hidden_dims = [128, 128, 256, 256, 512, 512]  # [128, 128, 256, 256, 512, 512], [128, 128, 256, 256]

    seed_setup(seed)
    runner = Runner(args=get_arg(
        gpu_id=gpu_id, has_sketch_flow=has_sketch_flow, which_sketch_flow=which_sketch_flow, which_gnn=which_gnn, hidden_dims=hidden_dims,
        run_name="{}_{}_{}layer_{}seed".format("_".join(which_sketch_flow), which_gnn.__name__, len(hidden_dims), seed)))
    runner.train()
    pass

