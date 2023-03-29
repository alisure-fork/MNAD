import os
import glob
import argparse
from utils import *
from tqdm import tqdm
import torch.optim as optim
import torch.utils.data as data
from model.utils import DataLoader
from collections import OrderedDict
from torch.autograd import Variable
from alisuretool.Tools import Tools
from torch.nn import functional as F
import torchvision.transforms as transforms
from model.Reconstruction import convAE as ConvAERecon
from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import convAE as ConvAEPred


def get_arg():
    parser = argparse.ArgumentParser(description="MNAD")
    parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs for training')
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

    parser.add_argument('--dataset_type', type=str, default='sht', help='type of dataset: ped2, avenue, shanghai')
    parser.add_argument('--exp_dir', type=str, default='./result/exp/sht', help='directory of log')

    args = parser.parse_args()

    assert args.method == 'pred' or args.method == 'recon', 'Wrong task name'

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance
    return args


class Runner(object):

    def __init__(self, args):
        self.args = args

        self.log_dir = Tools.new_dir(os.path.join('./exp', self.args.dataset_type, self.args.method, self.args.exp_dir))

        # Loading dataset
        self.train_folder = os.path.join(self.args.dataset_path, self.args.dataset_type, "training/frames")
        self.test_folder = os.path.join(self.args.dataset_path, self.args.dataset_type, "testing/video")
        self.train_dataset = DataLoader(self.train_folder, transforms.Compose([transforms.ToTensor()]),
                                        resize_height=self.args.h, resize_width=self.args.w, time_step=self.args.t_length-1)
        self.test_dataset = DataLoader(self.test_folder, transforms.Compose([transforms.ToTensor()]),
                                       resize_height=self.args.h, resize_width=self.args.w, time_step=self.args.t_length - 1)
        self.train_size = len(self.train_dataset)
        self.test_size = len(self.test_dataset)
        self.train_batch = data.DataLoader(self.train_dataset, batch_size=self.args.batch_size,
                                           shuffle=True, num_workers=self.args.num_workers, drop_last=True)
        self.test_batch = data.DataLoader(self.test_dataset, batch_size=self.args.test_batch_size,
                                          shuffle=False, num_workers=self.args.num_workers_test, drop_last=False)

        # Model setting
        if self.args.method == 'pred':
            self.model = ConvAEPred(self.args.c, self.args.t_length, self.args.msize, self.args.fdim, self.args.mdim)
        else:
            self.model = ConvAERecon(self.args.c, memory_size=self.args.msize, feature_dim=self.args.fdim, key_dim=self.args.mdim)
        self.params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=self.args.lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs)
        self.model.cuda()

        # Training
        self.loss_func_mse = nn.MSELoss(reduction='none')
        self.m_items = F.normalize(torch.rand((self.args.msize, self.args.mdim), dtype=torch.float), dim=1).cuda()
        pass

    def train(self):
        m_items = self.m_items
        for epoch in range(self.args.epochs):
            self.model.train()

            for j, (imgs) in tqdm(enumerate(self.train_batch)):
                imgs = Variable(imgs).cuda()

                if self.args.method == 'pred':
                    (outputs, _, _, m_items, _, _, separateness_loss, compactness_loss) = self.model.forward(imgs[:, 0:12], m_items, True)
                else:
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
            self.save_model(epoch=epoch)

            if self.args.method == 'pred':
                Tools.print('Loss: Prediction {:.6f}/ Compactness {:.6f}/ Separateness {:.6f}'.format(
                    loss_pixel.item(), compactness_loss.item(), separateness_loss.item()))
            else:
                Tools.print('Loss: Reconstruction {:.6f}/ Compactness {:.6f}/ Separateness {:.6f}'.format(
                    loss_pixel.item(), compactness_loss.item(), separateness_loss.item()))
                pass

            self.test(epoch=epoch)
            Tools.print('----------------------------------------')
            pass

        # Save and test the final model
        self.save_model()
        self.test()
        pass

    def test(self, epoch=-1):
        labels = np.load('./data/frame_labels_' + self.args.dataset_type + '.npy')
        if len(labels.shape) == 1:
            labels = np.expand_dims(labels, axis=0)

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
        for k, (imgs) in tqdm(enumerate(self.test_batch)):
            if self.args.method == 'pred':
                if k == label_length - 4 * (video_num + 1):
                    video_num += 1
                    label_length += videos[videos_list[video_num].split('/')[-1]]['length']
            else:
                if k == label_length:
                    video_num += 1
                    label_length += videos[videos_list[video_num].split('/')[-1]]['length']
                pass

            imgs = Variable(imgs).cuda()

            if self.args.method == 'pred':
                (outputs, feas, _, m_items_test, _, _, _, _, _, compactness_loss) = self.model.forward(imgs[:, 0:3 * 4], m_items_test, False)
                mse_imgs = torch.mean(self.loss_func_mse((outputs[0] + 1) / 2, (imgs[0, 3 * 4:] + 1) / 2)).item()
                mse_feas = compactness_loss.item()
                # Calculating the threshold for updating at the test time
                point_sc = point_score(outputs, imgs[:, 3 * 4:])
            else:
                (outputs, feas, _, m_items_test, _, _, compactness_loss) = self.model.forward(imgs, m_items_test, False)
                mse_imgs = torch.mean(self.loss_func_mse((outputs[0] + 1) / 2, (imgs[0] + 1) / 2)).item()
                mse_feas = compactness_loss.item()
                # Calculating the threshold for updating at the test time
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
        accuracy = AUC(anomaly_score_total_list, np.expand_dims(1-labels_list, 0))

        Tools.print('The result of {} in epoch {}'.format(self.args.dataset_type, epoch + 1))
        Tools.print('AUC: {} %'.format(accuracy*100))
        pass

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


"""
69.33%
70.40%
"""


if __name__ == '__main__':
    runner = Runner(args=get_arg())
    runner.test()
    runner.train()
    pass

