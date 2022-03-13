import os
import cv2
import dgl
import glob
import torch
import numpy as np
from tqdm import tqdm
import torch.utils.data as data
from collections import namedtuple
from collections import OrderedDict


rng = np.random.RandomState(2020)


def np_load_frame(filename, resize_height, resize_width):
    image_decoded = cv2.imread(filename)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized


class DataLoader(data.Dataset):

    def __init__(self, video_folder, transform, resize_height, resize_width, time_step=4, num_pred=1):
        self.dir = video_folder
        self.transform = transform
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step
        self._num_pred = num_pred
        self.setup()
        self.samples = self.get_all_samples()
        pass
        
    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
            self.videos[video_name]['frame'].sort()
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
            pass
        pass
            
    def get_all_samples(self):
        frames = []
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            for i in range(len(self.videos[video_name]['frame'])-self._time_step):
                frames.append(self.videos[video_name]['frame'][i])
            pass
        return frames
        
    def __getitem__(self, index):
        video_name = self.samples[index].split('/')[-2]
        frame_name = int(self.samples[index].split('/')[-1].split('.')[-2])
        
        batch = []
        for i in range(self._time_step+self._num_pred):
            image = np_load_frame(self.videos[video_name]['frame'][frame_name+i], self._resize_height, self._resize_width)
            if self.transform is not None:
                batch.append(self.transform(image))

        return np.concatenate(batch, axis=0)
        
    def __len__(self):
        return len(self.samples)

    pass


SketchFlowGraphNode = namedtuple("SketchFlowGraphNode",
                                 "index, point, length, angle, displacement, direction")


class SketchFlowGraph(object):

    def __init__(self, sketch_flow_txt, image_size):
        self.sketch_flow_txt = sketch_flow_txt
        self.image_size = image_size

        self.node_data = self.read_txt(self.sketch_flow_txt, self.image_size)
        self.edge_index, self.edge_w = self.set_edge()
        pass

    def __len__(self):
        return len(self.node_data)

    def set_edge(self):
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

    @staticmethod
    def read_txt(sketch_flow_txt, image_size):
        with open(sketch_flow_txt, "r") as f:
            all_txt = f.readlines()

            node_data = []
            for index, one_txt in enumerate(all_txt):
                one_txt = one_txt.split(" ")
                node_one = SketchFlowGraphNode(
                    index=index,
                    point=[float(one) / image_size if float(one) > 0 else 0 for one in one_txt[2].split(",")],
                    length=float(one_txt[4]) / image_size,
                    angle=float(one_txt[3]) / 360,
                    displacement=float(one_txt[6]) / image_size,
                    direction=float(one_txt[5]) / 360)
                node_data.append(node_one)
                pass

            if len(node_data) <= 0:
                print(sketch_flow_txt)

            return node_data
        pass

    pass


class DataLoaderSketchFlow(data.Dataset):

    def __init__(self, video_folder, sketch_flow_folder, transform, resize_height, resize_width, time_step=4, num_pred=1):
        self.video_folder = video_folder
        self.sketch_flow_folder = sketch_flow_folder
        self.transform = transform
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step
        self._num_pred = num_pred

        self.videos = self.setup(self.video_folder, self.sketch_flow_folder, self._resize_width)
        self.samples = self.get_all_samples()
        pass

    @staticmethod
    def setup(video_folder, sketch_flow_folder, image_size):
        videos = OrderedDict()
        video_images = glob.glob(os.path.join(video_folder, '*'))
        for video in tqdm(sorted(video_images)):
            video_name = video.split('/')[-1]
            videos[video_name] = {}
            videos[video_name]['path'] = video
            videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
            videos[video_name]['frame'].sort()
            videos[video_name]['length'] = len(videos[video_name]['frame'])

            videos[video_name]['sketch_flow'] = []
            for frame in videos[video_name]['frame']:
                video_name = frame.split("/")[-2]
                index = int(os.path.splitext(frame.split("/")[-1])[0])
                # sketch_flow_path = os.path.join(
                #     sketch_flow_folder, "{}/25_40_25/{}/9/track_line/{}.txt".format(video_name, video_name, index))
                sketch_flow_path = os.path.join(
                    sketch_flow_folder, "{}/25_40_25/{}/9/cluster/{}.txt".format(video_name, video_name, index))
                assert os.path.exists(sketch_flow_path)
                graph =  SketchFlowGraph(sketch_flow_path, image_size)
                videos[video_name]['sketch_flow'].append(graph)
                pass
            pass
        return videos

    def get_all_samples(self):
        frames = []
        videos = glob.glob(os.path.join(self.video_folder, '*'))
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            for i in range(len(self.videos[video_name]['frame']) - self._time_step):
                frames.append(self.videos[video_name]['frame'][i])
            pass
        return frames

    def __getitem__(self, index):
        video_name = self.samples[index].split('/')[-2]
        frame_name = int(self.samples[index].split('/')[-1].split('.')[-2])

        batch = []
        for i in range(self._time_step + self._num_pred):
            image = np_load_frame(self.videos[video_name]['frame'][frame_name + i], self._resize_height, self._resize_width)
            batch.append(self.transform(image) if self.transform is not None else image)
            pass

        my_graph = self.videos[video_name]['sketch_flow'][frame_name + self._time_step + self._num_pred - 1]
        node_data = my_graph.merge_node_data()
        edge_index, edge_w = my_graph.edge_index, my_graph.edge_w

        graph = dgl.DGLGraph()
        graph.add_nodes(len(my_graph))
        graph.add_edges(edge_index[:, 0], edge_index[:, 1])
        graph.edata['feat'] = torch.from_numpy(edge_w).unsqueeze(1).float()
        graph.ndata['feat'] = torch.from_numpy(node_data).float()

        return np.concatenate(batch, axis=0), graph

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def collate_fn(samples):
        images, graphs = map(list, zip(*samples))
        images = torch.tensor(np.array(images))

        _nodes_num = [graph.number_of_nodes() for graph in graphs]
        _edges_num = [graph.number_of_edges() for graph in graphs]
        nodes_num_norm_sqrt = torch.cat([torch.zeros((num, 1)).fill_(1./num) for num in _nodes_num]).sqrt()
        edges_num_norm_sqrt = torch.cat([torch.zeros((num, 1)).fill_(1./num) for num in _edges_num]).sqrt()
        batched_graph = dgl.batch(graphs)

        return images, batched_graph, nodes_num_norm_sqrt, edges_num_norm_sqrt

    pass

