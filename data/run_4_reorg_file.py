"""
整理线流结果文件
"""

import os
import glob
import shutil
from tqdm import tqdm
from alisuretool.Tools import Tools


def re_org(root_path, path_split="10_40_25", sketch_flow_name="sketch_flow_simple", result_sketch_flow_name="sketch_flow"):
    all_txt = glob.glob(os.path.join(root_path, "**/*.txt"), recursive=True)
    for one in tqdm(all_txt):
        path_split_result = one.split("\\{}".format(path_split))
        file_name = os.path.basename(path_split_result[1])
        result_name = os.path.join(path_split_result[0].replace(sketch_flow_name, result_sketch_flow_name), file_name)
        # os.unlink(result_name)  # 删除文件
        if not os.path.exists(result_name):
            shutil.copy(one, Tools.new_dir(result_name))  # 复制文件
            # print(one, result_name)
        pass
    pass


def one():
    # split_data = "testing"
    split_data = "training"
    # root_path = "D:\\Pycharm\\File\\sketch_flow\\UVAD\\MNAD\\data\\sketch_10_40_25\\ped2\\{}\\sketch_flow_simple\\9_40_8".format(split_data)
    root_path = "./sketch_10_40_25/ped2/{}/sketch_flow_simple/9_40_8".format(split_data)

    re_org(root_path=root_path, path_split="10_40_25", sketch_flow_name="sketch_flow_simple",
           result_sketch_flow_name="sketch_flow")
    pass


def two():
    _sketch_param = "25_40_25"
    sketch_flow_name = "sketch_flow_before_t"

    # _run_param = ["5_40_8", "7_40_8", "9_40_8", "11_40_8", "13_40_8"]
    _run_param = ["9_20_8", "9_30_8", "9_40_8", "9_50_8", "9_60_8"]
    for param in _run_param:
        for split_data in ["testing", "training"]:
            root_path = "D:\\Pycharm\\File\\sketch_flow\\UVAD\\MNAD\\data\\sketch_{}\\ped2\\{}\\{}\\{}".format(_sketch_param, split_data, sketch_flow_name, param)
            print(root_path)
            re_org(root_path=root_path, path_split=_sketch_param, sketch_flow_name=sketch_flow_name,
                   result_sketch_flow_name="sketch_flow_abl_cluster")
            pass
        pass
    pass


def three(sketch_flow_name, result_sketch_flow_name):
    _sketch_param = "25_40_25"
    _run_param = ["9_40_8"]
    for param in _run_param:
        for split_data in ["testing", "training"]:
            root_path = "D:\\Pycharm\\File\\sketch_flow\\UVAD\\MNAD\\data\\sketch_{}\\ped2\\{}\\{}\\{}".format(_sketch_param, split_data, sketch_flow_name, param)
            print(root_path)
            re_org(root_path=root_path, path_split=_sketch_param,
                   sketch_flow_name=sketch_flow_name, result_sketch_flow_name=result_sketch_flow_name)
            pass
        pass
    pass


def four(sketch_flow_name, result_sketch_flow_name):
    _sketch_param = "25_40_25"
    _run_param = ["9_40_8"]
    for param in _run_param:
        # for split_data in ["testing", "training"]:
        for split_data in ["training"]:
            root_path = "D:\\Pycharm\\File\\sketch_flow\\UVAD\\MNAD\\data\\sketch_{}\\sht\\{}\\{}\\{}".format(_sketch_param, split_data, sketch_flow_name, param)
            print(root_path)
            re_org(root_path=root_path, path_split=_sketch_param,
                   sketch_flow_name=sketch_flow_name, result_sketch_flow_name=result_sketch_flow_name)
            pass
        pass
    pass


if __name__ == '__main__':
    # one()  # 原始的代码
    # two()  # 重新整理叠加帧的参数的线流
    # three(sketch_flow_name="sketch_flow_remove_dsl", result_sketch_flow_name="sketch_flow_abl_remove_dsl")
    # three(sketch_flow_name="sketch_flow_remove_nsl", result_sketch_flow_name="sketch_flow_abl_remove_nsl")
    # three(sketch_flow_name="sketch_flow_remove_dsl_and_nsl", result_sketch_flow_name="sketch_flow_abl_remove_dsl_and_nsl")
    four(sketch_flow_name="sketch_flow_first", result_sketch_flow_name="sketch_flow_first_ok")
    pass


