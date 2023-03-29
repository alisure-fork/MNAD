import os
import glob
import shutil


def check_none_file(root_path):
    # 检查是否为空
    all_txt = glob.glob(os.path.join(root_path, "**/*.txt"), recursive=True)
    for one in all_txt:
        with open(one) as f:
            now_txt = f.readlines()
            if len(now_txt) < 1:
                print(one)
            pass
        pass
    pass


def generate_none_file_content(root_path, image_root):
    # 检查文件是否够
    all_video = glob.glob(os.path.join(image_root, "*"))
    for one_video in all_video:
        all_file = glob.glob(os.path.join(one_video, "*.jpg"))
        all_name = sorted([os.path.basename(one_file).split(".")[0] for one_file in all_file])
        all_txt_file = glob.glob(os.path.join(root_path, os.path.basename(one_video),  "**/*.txt"), recursive=True)
        z_fill = 4 if len(all_txt_file) > 999 else 3
        all_txt_name = sorted([os.path.basename(one_txt).split(".")[0].zfill(z_fill) for one_txt in all_txt_file])
        for one_name in all_name:
            if one_name not in all_txt_name:
                print(one_video, one_name)  # 缺少

                # 找到最近的文件
                source_index = int(one_name)
                source_txt_file = os.path.join(os.path.split(all_txt_file[-1])[0], "{}.txt".format(source_index))
                while source_index > 0:
                    source_index -= 1
                    source_txt_file = os.path.join(os.path.split(all_txt_file[-1])[0], "{}.txt".format(source_index))
                    if os.path.exists(source_txt_file):
                        break
                result_txt_file = os.path.join(os.path.split(all_txt_file[-1])[0], "{}.txt".format(int(one_name)))
                # 复制
                shutil.copy(source_txt_file, result_txt_file)
                print(source_txt_file, result_txt_file)
            pass
    pass


def one():
    _root_path = "D:\\Pycharm\\File\\sketch_flow\\UVAD\\MNAD\\data"
    _run_sketch = "sketch_10_40_25"
    _run_name = "sketch_flow_simple"
    _run_param = "9_40_8"
    for split_data in ["testing", "training"]:
        root_path = os.path.join(_root_path, "{}\\ped2\\{}\\{}\\{}".format(_run_sketch, split_data, _run_name, _run_param))
        check_none_file(root_path=root_path)
        generate_none_file_content(
            root_path=root_path, image_root=os.path.join(_root_path, "ped2\\{}\\frames".format(split_data)))
        pass
    pass


def two():
    _root_path = "D:\\Pycharm\\File\\sketch_flow\\UVAD\\MNAD\\data"
    _run_sketch = "sketch_25_40_25"
    _run_name = "sketch_flow_before_t"
    # _run_param = ["5_40_8", "7_40_8", "9_40_8", "11_40_8", "13_40_8"]
    _run_param = ["9_20_8", "9_30_8", "9_40_8", "9_50_8", "9_60_8"]
    for param in _run_param:
        for split_data in ["testing", "training"]:
            root_path = os.path.join(_root_path, "{}\\ped2\\{}\\{}\\{}".format(_run_sketch, split_data, _run_name, param))
            print(root_path)
            check_none_file(root_path=root_path)
            generate_none_file_content(
                root_path=root_path, image_root=os.path.join(_root_path, "ped2\\{}\\frames".format(split_data)))
            pass
    pass


def three(_run_name):
    _root_path = "D:\\Pycharm\\File\\sketch_flow\\UVAD\\MNAD\\data"
    _run_sketch = "sketch_25_40_25"
    _run_param = ["9_40_8"]
    for param in _run_param:
        for split_data in ["testing", "training"]:
            root_path = os.path.join(_root_path, "{}\\ped2\\{}\\{}\\{}".format(_run_sketch, split_data, _run_name, param))
            print(root_path)
            check_none_file(root_path=root_path)
            generate_none_file_content(
                root_path=root_path, image_root=os.path.join(_root_path, "ped2\\{}\\frames".format(split_data)))
            pass
    pass


def four(_run_name):
    _root_path = "D:\\Pycharm\\File\\sketch_flow\\UVAD\\MNAD\\data"
    _run_sketch = "sketch_25_40_25"
    _run_param = ["9_40_8"]
    for param in _run_param:
        for split_data in ["testing", "training"]:
            root_path = os.path.join(_root_path, "{}\\sht\\{}\\{}\\{}".format(_run_sketch, split_data, _run_name, param))
            print(root_path)
            check_none_file(root_path=root_path)
            generate_none_file_content(
                root_path=root_path, image_root=os.path.join(_root_path, "sht\\{}\\frames".format(split_data)))
            pass
    pass


if __name__ == '__main__':
    # one()  # 原始的代码
    # two()  # 检查叠加帧的参数
    # three(_run_name="sketch_flow_remove_dsl")
    # three(_run_name="sketch_flow_remove_nsl")
    # three(_run_name="sketch_flow_remove_dsl_and_nsl")
    four(_run_name="sketch_flow_first")
    pass



