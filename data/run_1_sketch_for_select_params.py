import os
import glob
import shutil
import itertools
from PIL import Image, ImageEnhance
import multiprocessing
from alisuretool.Tools import Tools


# PARM_1 = 25
# PARM_2 = 40
# PARM_3 = 25


class Image2Sketch(object):

    def __init__(self, image_path, result_path, temp_path, result_image_size=None,
                 exe_path="D:\\Pycharm\\File\\sketch_flow\\SketchFlow\\exe\\Sketching_thres_3.exe", param=None):
        self.image_path = image_path
        self.temp_path = Tools.new_dir(temp_path)
        self.result_image_size = result_image_size
        self.exe_path = exe_path
        self.param = param if param is not None else {"PARM_1": 25, "PARM_2": 40, "PARM_3": 25}

        self.result_path = os.path.join(result_path)

        self.path_origin_image = Tools.new_dir(os.path.join(self.result_path, "image_origin"))
        self.path_gray_image = Tools.new_dir(os.path.join(self.result_path, "image_gray"))
        self.image_name_ext = os.path.basename(self.image_path)
        self.image_name = os.path.splitext(self.image_name_ext)[0]

        self.path_sketch_image = Tools.new_dir(os.path.join(self.result_path, "sketch_image"))
        self.sketch_image_filename = "{}_sketch.bmp".format(self.image_name)
        self.path_sketch_txt = Tools.new_dir(os.path.join(self.result_path, "sketch_txt"))
        self.sketch_txt_filename = "{}_1_m{}_graph.txt".format(self.image_name, self.param["PARM_1"])
        pass

    def run_sketch(self, check_has_finish=True):
        # 判断是否已经执行过（断点重启）
        if check_has_finish and os.path.exists(os.path.join(self.path_sketch_txt, self.sketch_txt_filename)):
            return

        # 复制图片数据
        if self.result_image_size:
            tem_dst = os.path.join(self.temp_path, os.path.basename(self.image_path))

            # 正常图片生成素描图
            Image.open(self.image_path).resize(self.result_image_size).save(tem_dst)

            # 有些图片太暗了，所以增强一下
            # im = Image.open(self.image_path).resize(self.result_image_size)
            # bright = ImageEnhance.Brightness(im)
            # im = bright.enhance(1.2)
            # im.save(tem_dst)
        else:
            tem_dst = shutil.copy(self.image_path, self.temp_path)
            pass

        # 执行程序
        # try:
        os.system("{} {} {} {} {}".format(self.exe_path, tem_dst, self.param["PARM_1"],
                                          self.param["PARM_2"], self.param["PARM_3"]))
        # except Exception:
        #     return

        # 复制需要的文件
        try:
            shutil.copy(tem_dst, self.path_origin_image)  # 原始图片
            temp_result_path = os.path.splitext(tem_dst)[0]
            shutil.copy(os.path.join(temp_result_path, self.image_name_ext), self.path_gray_image)  # 灰度图
            shutil.copy(os.path.join(temp_result_path, self.sketch_image_filename),
                        os.path.join(self.path_sketch_image, "{}-{}-{}_{}".format(
                            self.param["PARM_1"], self.param["PARM_2"], self.param["PARM_3"],
                            self.sketch_image_filename)))  # 素描图
            shutil.copy(os.path.join(temp_result_path, self.sketch_txt_filename),
                        os.path.join(self.path_sketch_txt, "{}-{}-{}_{}".format(
                            self.param["PARM_1"], self.param["PARM_2"], self.param["PARM_3"],
                            self.sketch_txt_filename)))  # 素描图txt
        except Exception:
            Tools.print("------------------------------------------------------------------------------")
            Tools.print(self.image_path)
            Tools.print(tem_dst)
            return

        # 删除不需要的文件
        shutil.rmtree(temp_result_path)
        pass

    @staticmethod
    def test():
        image_2_sketch = Image2Sketch(image_path="..\\data\\image\\demo\\0.jpg",
                                      result_path="..\\result\\image\\demo",
                                      temp_path="..\\data_temp\\temp\\demo")
        # image_2_sketch.run_sketch()
        image_2_sketch.run_sketch(check_has_finish=False)
        pass

    pass


class Image2SketchList(object):

    def __init__(self, image_path, result_path, result_image_size=None, param=None):
        self.image_path = image_path
        self.result_path = result_path
        self.result_image_size = result_image_size
        self.temp_path = os.path.join(".\\temp", os.path.basename(image_path))
        self.param = param
        pass

    def _sketch_one(self, image_index, image_one, all_num):
        # Tools.print("#############################################################################")
        # Tools.print("{} {}/{}".format(image_one, image_index, all_num))
        # Tools.print("#############################################################################")

        image_2_sketch = Image2Sketch(image_path=image_one, result_path=self.result_path, temp_path=self.temp_path,
                                      result_image_size=self.result_image_size, param=self.param)
        image_2_sketch.run_sketch()
        pass

    def run_sketch_single(self):
        image_path_list = glob.glob(os.path.join(self.image_path, "*.*"))
        for image_index, image_one in enumerate(image_path_list):
            self._sketch_one(image_index, image_one, len(image_path_list))
            pass
        shutil.rmtree(self.temp_path)
        pass

    def run_sketch_multi(self):
        image_path_list = glob.glob(os.path.join(self.image_path, "*.*"))

        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
        for image_index, image_one in enumerate(image_path_list):
            pool.apply_async(func=self._sketch_one, args=(image_index, image_one, len(image_path_list)))
            pass
        pool.close()
        pool.join()

        shutil.rmtree(self.temp_path)
        pass

    @staticmethod
    def run_all_sketch(video_root, result_root, multi_processing=False, result_image_size=None):
        all_video_name = os.listdir(video_root)
        for video_index, video_name in enumerate(all_video_name):
            # Tools.print("#############################################################################")
            # Tools.print("{} {}/{}".format(video_name, video_index, len(all_video_name)))
            # Tools.print("#############################################################################")

            image_2_sketch_list = Image2SketchList(image_path=os.path.join(video_root, video_name),
                                                   result_path=os.path.join(result_root, video_name),
                                                   result_image_size=result_image_size)

            if multi_processing:
                image_2_sketch_list.run_sketch_multi()
            else:
                image_2_sketch_list.run_sketch_single()

            pass
        pass

    @staticmethod
    def test():
        # one
        image_2_sketch_list = Image2SketchList(image_path="..\\data\\image\\demo",
                                               result_path="..\\result\\image\\demo")
        # one single
        image_2_sketch_list.run_sketch_single()

        # one multi
        image_2_sketch_list.run_sketch_multi()

        # all single
        Image2SketchList.run_all_sketch(video_root="..\\data\\image", result_root="..\\result\\image")

        # all multi
        Image2SketchList.run_all_sketch(video_root="..\\data\\image",
                                        result_root="..\\result\\image", multi_processing=True)
        pass

    pass


if __name__ == '__main__':
    _result_image_size = [256, 256]
    image_name = "055.jpg"
    image_path = "./ped2/testing/frames/01"
    result_root = os.path.join("./sketch_select", image_path)

    # param_1 = list(range(10, 101, 10))
    # for one, two, three in itertools.combinations_with_replacement(param_1, 3):
    param_1 = [[10, 40, 25]]
    for one, two, three in param_1:
        param = {"PARM_1": one, "PARM_2": two, "PARM_3": three}
        Tools.print(param)
        image_2_sketch_list = Image2SketchList(image_path=image_path, result_path=result_root,
                                               result_image_size=_result_image_size, param=param)
        image_2_sketch_list._sketch_one(0, os.path.join(image_path, image_name), 1)
        shutil.rmtree(image_2_sketch_list.temp_path)
        pass
    pass
