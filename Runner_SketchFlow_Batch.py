from alisuretool.Tools import Tools

from Runner_SketchFlow import Runner, seed_setup, get_arg
from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import \
    GCNNet, GraphSageNet, GatedGCNNet


def abl_t(seed=2, gpu_id=0):
    has_sketch_flow = True
    which_gnn_list = [GraphSageNet,
                      GCNNet,
                      GatedGCNNet]
    hidden_dims_list = [[128, 256],
                        [128, 128, 256, 256],
                        [128, 128, 256, 256, 512, 512]]
    which_sketch_list = ["sketch_25_40_25",
                         "sketch_10_40_25"]
    which_sketch_flow_list = ["sketch_flow_abl_t/5_40_8",
                              "sketch_flow_abl_t/7_40_8",
                              "sketch_flow_abl_t/9_40_8",
                              "sketch_flow_abl_t/11_40_8",
                              "sketch_flow_abl_t/13_40_8"]

    hidden_dims = hidden_dims_list[1]
    which_sketch = which_sketch_list[0]
    for which_gnn in which_gnn_list[:2]:
        for which_sketch_flow in which_sketch_flow_list:
            seed_setup(seed)
            run_name = "{}_{}seed_{}_{}layer_{}".format(
                which_sketch, seed, which_gnn.__name__, len(hidden_dims),
                which_sketch_flow.replace("/", "_"))
            Tools.print(run_name)
            runner = Runner(args=get_arg(
                gpu_id=gpu_id,
                has_sketch_flow=has_sketch_flow,
                which_gnn=which_gnn,
                hidden_dims=hidden_dims,
                which_sketch=which_sketch,
                which_sketch_flow=which_sketch_flow,
                run_name=run_name))
            Tools.print(runner.log_dir)
            runner.train()
            pass
        pass
    pass


def abl_c(seed=2, gpu_id=0):
    has_sketch_flow = True
    which_gnn_list = [GraphSageNet,
                      GCNNet,
                      GatedGCNNet]
    hidden_dims_list = [[128, 256],
                        [128, 128, 256, 256],
                        [128, 128, 256, 256, 512, 512]]
    which_sketch_list = ["sketch_25_40_25",
                         "sketch_10_40_25"]
    which_sketch_flow_list = ["sketch_flow_abl_cluster/9_20_8",
                              "sketch_flow_abl_cluster/9_30_8",
                              "sketch_flow_abl_cluster/9_40_8",
                              "sketch_flow_abl_cluster/9_50_8",
                              "sketch_flow_abl_cluster/9_60_8"]

    hidden_dims = hidden_dims_list[1]
    which_sketch = which_sketch_list[0]
    for which_gnn in which_gnn_list[:2]:
        for which_sketch_flow in which_sketch_flow_list:
            seed_setup(seed)
            run_name = "{}_{}seed_{}_{}layer_{}".format(
                which_sketch, seed, which_gnn.__name__, len(hidden_dims),
                which_sketch_flow.replace("/", "_"))
            Tools.print(run_name)
            runner = Runner(args=get_arg(
                gpu_id=gpu_id,
                has_sketch_flow=has_sketch_flow,
                which_gnn=which_gnn,
                hidden_dims=hidden_dims,
                which_sketch=which_sketch,
                which_sketch_flow=which_sketch_flow,
                run_name=run_name))
            Tools.print(runner.log_dir)
            runner.train()
            pass
        pass
    pass


def abl_l(seed=2, gpu_id=0):
    has_sketch_flow = True
    which_gnn = GraphSageNet
    hidden_dims_list = [[128, 128],
                        [128, 128, 256, 256],
                        [128, 128, 256, 256, 512, 512],
                        [128, 128, 128, 128, 256, 256, 512, 512]]
    which_sketch = "sketch_25_40_25"
    which_sketch_flow = "sketch_flow/9_40_8"

    for hidden_dims in hidden_dims_list:
        seed_setup(seed)
        run_name = "{}_{}seed_{}_{}layer_{}".format(
            which_sketch, seed, which_gnn.__name__, len(hidden_dims),
            which_sketch_flow.replace("/", "_"))
        Tools.print(run_name)
        runner = Runner(args=get_arg(
            gpu_id=gpu_id,
            has_sketch_flow=has_sketch_flow,
            which_gnn=which_gnn,
            hidden_dims=hidden_dims,
            which_sketch=which_sketch,
            which_sketch_flow=which_sketch_flow,
            run_name=run_name))
        Tools.print(runner.log_dir)
        runner.train()
    pass


def abl_remove(seed=2, gpu_id=0, which_sketch_flow_list=None):
    Tools.print("seed={} gpu id={}".format(seed, gpu_id))
    has_sketch_flow = True
    which_gnn = GraphSageNet
    # hidden_dims = [128, 128, 256, 256, 512, 512]
    # hidden_dims = [128, 128, 256, 256]
    hidden_dims = [128, 128]
    which_sketch = "sketch_25_40_25"
    if which_sketch_flow_list is None:
        # which_sketch_flow_list = ["sketch_flow_abl_remove_dsl/9_40_8",
        #                           "sketch_flow_abl_remove_nsl/9_40_8"]
        which_sketch_flow_list = ["sketch_flow_abl_remove_dsl_and_nsl/9_40_8",
                                  "sketch_flow/9_40_8"]
        pass

    for which_sketch_flow in which_sketch_flow_list:
        seed_setup(seed)
        run_name = "{}_{}seed_{}_{}layer_{}".format(
            which_sketch, seed, which_gnn.__name__, len(hidden_dims),
            which_sketch_flow.replace("/", "_"))
        Tools.print(run_name)
        runner = Runner(args=get_arg(
            gpu_id=gpu_id,
            has_sketch_flow=has_sketch_flow,
            which_gnn=which_gnn,
            hidden_dims=hidden_dims,
            which_sketch=which_sketch,
            which_sketch_flow=which_sketch_flow,
            run_name=run_name))
        Tools.print(runner.log_dir)
        runner.train()
        pass

    pass


if __name__ == '__main__':
    # abl_t()
    # abl_c()
    # abl_l(gpu_id=1)
    # abl_remove(gpu_id=0)
    # abl_remove(gpu_id=0, which_sketch_flow_list=["sketch_flow_abl_remove_nsl/9_40_8"])
    pass

