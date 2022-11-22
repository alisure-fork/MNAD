
### 运行步骤

* 选择合适的素描图参数：`./data/run_1_sketch_for_select_params.py`

* 跑素描图: `./data/run_2_sketch_for_sketch_flow.py`

* 跑线流: `./SketchFlow/src/o_generate_sf_ped2_simple.py`

* 跑无监督视频异常检测: `./Runner_SketchFlow.py`

* 跑无监督视频异常检测的消融实验: `./Runner_SketchFlow_Batch.py`


### 跑线流

* `./data/run_3_check_none_file.py`: 检查线流结果是否有空文件

* `./data/run_4_reorg_file.py`: 重新整理线流文件


### 画论文中的图

* `./SketchFlow/draw_graph.py`: 画生成的图Graph

*  `./Runner_Paper_Draw.py`: 画消融实验的图


