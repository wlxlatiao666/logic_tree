# Logic tree代码运行方式

目前dataset只有gsm8k（数学）和reclor（逻辑推理）

## 单个token分支模式

**首先checkout到single_token分支下，然后新建一个自己的分支（不要改原分支上的代码）**

运行generate.py指定dataset和test_size，例如

```bash
python generate.py --dataset gsm8k --test_size 100
```

即测试gsm8k的前100条数据

生成的文件在`results/{dataset}`路径下，包含相关指标和叶子节点文本等

运行metric.py计算相关metric，例如

```bash
python metric.py --dataset gsm8k --file_name logic_tree_results_100.json
```

运行后会在同一路径下生成一个名为`{file_name}_metrics.json`的文件，其中包括相关metric指标

然后运行auroc.py，同样需要指定dataset和file_name

```bash
python auroc.py --dataset gsm8k --file_name logic_tree_results_100_metrics.json
```

会以类似表格的形式输出所有uncertainty指标和label之间的auroc

## 滑动窗口模式

**首先checkout到sliding_window_wlx分支下，然后新建一个自己的分支（不要改原分支上的代码）**

运行generate.py指定dataset和test_size，例如

```bash
python generate.py --dataset gsm8k --test_size 100
```

即测试gsm8k的前100条数据

生成的文件在`results/{dataset}`路径下，包含相关指标和叶子节点文本等

运行auroc.py测试auroc，需要指定文件路径，例如

```bash
python auroc.py --result_file ./results/gsm8k/logic_tree_results_100.json
```

如果要更换uncertainty指标，记得修改auroc.py源码