# TinyChat 150MB

**从随机初始化训练 74.4M 参数中文语言模型的实验项目。**

![74.4M parameters](https://img.shields.io/badge/parameters-74.4M-0969da)
![150 MB class BF16](https://img.shields.io/badge/BF16-150%20MB%20class-1f883d)
![Chinese language modeling](https://img.shields.io/badge/language-Chinese-8250df)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-EE4C2C)

项目记录小参数、短上下文约束下的中文预训练流程，从冻结分词器和混合语料，到训练配方筛选与基座评估。500M 路线的四组 pilot 已完成，正式训练仍在等待语料冻结。

## 模型规格

| 项目 | 规格 |
| --- | --- |
| 架构 | Decoder-only Transformer，使用 RoPE、RMSNorm 和因果 SDPA |
| 参数量 | 74,400,640 |
| 层数与隐藏维度 | 13 层，`d_model = 640` |
| 注意力头数 | 10 |
| FFN 中间维度 | 2,560 |
| 最大上下文 | 512 tokens |
| 分词器 | 冻结的 BPE tokenizer，16,384 词表 |
| 权重共享 | 输入嵌入与输出投影共享权重 |
| BF16 权重大小 | 约 148.8 MB，按历史导出计，仅含模型权重 |

模型结构见 [配置](configs/pilot.json) 和 [模型实现](tinychat/model.py)。

## 500M 实验路线

```mermaid
flowchart LR
    A["混合中文语料"] --> B["清洗、去重与评估集重合筛查"]
    B --> C["沿用冻结 BPE，生成固定语料切分"]
    C --> D["四组 5M-token pilot"]
    D --> E["WSD：预热与稳定训练至 450M"]
    E --> F["学习率衰减 50M"]
    F --> G["基座 LM 指标与 12 条固定续写"]
```

| 阶段 | 当前状态 |
| --- | --- |
| 配方筛选 | 四组 pilot 已完成，选定峰值学习率 `6e-4` 和 `scaled` 残差初始化 |
| 正式预训练 | 混合语料准备中，500M 训练尚未启动。两段训练均采用 micro-batch `8`、梯度累积 `8` |
| 基座评估 | 计划比较历史 100M 基座与新基座在通用、混合验证集上的 LM loss 和困惑度，并检查 12 个固定前缀的续写 |
| SFT | 暂停，待 500M 基座评估完成后再决定 |

历史 100M 基座与其 SFT 分支已完成实验。固定题盲评显示对话成功率仍很低，现有模型尚不能作为中文助手。两个基座的训练预算、语料和配方均有变化，后续评估只能比较整体结果。[第一阶段历史记录](README.txt)

## 四组 pilot 结果

每组提交 **5,013,504** 个有效目标 token，共 **153** 次优化器更新。各组最终验证使用相同的通用目标数（1,000,000）和混合目标数（144,827）。

| 峰值学习率 | 残差初始化 | 通用验证 loss ↓ | 混合验证 loss ↓ | 决策 |
| ---: | --- | ---: | ---: | --- |
| `6e-4` | `original` | 7.6745 | 7.1212 | |
| **`6e-4`** | **`scaled`** | **7.5305** | **6.9424** | **采用** |
| `1e-3` | `original` | 7.7056 | 7.1485 | |
| `1e-3` | `scaled` | 7.5436 | 6.9473 | |

预先约定的规则是保留通用 loss 距最佳值不超过 `0.10` 的配方，再比较混合 loss。混合 loss 相差不超过 `0.02` 时优先较低学习率，其次优先 `scaled` 初始化。[可下载的 pilot 指标与口径](docs/experiments/500m-pilots.json)

pilot loss 屏蔽了未使用的输出控制 token，和历史全词表困惑度的计算口径不同。固定的 5M-token 预算支持本轮配方选择，完整的语言建模与续写评估将在 500M 基座训练后进行。

## 运行入口与复现范围

公开快照包含训练与评估源码、依赖版本、冻结 tokenizer、固定评估题库和 pilot 指标。完整训练及部分测试需要另外准备原始或处理后的语料、与本地文件匹配的冻结清单，以及相关模型权重。训练 checkpoint、模型导出物、运行记录和历史 `comparison_v2` 评估的原始清单保存在本机。[本机历史实验和环境说明](README.txt)

已验证的 GPU 环境使用 Windows、PowerShell 7、Python 3.12 和 CUDA 版 PyTorch。在仓库根目录创建虚拟环境，先从 PyTorch 官方 `cu130` 索引安装 `torch==2.12.1`，再装其余依赖。[依赖](requirements.txt) · [已记录的版本](requirements.lock.txt)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -X utf8 -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -X utf8 -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -X utf8 -m tinychat.stage2_data --help
```

本地正式混合语料冻结后，先在新目录运行 stable 阶段，再从 `stable.pt` 开始 decay。基座对照还需要历史 100M 模型的本地导出。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m tinychat.train_stage2 --phase stable `
  --data-dir data/pretrain_stage2_500m --run-dir runs/stage2_500m_stable `
  --config configs/pilot.json --lr 6e-4 --residual-init scaled
.\.venv\Scripts\python.exe -X utf8 -m tinychat.train_stage2 --phase decay `
  --data-dir data/pretrain_stage2_500m --run-dir runs/stage2_500m_decay `
  --config configs/pilot.json --lr 6e-4 --residual-init scaled `
  --branch-from runs/stage2_500m_stable/stable.pt
.\.venv\Scripts\python.exe -X utf8 -m tinychat.train_stage2 export `
  --checkpoint runs/stage2_500m_decay/final.pt --output artifacts/base_500m
.\.venv\Scripts\python.exe -X utf8 -m scripts.evaluate_stage2 base `
  --mixed-data data/pretrain_stage2_500m --old-model artifacts/base_100m_v1 `
  --new-model artifacts/base_500m --output reports/stage2_base_eval
```

| 目录 | 内容 |
| --- | --- |
| [`tinychat/`](tinychat/) | 模型、语料构建、训练、导出与推理模块 |
| [`scripts/`](scripts/) | 数据获取、诊断、基座评估及历史实验工具 |
| [`tests/`](tests/) | 模型行为、数据筛查、checkpoint 恢复与评估协议测试 |
| [`evaluation/`](evaluation/) | 固定评估题库、题库清单与评分说明 |
| [`data/stage1_v1/tokenizer.json`](data/stage1_v1/tokenizer.json) | 第一阶段预训练前冻结、由 500M 路线复用的分词器 |

## 数据来源与许可

500M 混合语料来自 FineWeb2-HQ 中文子集、Chinese FineWeb Edu、Chinese Cosmopedia 和本地中文 Wiki 语料。原始网页文本适用对应数据源的许可及上游条款；OpenCSG 的两份源卡还列出 OpenCSG Community License 和商业使用需取得许可的说明。历史 BELLE SFT 数据的使用范围以原始源卡为准。仓库公开提供研究代码与固定资产，尚未声明额外的开源授权。[FineWeb2-HQ](https://huggingface.co/datasets/epfml/FineWeb2-HQ) · [Chinese FineWeb Edu](https://huggingface.co/datasets/opencsg/chinese-fineweb-edu) · [Chinese Cosmopedia](https://huggingface.co/datasets/opencsg/chinese-cosmopedia) · [BELLE 单轮](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN) · [BELLE 多轮](https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M)
