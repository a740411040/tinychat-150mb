中文小模型训练项目

74,400,640 参数的中文 decoder Transformer，使用冻结的 16,384 词表 BPE tokenizer，context 为 512。BF16 权重导出约 148.8 MB。

当前进度（500M 基座预训练）
四组 pilot 均完成，各训练 5,013,504 个有效目标 token。按预定门槛选定峰值学习率 6e-4 和 scaled residual init。
正式 500M 预训练尚未启动，混合语料正在准备。计划从随机权重开始，先训练 450M stable，再从该阶段最终 checkpoint 分支训练 50M decay。两个阶段均使用 micro-batch 8、梯度累积 8。
当前暂停 SFT。完成 500M 基座训练后，将比较新旧基座在通用及混合验证集上的 LM loss 和困惑度，并检查 12 个固定前缀的续写，再决定是否开展 SFT。现有结果仍属于训练实验，历史对话评估见下文。

500M 流程入口
以下命令从仓库根目录运行。训练须等正式混合语料完成冻结。每步使用新的运行或输出目录。
  .\.venv\Scripts\python.exe -X utf8 -m tinychat.stage2_data --help
  .\.venv\Scripts\python.exe -X utf8 -m tinychat.train_stage2 --phase stable --data-dir data/pretrain_stage2_500m --run-dir runs/stage2_500m_stable --config configs/pilot.json --lr 6e-4 --residual-init scaled
  .\.venv\Scripts\python.exe -X utf8 -m tinychat.train_stage2 --phase decay --data-dir data/pretrain_stage2_500m --run-dir runs/stage2_500m_decay --config configs/pilot.json --lr 6e-4 --residual-init scaled --branch-from runs/stage2_500m_stable/stable.pt
  .\.venv\Scripts\python.exe -X utf8 -m tinychat.train_stage2 export --checkpoint runs/stage2_500m_decay/final.pt --output artifacts/base_500m
  .\.venv\Scripts\python.exe -X utf8 -m scripts.evaluate_stage2 base --mixed-data data/pretrain_stage2_500m --old-model artifacts/base_100m_v1 --new-model artifacts/base_500m --output reports/stage2_base_eval

GitHub 快照包含代码、固定配置、冻结 tokenizer 和用于评估及语料重合筛查的固定题库。原始与处理后的训练语料、语料清单、模型权重、checkpoint、运行输出和内部工作记录均留在本机。克隆后仍需单独准备本地语料与对应清单，才能运行正式训练、基座对照评估及依赖这些数据的测试。
仓库中的 comparison SFT、natural SFT 和 Stage 2 SFT 脚本用于保留历史实验，当前流程止于 500M 基座评估。
500M 语料来源包括本地中文 Wiki、FineWeb2-HQ、Chinese FineWeb Edu 和 Chinese Cosmopedia。后两者的官方源卡同时列出 Apache 2.0 和 OpenCSG Community License，并附商业使用许可说明。实际使用范围见各数据源卡。
  https://huggingface.co/datasets/epfml/FineWeb2-HQ
  https://huggingface.co/datasets/opencsg/chinese-fineweb-edu
  https://huggingface.co/datasets/opencsg/chinese-cosmopedia

第一阶段历史记录（100M 预训练及 SFT）
已完成第一阶段：正式中文语料和分词器冻结、随机初始化的100,000,000个有效目标token预训练，以及20,000例中文对话的一轮SFT。两阶段均使用micro-batch 8、梯度累积8。
基础模型与SFT模型的BF16权重文件各148,811,944字节（148.81 MB，不包含tokenizer、配置和完整训练检查点）。100道固定测试题的AI盲评成功数：基座0/100、SFT1/100；SFT严重失败62/100，尚不能作为可用的中文助手。实际结果、限制和逐题证据见reports/stage1_report.txt。
第一阶段结束后曾暂停训练，完成诊断后才开展上述 500M 路线的 pilot。历史15分钟测速结果保留在reports/pilot_report.txt。
随后已完成只读数据排查和已有权重前向对照，没有加训；根因证据、同预算改进对照方案及后续训练时间估算见reports/diagnosis_v1/diagnosis_report.txt。

使用第一阶段模型
在 pwsh7 中进入仓库根目录后运行：
  .\.venv\Scripts\python.exe -X utf8 -m tinychat.chat --model artifacts/chat_stage1_v1 --prompt '请用两句话介绍你自己。'
默认CUDA、greedy、最多96个新token。--device cpu可用但未做速度承诺；--max-new-tokens范围1..256。输入加输出预算必须不超过512，超长会明确拒绝，不会静默截断历史。
多轮输入使用 --messages-file 路径，文件为UTF-8 JSON数组，例如：
  [{"role":"user","content":"我叫小林。"},{"role":"assistant","content":"你好，小林。"},{"role":"user","content":"我叫什么？"}]
不要在输入内容内嵌<|user|>等角色控制标记。基础模型路径为artifacts/base_100m_v1；两者共用已冻结tokenizer和聊天格式。
模型按本项目tinychat/model.py加载；这些导出文件尚未适配为Transformers或Ollama即插即用格式。

第一阶段产物
  data/stage1_v1/               FineWeb2-HQ中文正式分片、128.24M训练token、冻结的16K BPE
  data/sft_stage1_v1/           BELLE训练20000/验证500/测试500对话及来源清单
  runs/pretrain_100m_v1/        恰好100M有效目标，最终step_00003052.pt，保留完整优化器和RNG
  runs/sft_stage1_v1/           一轮20000例，最终step_00000313.pt，1,633,508个assistant目标
  artifacts/base_100m_v1/       基础模型BF16权重、tokenizer、配置、聊天格式及导出证据
  artifacts/chat_stage1_v1/     SFT模型BF16权重、tokenizer、配置、聊天格式及导出证据
  evaluation/stage1_v1/         冻结的200开发题+100测试题、两版回答、自动诊断和盲评
  reports/stage1_report.txt     实际训练资源、对话评估结论、限制和后续决策依据
  reports/stage1_execution_review.txt  独立Sol对实际产物的审核
  planning/train_150mb/         冻结方案、来源版本、执行及委派记录

正式数据与训练约束
预训练使用536,950,622字节清洗后的中文网页文本，先做规范化精确去重、按文档分组划分，再仅用TRAIN中的128 MiB文本拟合分词器。149,640篇文档的train/validation/test分别为146,640/1,466/1,534；语料、tokenizer和评估题身份已冻结。
FineWeb2-HQ源页：https://huggingface.co/datasets/epfml/FineWeb2-HQ 。原始网页质量参差，规则过滤不等于逐篇人工审核，精确去重不能排除全部近似改写。
SFT来源：https://huggingface.co/datasets/BelleGroup/train_0.5M_CN 和 https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M 。发布方说明研究使用、非商业用途等限制；本次SFT按研究实验使用，不能据此当作无条件商用许可。
SFT使用18,000单轮+2,000多轮训练例，仅assistant回答及其结束标记参与loss，pad和user/system不计入目标。SFT从100M基座权重新建优化器，不覆盖基座。独立验证集和固定测试题不参与梯度训练；500例SFT test留存未用于选择模型。
冻结配置为configs/pilot.json（文件名沿用，结构与正式模型相同）；正式预训练从随机权重开始，没有续训测速模型。

第一阶段历史复现入口（仅在明确需要重跑旧实验时执行，使用新目录）
  .\.venv\Scripts\python.exe -X utf8 -m tinychat.train_stage1 --stage pretrain --data-dir data/stage1_v1 --run-dir runs/pretrain_reproduction --config configs/pilot.json --micro-batch 8 --grad-accum 8 --target-tokens 100000000
  .\.venv\Scripts\python.exe -X utf8 -m tinychat.train_stage1 --stage sft --data-dir data/sft_stage1_v1 --run-dir runs/sft_reproduction --config configs/pilot.json --micro-batch 8 --grad-accum 8 --init-checkpoint runs/pretrain_100m_v1/step_00003052.pt
实际运行参数完整保存在各run的launcher.json。恢复时用--resume指向可信本地checkpoint，并保持配置、语料身份、batch及预算一致；完成态checkpoint不应当作更大预算的隐式入口。正式GPU训练器使用native BF16，本机已验证；未验证Colab T4的FP16训练路径，不能直接照搬为T4可运行承诺。
300题配对评测使用相同聊天模板、greedy、max-new-tokens 96和batch-size 1。开发题上的CUDA BF16批量一致性检查未全通过，因此按预定规则两版都采用单条解码。模型盲评不等同真人用户验证，自动重复率也不等同任务正确率。

历史测速入口与公共模块
  configs/pilot.json            已冻结的13层、640维、16K词表模型结构
  tinychat/model.py             causal SDPA、RoPE、RMSNorm、共享输出权重
  tinychat/prepare_data.py      FineWeb2-HQ中文单分片小样本与临时BPE
  tinychat/benchmark.py         60至1200秒短训练，墙钟吞吐和恢复状态
  tinychat/export.py            导出BF16 safetensors并实际加载生成
  tinychat/environment.py       当前CUDA和主机环境记录
  scripts/monitor_gpu.py        有时限的nvidia-smi采样
  tests/                       因果性、重计算梯度、数据游标/恢复/JSON行为测试
  logs/、runs/、artifacts/      安装、环境、训练与导出证据（不纳入Git）
  planning/train_150mb/         已审核的完整训练方案与委派记录

使用环境
在 pwsh7 中从仓库根目录运行以下命令。
所有Python命令都使用项目 .\.venv\Scripts\python.exe -X utf8；不要用全局CPU版torch测速。
全新安装的顺序：python -m venv .venv；从PyTorch官方cu130索引安装torch==2.12.1；再安装requirements.txt。
依赖完整版本以安装后生成的requirements.lock.txt为准。CUDA安装包在本次网络环境下可能需使用scripts/download_cuda_wheel.py做有界分段下载；它只接受固定官方包并验证官方索引摘要一次。

准备小样本（联网，只在不存在的输出目录执行）
  $env:HF_HOME = Join-Path $PWD.Path '.cache\huggingface'
  $env:HF_HUB_DISABLE_TELEMETRY = '1'
  $env:HF_HUB_DISABLE_XET = '1'
  .\.venv\Scripts\python.exe -X utf8 -m tinychat.prepare_data --output data/pilot --max-mib 32 --max-seconds 300
32 MiB是采纳的清洗后文本上限，不是Parquet网络字节上限。数据来源、revision、文件和split统计写入meta.json。
先去重、按文档分train/validation/test，再只用train拟合16384词表；本次分词器只是小样本测速用途。
源数据为epfml/FineWeb2-HQ的cmn_Hani配置，ODC-By1.0并适用Common Crawl条款。
正式第一阶段已另建data/stage1_v1和随机初始化模型；不要更换分词器后直接续训测速checkpoint。扩大训练时仍需审查语料覆盖和质量。

运行有限时长测速（会使用GPU；每次选择一个新的run目录）
  .\.venv\Scripts\python.exe -X utf8 -m tinychat.environment
  .\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests
  .\.venv\Scripts\python.exe -X utf8 -u -m tinychat.benchmark --seconds 720 --run-dir runs/pilot_12min --micro-batch 1 --grad-accum 64
默认序列512；每次optimizer update提交32768个token。计时在训练循环内，初始化、初末验证、保存时间单独处理。
到时在microbatch边界停止，尚未提交的累积更新回滚游标与RNG、丢弃梯度，并报告discarded_tokens。
只把成功提交的token计入有效训练量。metrics中的effective_tokens_per_second使用整段训练墙钟；post_warmup_effective_tokens_per_second为去掉最初两次更新后的墙钟口径；steady_state_tokens_per_second仅为成功更新compute口径，不能单独当作端到端预算。
本次schedule仅用于测速：前2次update warmup，然后固定lr=3e-4，不能直接替代正式训练的token预算调度。

验证续训（同一份本机数据、同一micro-batch与累积设置）
  .\.venv\Scripts\python.exe -X utf8 -u -m tinychat.benchmark --seconds 60 --resume runs/pilot_12min --run-dir runs/pilot_resume --micro-batch 1 --grad-accum 64
checkpoint保存FP32模型、AdamW、scheduler/scaler、随机状态、游标与更新计数。
恢复校验来源元数据和本地文件size/mtime；这是本机数据身份检查，不是防篡改的加密保证，跨机器迁移需单独处理。
torch.load仅用于本项目自己生成的可信checkpoint；不要加载来历不明的pickle检查点。

导出
  .\.venv\Scripts\python.exe -X utf8 -m tinychat.export --checkpoint runs/pilot_resume/checkpoint.pt --tokenizer data/pilot/tokenizer.json --output artifacts/pilot_bf16
最终model.safetensors仅保存一次embedding；导出后检查实际体积、重载权重并生成短样例。
完整checkpoint约0.9 GB，不受最终150 MB权重限制。下载分段、检查点和日志默认保留，本项目不会自动删除文件。
