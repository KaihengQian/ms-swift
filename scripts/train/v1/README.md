# Describe-then-Embed v1 —— Qwen3-VL-Embedding 描述式训练

> 目录：`scripts/train/v1/`
> 目标模型：`Qwen3-VL-Embedding-2B`（`model_type=qwen3_vl_emb`）
> 本目录自成一体：训练脚本、插件（Loss+Template+评估指标）、数据构建脚本、prompt、环境锁定文件。

---

## 1. 方法概述

Describe-then-Embed：让 embedding 模型先"描述"再"嵌入"。联合损失：

```
L = λ · L_gen + L_contra        （λ = DESC_EMBED_LAMBDA，默认 5）
```

- **L_gen（生成式损失）**：对监督 token（Query caption + Doc 疾病描述）做词表级交叉熵——把原始 lm_head 权重重新作用到最终层隐藏态 `Z = F.linear(H_sup, lm_head.weight)`。λ 乘在生成式项上（生成式学习需要比对比学习更大的有效学习率）。
- **L_contra（对比损失）**：复用 ms-swift `InfonceLoss`（非 batch 模式，每样本 1 正例 + K 个硬负例，τ=0.05）。embedding 取序列末尾 `GEN_EMB_NUM` 个连续 `<gen_emb>` token 位置最终隐藏态的**均值**（L2 归一化；`GEN_EMB_NUM=1` 即原单 token 协议）；`<gen_emb>` 的输入 embedding 初始化为 `<eos>` 的 embedding 行。
- 推理口径（方式 B）：先用训练侧指令生成描述，再把生成的描述文本编码为向量；直接用末 token 表征（方式 A）也兼容。

## 2. 目录内容

| 文件 | 说明 |
| --- | --- |
| `scripts/describe_then_embed_qwen3_vl_emb_2b.sh` | 训练入口（含全部环境变量注释；路径可用环境变量覆盖，默认相对脚本定位，可在任意目录执行） |
| `scripts/describe_then_embed_plugin.py` | **核心插件**：`DescribeThenEmbedLoss`（联合损失+GEN_LOSS_RATIO 调度+组件日志）、`DescribeThenEmbedTemplate`（追加 `<gen_emb>`、gen_labels、DOC_GEN_SUPERVISE 三态、collator）、`DescribeThenEmbedMetrics`（retrieval_top1/mrr 评估指标）。经 `--custom_register_path` 加载，**不改动 ms-swift 核心源码** |
| `prompts/query_instruction_prompt.json` | Query 侧指令（cfp_only / oct_only / cfp_oct_combined） |
| `prompts/doc_instruction_prompt.json` | Doc 侧指令（`{DISEASE_NAME}` 占位符） |
| `requirements.txt` / `environment.yml` | 环境锁定（python 3.10 / torch 2.14.1+cu130 / transformers 5.20.2 / peft 0.20.0 / deepspeed 0.19.9 / kernels 0.17.2） |

## 3. 数据格式

`describe_then_embed_train.jsonl` 每行一个 (样本, 标签) 记录：

```json
{"messages": [{"role":"user","content":"CFP Image: <image>\nDescribe..."},
              {"role":"assistant","content":"<caption>"}],
 "positive_messages": [[{"role":"user","content":"Disease: Diabetic Retinopathy\n..."},
                        {"role":"assistant","content":"<字典描述文本>"}]],
 "negative_messages": [[...], ... 16 个],
 "images": ["/abs/path/fundus.jpg"]}
```

- CFP=1 图；OCT=n 张编号 B-scan（仅样本根目录的 `_NNN.jpg`，octa/ en-face 已排除）；
- 负例为按相似度从高到低的前 K=16 硬负例（`INFONCE_HARD_NEGATIVES` 可在训练时再截短）；
- 数据来源依赖项目内另一数据合成仓库（caption v2 + 描述字典 v1 + 质量评估分数）及其构建脚本，私有数据不在本仓库；本目录只负责训练。

## 4. 训练配置项（环境变量）

| 变量 | 默认 | 含义 |
| --- | --- | --- |
| `DESC_EMBED_LAMBDA` | 5 | 生成式项权重 λ（`L = λ·L_gen + L_contra`；两个目标的相对学习率） |
| `INFONCE_TEMPERATURE` | 0.05 | InfoNCE 温度 |
| `INFONCE_HARD_NEGATIVES` | 16 | 每样本参与对比的前 K 个负例（数据已按最难在前排序） |
| `INFONCE_USE_BATCH` | false | 不做 batch 内负例（负例已显式给定） |
| `DOC_GEN_SUPERVISE` | positive | Doc 侧生成监督范围：`all`（正+负例描述）/ `positive`（仅正例）/ `none`（仅 anchor caption） |
| `GEN_LOSS_RATIO` | 1.0 | L_gen 参与训练的步数比例（`0.2`=前 20% 步，之后纯对比） |
| `GEN_EMB_ATTN` | default | `<gen_emb>` 行注意力消融：`default` / `caption_only` / `image_only` |
| `GEN_EMB_NUM` | 1 | 序列末尾连续 `<gen_emb>` token 数；表征 = 这些位置最终隐藏态均值（L2）。`<gen_emb>` 输入 embedding 由 `<eos>` 初始化 |

关键 CLI：`--loss_type describe_then_embed --template describe_then_embed --new_special_tokens '<gen_emb>' --eval_metric describe_then_embed`。

## 5. 显存与 batch 实测（2×RTX 5880 48G，deepspeed zero2，bf16，max_length 8192，gradient checkpointing）

| per_device_batch | 结果 |
| --- | --- |
| 1 | ✅ 跑通（长样本 4–5 图峰值 ~22.8 GiB） |
| 2 | ❌ 长样本（4–5 图锚点）OOM 46.5 GiB（与 DOC_GEN_SUPERVISE 无关，瓶颈是长序列前向） |
| 4 / 8 | ❌ 普通样本即 OOM |

→ **最大 batch=1 × grad_accum=32（全局 batch 64）**，5 epoch ≈ 835 步。

## 6. 使用步骤

```bash
# 1) 环境（见 requirements.txt 头部注释；本仓库 pip install -e .）
conda create -n swift python=3.10 -y && conda activate swift
pip install -e <本仓库根目录>
pip install -r requirements.txt

# 2) 训练（GPU 卡号、模型/数据/插件路径可用环境变量覆盖；训练数据 JSONL 需自行按 §3 格式准备）
#    快速验证可加 --max_steps 10 小跑
bash scripts/describe_then_embed_qwen3_vl_emb_2b.sh
bash scripts/describe_then_embed_qwen3_vl_emb_2b.sh
```

训练产物：`output_dir` 下 5 个 checkpoint（`save_steps 0.2`），`logging.jsonl` 含总 loss、`loss_gen`、`loss_contrastive`、grad_norm、lr。
