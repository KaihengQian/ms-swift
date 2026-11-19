# Describe-then-Embed v1 —— 检索式评测

> 目录：`scripts/eval/`
> 评测对象：`scripts/train/v1` 训练产出的 describe-then-embed checkpoint（基座 + LoRA）
> 本目录自成一体：评测框架本体（实验室交付）、两步表征 plugin encoder、评测入口脚本、环境锁定文件。

---

## 1. 评测方式概述

两步表征（方式 B，与训练范式对齐）：

1. **生成步**：`TransformersEngine(task_type='causal_lm', template='qwen3_vl')` + 基座 + LoRA，用**训练侧指令**（`scripts/train/v1/prompts/`，`DTE_TRAIN_DIR` 可覆盖）：
   - 图像分支按实际图数重建 user 消息（C=1 张 CFP；O=6 张中心 OCT；B=CFP+6 OCT），greedy、`max_new_tokens=256` 生成描述；
   - 文本分支把候选疾病名映射回训练 label（`Diabetic retinopathy.` → `Disease: Diabetic Retinopathy\n…`），按模态取 doc 指令生成描述。
2. **嵌入步**：`TransformersEngine(task_type='embedding', template='qwen3_vl_emb')`，把**生成的描述文本**喂模型，取末尾 `GEN_EMB_NUM` 个 `<gen_emb>` token 位置最终隐藏态的**均值**（L2 归一化；`GEN_EMB_NUM=1` 走框架原 last-valid-token 协议）。`<gen_emb>` 输入 embedding 由 `<eos>` 初始化，与训练一致。

两侧同模型、同 adapter、同 `system="Represent the user's input."`；评测框架自带的 I0/I1 指令不进入模型（生成必须在训练分布内）。

## 2. 检索协议要点

- **query 划分**：只用 `internal_test`（1038 例）/ `external_test`（984 例，山东）。val 已并入训练集，作为 query 会训练-评测泄漏，脚本硬性拒绝。
- **病例检索（候选库与 query 同测试集）**：`gallery_split = query_split`（`case_protocol=within_split`，由入口脚本自动写入；协议 id 自动追加 `_within_split`）。候选自动排除自病例（tie_key）、同患者（center+patient_id）、共享影像内容 MD5 的重复检查；相关性 = 标签集合 Jaccard。
- **文本检索**：query 为影像分支，候选固定 22 类疾病 name/definition，与划分无关。
- **方向**：C→C、O→O、B→B、O→C、C→O + 各模态→文本；modal 与 combined 两种标签协议分别输出。
- **指标**：Hit/Recall/NDCG@1/3/5（等权均值）+ `coverage.csv`（标签有效数、无相关候选数、两种覆盖率）。

## 3. 目录内容

| 文件 | 说明 |
| --- | --- |
| `funembed_eval/` | 评测框架本体（实验室交付，pin 自 `wvw13/ms-swift@4504dad` 并含 within_split 支持；`pip install -e` 后提供 `funembed-eval` 命令；协议详见 `funembed_eval/docs/PROTOCOL.md`、`funembed_eval/docs/MODEL_INPUT.md`） |
| `scripts/describe_then_embed_encoder.py` | **两步表征 plugin encoder**。契约：`create_encoder(config)` → 带 `report: dict` 与 `encode(branches) -> (单位向量矩阵, token 数)` 的对象；评测框架以 `encoder.backend=plugin` 加载 |
| `scripts/run_eval.sh` | **评测入口**（init-lab → patch → check → encode → evaluate → audit 一条龙；双模式见 §5） |
| `requirements.txt` | 评测环境锁定依赖（头注释含复现步骤） |
| `environment.yml` | conda 版环境定义（pip 层引用 requirements.txt） |

**私有输入包不在本仓库**：病例清单（manifest.jsonl，6263 例）、22 类映射（classes.json）、疾病定义（texts.json）按实验室渠道获取后解压到工作目录（如 `/home/qkh/FunEmbed/20261108_model_eval/`）。

## 4. 评测配置项（脚本参数与环境变量）

| 参数/变量 | 默认 | 含义 |
| --- | --- | --- |
| `--bundle` | —（必填） | 私有输入包目录（含 manifest/classes/texts/bundle.json） |
| `--adapter` | —（必填） | 单 checkpoint 目录（只评它）或实验目录（含 `checkpoint-*`，评全部） |
| `--splits` | "internal_test external_test" | 评测划分；空格分隔多个则**并行**（各占一卡） |
| `--gpu` | "3,2" | 与 --splits 顺序一一对应；数量不足时循环使用 |
| `--batch` | 8 | 单卡推理批（生成步+嵌入步同批）；实测上限 16（见 §6） |
| `--model` | Qwen3-VL-Embedding-2B 本机路径 | 基座模型目录（含 tokenizer/processor） |
| `--work-dir` | `/home/qkh/FunEmbed/20261108_model_eval` | 运行产物根目录 |
| `--run-name` | `dte_<split 去掉 _test>` | 覆盖默认 run 名 |
| `GEN_EMB_NUM`（环境变量） | 1 | 末尾 `<gen_emb>` token 数（须与训练一致） |
| `--smoke` | — | 短测：2 query / 4 gallery / 22 类文本 |

关键配置字段（init-lab 自动写入）：`case_protocol=within_split`、`representation=describe_then_embed_two_step`、`encoder.backend=plugin`。

## 5. 单卡 batch 实测（RTX 5880 48G，B 模态分支 = 7 图最重负载）

| batch | 结果 | 峰值显存 | 吞吐 |
| --- | --- | --- | --- |
| 2 / 4 / 8 | ✅ | 11.4 / 14.8 / 21.7 GiB | 2.7 / 1.85 / **1.45 s/条** |
| **16** | ✅ | 35.4 GiB | 1.34 s/条 |
| 17 / 18 / 20 / 24 | ❌ OOM | ~45 GiB | — |

→ **上限 16，默认 8**（16 贴边且吞吐收益递减）。文本分支远轻于图像分支。多卡并行见 §6。

## 6. 使用步骤

```bash
# 1) 环境（见 requirements.txt 头注释；评测框架随本仓库）
conda env create -f environment.yml && conda activate funeval
pip install -e <本仓库根目录>          # ms-swift（推理框架）
pip install -e funembed_eval/          # 评测 CLI

# 2) 单 checkpoint 评测
bash scripts/run_eval.sh --bundle <私有输入包目录> \
    --adapter <训练产物>/models/default/v0-.../checkpoint-835 \
    --splits internal_test --gpu 3

# 3) 整次实验全部 checkpoint，内外测试集各占一卡并行
bash scripts/run_eval.sh --bundle <私有输入包目录> \
    --adapter <训练产物>/models/default/v0-... \
    --splits "internal_test external_test" --gpu "3,2"
```

- 产物落 `<work-dir>/runs/<models/之后的checkpoint链>/dte_<split>/{config.json,encoding/,results/,descriptions/}`：
  - `results/`：`summary.json`、`results_long.csv`（全部指标长表）、`coverage.csv`、`modal|combined/`（逐 query 明细与 score NPZ）；
  - `descriptions/`：**模型生成的描述文本**（`image_C.jsonl`、`doc_B.jsonl` 等，含 prompt 与 description，供人工查看或单独评估）；
  - 重复执行自动跳过已有 `results/` 的 checkpoint（重跑删对应目录）。
- 多 checkpoint 模式跑完自动打印各 checkpoint × 划分 × 方向的 ndcg@5/recall@5/hit@1 汇总表；单点失败不阻断整批，结束统一报告。
- GPU 被占用时可先提交脚本等卡：run_eval.sh 前加轮询（示例见 `20261108_model_eval/auto_eval_gpu3.sh` 的等待循环）。
