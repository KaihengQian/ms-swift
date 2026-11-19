BUNDLE_DEFAULT="/home/qkh/FunEmbed/20261108_model_eval/funembed_eval_linyan_20261109_v3_2"
MODEL_DEFAULT="/home/qkh/models/Qwen--Qwen3-VL-Embedding-2B"
ADAPTER_DEFAULT="/home/qkh/FunEmbed/20261027_model_train/v1/models/default/v1-20261112-184439"
WORK_DIR_DEFAULT="/home/qkh/FunEmbed/20261108_model_eval/v1"
SPLITS_DEFAULT="internal_test external_test"
GPU_DEFAULT="0,1"
BATCH_DEFAULT=8
GEN_EMB_NUM_DEFAULT=1
MAX_NEW_TOKENS=256
GENERATION_TEMPERATURE=0
TRAIN_PLUGIN="${TRAIN_PLUGIN:-/home/qkh/FunEmbed/ms-swift/scripts/train/v1/scripts/describe_then_embed_plugin.py}"

usage() {
  cat <<'USAGE'
用法: run_eval.sh [--bundle DIR] [--adapter DIR] [--splits "internal_test external_test"]
                  [--gpu "3,2"] [--batch 8] [--model DIR] [--work-dir DIR] [--run-name NAME] [--smoke]
  --bundle    私有输入包目录（含 manifest/classes/texts/bundle.json）
  --adapter   单 checkpoint 目录 或 含 checkpoint-* 的实验目录
  --splits    空格分隔，默认 "internal_test external_test"（各占一张卡并行）
  --gpu       逗号分隔，与 --splits 顺序对应（不足循环）
  --batch     单卡推理批（默认 8；B 模态 7 图实测上限 16）
  --run-name  覆盖默认 run 名（默认 dte_<split 去掉 _test>）
  --smoke     短测（2 query/4 gallery/22 类文本）
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"  # ms-swift 仓库根（framework_path 用）
ENCODER="$SCRIPT_DIR/describe_then_embed_encoder.py"

BUNDLE="$BUNDLE_DEFAULT"; ADAPTER="$ADAPTER_DEFAULT"; MODEL="$MODEL_DEFAULT"
SPLITS="$SPLITS_DEFAULT"; GPU="$GPU_DEFAULT"; BATCH="$BATCH_DEFAULT"
WORK_DIR="$WORK_DIR_DEFAULT"; RUN_NAME=""; SMOKE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle)   BUNDLE="$2"; shift 2 ;;
    --adapter)  ADAPTER="$2"; shift 2 ;;
    --splits)   SPLITS="$2"; shift 2 ;;
    --gpu)      GPU="$2"; shift 2 ;;
    --batch)    BATCH="$2"; shift 2 ;;
    --model)    MODEL="$2"; shift 2 ;;
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --smoke)    SMOKE="--smoke"; shift ;;
    *) usage >&2; echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[ -z "$BUNDLE" ] && { usage >&2; exit 2; }
[ -z "$ADAPTER" ] && { echo "缺少 --adapter" >&2; exit 2; }
ADAPTER="$(cd "$ADAPTER" && pwd)"
command -v funembed-eval >/dev/null || { echo "未找到 funembed-eval（funeval 环境安装）" >&2; exit 2; }

read -r -a SPLIT_ARR <<< "$SPLITS"
read -r -a GPU_ARR <<< "$(echo "$GPU" | tr ',' ' ')"   # "0,1" → (0 1)
export GEN_EMB_NUM="$GEN_EMB_NUM_DEFAULT"

# ---------- checkpoint 收集（单 ckpt 或实验目录全部） ----------
if compgen -G "$ADAPTER/checkpoint-*" >/dev/null; then
  CKPTS=()
  for d in "$ADAPTER"/checkpoint-*; do
    [ -f "$d/adapter_config.json" ] && CKPTS+=("$d")
  done
  [ ${#CKPTS[@]} -eq 0 ] && { echo "$ADAPTER 下无有效 checkpoint-*" >&2; exit 2; }
  IFS=$'\n' CKPTS=($(sort -t- -k2 -n <<< "${CKPTS[*]}")); unset IFS
else
  [ -f "$ADAPTER/adapter_config.json" ] || { echo "$ADAPTER 既不是有效 checkpoint（缺 adapter_config.json），也不是含 checkpoint-* 的实验目录" >&2; exit 2; }
  CKPTS=("$ADAPTER")
fi

# ---------- 单次评测（一个 split × 一个 checkpoint） ----------
run_one() {  # $1=split $2=gpu $3=checkpoint
  local SPLIT="$1" GPU="$2" ADPT="$3"
  local CKPT_REL="${ADPT#*models/}"
  local RUN="${RUN_NAME:-dte_${SPLIT%_test}}"
  local RUN_DIR="$WORK_DIR/runs/$CKPT_REL/$RUN"
  local CONFIG="$RUN_DIR/config.json"

  if [ -e "$RUN_DIR/results" ]; then
    echo "[skip] $RUN_DIR/results 已存在"; return 0
  fi

  # 1) init-lab（within_split: 候选库=query 同测试集；protocol_id 自动带 _within_split）
  funembed-eval init-lab --bundle "$BUNDLE" \
    --model "$MODEL" --adapter "$ADPT" \
    --framework-path "$REPO_ROOT" \
    --model-type qwen3_vl_emb --representation describe_then_embed_two_step \
    --run-dir "$RUN_DIR" --config "$CONFIG" \
    --case-directions C-C O-O B-B O-C C-O --text-modalities C O B \
    --query-split "$SPLIT" --case-protocol within_split --batch-size "$BATCH" $SMOKE

  # 2) patch：encoder 段为两步表征 plugin；开启描述文本导出
  python3 - "$CONFIG" "$ENCODER" "$RUN_DIR" <<'PYEOF'
import json, sys
cfg_path, encoder, run_dir = sys.argv[1], sys.argv[2], sys.argv[3]
c = json.load(open(cfg_path, encoding='utf-8'))
e = c['encoder']
e['backend'] = 'plugin'
e['plugin'] = encoder
e['representation'] = 'describe_then_embed_two_step'
e['representation_protocol'] = ('describe_then_embed_two_step: generate(training-side '
                                'instructions, greedy, max_new_tokens=256) then embed '
                                'generated text (mean over GEN_EMB_NUM trailing <gen_emb> '
                                'final hidden states, L2)')
e['save_descriptions'] = True
e['descriptions_dir'] = run_dir + '/descriptions'
c['descriptions_dir'] = run_dir + '/descriptions'
json.dump(c, open(cfg_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print('   encoder patched -> plugin (descriptions ->', e['descriptions_dir'] + ')')
PYEOF

  # 3) check → encode（生成+嵌入，descriptions 落盘） → evaluate → audit
  funembed-eval check --config "$CONFIG" --files --environment
  CUDA_VISIBLE_DEVICES="$GPU" funembed-eval encode --config "$CONFIG"
  funembed-eval evaluate --config "$CONFIG"
  funembed-eval audit-results --output "$RUN_DIR/results"
  echo "   results : $RUN_DIR/results/"
}

# ---------- 调度：splits 数 ≤ gpus 数时按划分并行，否则串行 ----------
# 每个划分一个后台子进程（该划分的全部 checkpoint 在卡内串行）
PIDS=(); NAMES=()
if [ "${#SPLIT_ARR[@]}" -ge 1 ] && [ "${#GPU_ARR[@]}" -ge "${#SPLIT_ARR[@]}" ] && [ "${#SPLIT_ARR[@]}" -gt 1 ]; then
  for i in "${!SPLIT_ARR[@]}"; do
    SPLIT="${SPLIT_ARR[$i]}"
    FIRST_GPU="${GPU_ARR[$(( i % ${#GPU_ARR[@]} ))]}"
    (
      for ADPT in "${CKPTS[@]}"; do
        run_one "$SPLIT" "$FIRST_GPU" "$ADPT" || { echo "[fail] $ADPT $SPLIT" >&2; continue; }
      done
    ) &
    PIDS+=($!); NAMES+=("$SPLIT@gpu$FIRST_GPU")
  done
  FAIL=0
  for p in "${PIDS[@]}"; do wait "$p" || FAIL=1; done
  [ "$FAIL" -ne 0 ] && exit 1
else
  gi=0
  for SPLIT in "${SPLIT_ARR[@]}"; do
    for ADPT in "${CKPTS[@]}"; do
      GPU_NOW="${GPU_ARR[$(( gi % ${#GPU_ARR[@]} ))]}"
      run_one "$SPLIT" "$GPU_NOW" "$ADPT"
      gi=$(( gi + 1 ))
    done
  done
fi

# ---------- 汇总表（modal 协议，全部 checkpoint × 全部划分） ----------
echo "== 汇总（modal，ndcg@5 / recall@5 / hit@1）=="
python3 - "$WORK_DIR" "${CKPTS[@]}" <<'PYEOF'
import csv, sys
from pathlib import Path
work, ckpts = sys.argv[1], sys.argv[2:]
splits = ['internal_test', 'external_test']
tasks = ['C_C', 'O_O', 'B_B', 'O_C', 'C_O']
print(f"{'checkpoint':<15}{'split':<15}{'task':<7}{'ndcg@5':>9}{'recall@5':>10}{'hit@1':>8}")
for adpt in ckpts:
    rel = adpt.split('models/', 1)[1] if 'models/' in adpt else adpt
    for split in splits:
        run = 'dte_' + split.replace('_test', '')
        p = Path(work) / 'runs' / rel / run / 'results' / 'results_long.csv'
        d = {}
        if p.is_file():
            for r in csv.DictReader(open(p)):
                if r['label_protocol'] == 'modal' and r['metric'] in ('ndcg@5', 'recall@5', 'hit@1'):
                    d[(r['task'], r['metric'])] = float(r['value'])
        if not d:
            print(f"{Path(adpt).name:<15}{split:<15}（无结果）")
            continue
        for task in tasks:
            if not any(k[0] == task for k in d):
                continue
            print(f"{Path(adpt).name:<15}{split:<15}{task:<7}"
                  f"{d.get((task, 'ndcg@5'), float('nan')):>9.4f}"
                  f"{d.get((task, 'recall@5'), float('nan')):>10.4f}"
                  f"{d.get((task, 'hit@1'), float('nan')):>8.4f}")
PYEOF
echo "== done =="
