#!/usr/bin/env python3
"""评测结果可视化对比工具。

从 runs/ 下各 checkpoint × dte_{internal,external}/results/results_long.csv 汇总指标，
生成:
  1. 终端对比表(全部指标 × checkpoint × 划分, 含最优标记)
  2. matplotlib 图组: 每个指标一张折线图(x=checkpoint, 多线=任务, 两栏=内外测试)
  3. 最优 checkpoint 摘要 + 文本检索对比
用法:
  python3 visualize_results.py [runs_root] [-o 输出png] [--ckpt 选择子集]
  runs_root 默认 /home/qkh/FunEmbed/20261108_model_eval/runs
"""
import argparse
import csv
import json
from pathlib import Path

METRICS = ['ndcg@5', 'recall@5', 'hit@1', 'ndcg@3', 'recall@3', 'hit@3']
CASE_TASKS = ['C_C', 'O_O', 'B_B', 'O_C', 'C_O']
TEXT_TASKS = ['C_text_definition', 'O_text_definition', 'B_text_definition']
SPLITS = ['internal_test', 'external_test']
SPLIT_LABELS = {'internal_test': 'internal', 'external_test': 'external'}


def collect(runs_root):
    """{(ckpt, split): {task: {metric: value}}} — 只取 modal 协议。"""
    data = {}
    for ckpt_dir in sorted(runs_root.glob('**/checkpoint-*')):
        for split in SPLITS:
            run = 'dte_' + split.replace('_test', '')
            csv_path = ckpt_dir / run / 'results' / 'results_long.csv'
            if not csv_path.is_file():
                continue
            d = {}
            for r in csv.DictReader(open(csv_path)):
                if r['label_protocol'] == 'modal' and r['metric'] in METRICS:
                    d.setdefault(r['task'], {})[r['metric']] = float(r['value'])
            if d:
                data[(ckpt_dir.name, split)] = d
    return data


def best_ckpt(data, split, metric, tasks):
    """返回该划分该指标(对给定任务集合取均值)最高的 checkpoint。"""
    scores = {}
    for (ck, sp), tasks_d in data.items():
        if sp != split:
            continue
        vals = [tasks_d[t][metric] for t in tasks if t in tasks_d]
        if vals:
            scores[ck] = sum(vals) / len(vals)
    return max(scores, key=scores.get) if scores else None, scores


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs_root', nargs='?', default='/home/qkh/FunEmbed/20261108_model_eval/runs')
    ap.add_argument('-o', '--output', default=None, help='输出 PNG 路径(默认 runs_root/eval_dashboard.png)')
    ap.add_argument('--ckpt', nargs='*', default=None, help='只对比指定 checkpoint(如 checkpoint-820)')
    ap.add_argument('--no-plot', action='store_true', help='只打印表格, 不画图')
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    data = collect(runs_root)
    if not data:
        print(f'未在 {runs_root} 下找到评测结果(results_long.csv)')
        return
    if args.ckpt:
        data = {k: v for k, v in data.items() if k[0] in args.ckpt}
    ckpts = sorted({k[0] for k in data}, key=lambda c: int(c.split('-')[1]))
    out_png = args.output or str(runs_root / 'eval_dashboard.png')

    # ---------- 终端对比表 ----------
    for split in SPLITS:
        sl = SPLIT_LABELS[split]
        print(f'\n{"=" * 90}\n【{sl}】modal 协议全部指标 (checkpoint × 任务)\n{"=" * 90}')
        for metric in METRICS:
            best, _ = best_ckpt(data, split, metric, CASE_TASKS)
            tag = f'(最优 checkpoint: {best})' if best else ''
            print(f'\n-- {metric} -- {tag}')
            hdr = f'{"checkpoint":<16}' + ''.join(f'{t:>10}' for t in CASE_TASKS)
            print(hdr)
            for ck in ckpts:
                d = data.get((ck, split))
                if not d:
                    continue
                row = f'{ck:<16}'
                for t in CASE_TASKS:
                    v = d.get(t, {}).get(metric)
                    mark = ' *' if ck == best else '  '
                    row += (f'{v:>8.4f}{mark}' if v is not None else f'{"—":>10}')
                print(row)

    # ---------- 最优摘要 ----------
    print(f'\n{"=" * 90}\n【最优 checkpoint 摘要】(按任务均值)\n{"=" * 90}')
    summary = {}
    for split in SPLITS:
        sl = SPLIT_LABELS[split]
        for metric in ['ndcg@5', 'hit@1']:
            best, scores = best_ckpt(data, split, metric, CASE_TASKS)
            if best:
                print(f'{sl:<10} {metric:<9} → {best} (均值 {scores[best]:.4f})')
                summary[(split, metric)] = (best, scores)

    # ---------- 文本检索对比 ----------
    print(f'\n{"=" * 90}\n【文本检索 (image→22 类疾病定义) ndcg@5】\n{"=" * 90}')
    for split in SPLITS:
        sl = SPLIT_LABELS[split]
        print(f'\n-- {sl} --')
        print(f'{"checkpoint":<16}' + ''.join(f'{t.replace("_text_definition","→text"):>10}' for t in TEXT_TASKS))
        for ck in ckpts:
            d = data.get((ck, split))
            if not d:
                continue
            row = f'{ck:<16}'
            for t in TEXT_TASKS:
                v = d.get(t, {}).get('ndcg@5')
                row += (f'{v:>10.4f}' if v is not None else f'{"—":>10}')
            print(row)

    if args.no_plot:
        return

    # ---------- 图组 ----------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plot_metrics = ['ndcg@5', 'recall@5', 'hit@1']
    fig, axes = plt.subplots(len(plot_metrics), 2, figsize=(14, 4 * len(plot_metrics)))
    fig.suptitle('Describe-then-Embed eval: checkpoint comparison', fontsize=14, y=0.995)
    colors = {'C_C': '#1f77b4', 'O_O': '#ff7f0e', 'B_B': '#2ca02c', 'O_C': '#d62728', 'C_O': '#9467bd'}
    xs = list(range(len(ckpts)))
    xlabels = [c.replace('checkpoint-', 'ck') for c in ckpts]

    for mi, metric in enumerate(plot_metrics):
        for si, split in enumerate(SPLITS):
            ax = axes[mi, si]
            for task in CASE_TASKS:
                ys = [data.get((ck, split), {}).get(task, {}).get(metric) for ck in ckpts]
                ax.plot(xs, ys, marker='o', label=task, color=colors[task], linewidth=2)
            ax.set_title(f'{SPLIT_LABELS[split]} — {metric}')
            ax.set_xticks(xs)
            ax.set_xticklabels(xlabels, rotation=45)
            ax.grid(alpha=0.3)
            if si == 0:
                ax.set_ylabel(metric)
            if mi == 0:
                ax.legend(fontsize=8, loc='best')

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=150)
    print(f'\n图组已保存: {out_png}')

    # 附: 文本检索一张图
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    for task in TEXT_TASKS:
        for split, ls in zip(SPLITS, ['-', '--']):
            ys = [data.get((ck, split), {}).get(task, {}).get('ndcg@5') for ck in ckpts]
            ax2.plot(xs, ys, marker='o', linestyle=ls,
                     label=f'{task.replace("_text_definition", "")} {SPLIT_LABELS[split]}')
    ax2.set_title('Text retrieval ndcg@5 (solid=internal, dashed=external)')
    ax2.set_xticks(xs)
    ax2.set_xticklabels(xlabels, rotation=45)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9, ncol=2)
    fig2.tight_layout()
    out_png2 = str(Path(out_png).with_name(Path(out_png).stem + '_text.png'))
    fig2.savefig(out_png2, dpi=150)
    print(f'文本检索图已保存: {out_png2}')


if __name__ == '__main__':
    main()
