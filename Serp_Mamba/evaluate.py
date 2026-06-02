"""
evaluate.py - 独立测评入口脚本。

可脱离推理单独运行，加载已保存的预测结果 + 标签 → 调用对应 evaluator → 输出指标。

用法：
    python evaluate.py --dataset fives --pred_dir experiments/FIVES/test_results/predictions
    python evaluate.py --dataset fives --pred_dir ... --prob_dir ...  # 指定概率图目录
"""

import argparse
import os
import numpy as np
from PIL import Image
from dataloaders.dataset_registry import load_dataset_config, ConfigDataSets
from evaluators import get_evaluator
from utils.metrics import PROB_BASED_METRICS

parser = argparse.ArgumentParser(description="独立测评脚本")
parser.add_argument("--dataset", type=str, required=True,
                    help="Dataset config name or path to YAML file")
parser.add_argument("--pred_dir", type=str, required=True,
                    help="Directory containing prediction PNG files")
parser.add_argument("--prob_dir", type=str, default=None,
                    help="Directory containing probability .npy files (for AUC etc.)")
parser.add_argument("--split", type=str, default="test",
                    help="Dataset split to load labels from")
parser.add_argument("--output", type=str, default=None,
                    help="Output file path (default: pred_dir/../output.txt)")


def load_prediction(pred_path):
    """加载预测掩码 PNG → 二值数组 (0/1)"""
    pred = np.array(Image.open(pred_path))
    return (pred > 127).astype(np.uint8)


def load_probability(prob_path):
    """加载概率图 .npy"""
    return np.load(prob_path)


def run_evaluation(config, pred_dir, prob_dir, save_dir, checkpoint_names=None, split="test", output_path=None):
    """
    核心测评流程，供 evaluate.py 独立运行和 test.py --evaluate 共用。

    Args:
        config: 数据集配置 dict
        pred_dir: 预测掩码目录
        prob_dir: 概率图目录（可为 None，则跳过概率相关指标）
        save_dir: 输出目录
        checkpoint_names: checkpoint 文件名列表（用于写入 output.txt，仅 test.py 调用时提供）
        split: 数据集划分名
        output_path: 输出文件路径（默认 save_dir/output.txt）
    """
    evaluator = get_evaluator(config)
    db_test = ConfigDataSets(config=config, split=split)

    # 检查是否需要概率图
    needs_prob = bool(set(evaluator.get_metric_names()) & PROB_BASED_METRICS)

    per_image_results = []

    for i in range(len(db_test)):
        sample = db_test[i]
        case_name = sample["name"]
        base_name = os.path.splitext(case_name)[0]

        # 加载预测
        pred_path = os.path.join(pred_dir, base_name + ".png")
        if not os.path.exists(pred_path):
            print(f"Warning: prediction not found for {case_name}, skipping")
            continue

        prediction = load_prediction(pred_path)

        # 加载标签
        label = sample["label"]
        if hasattr(label, 'astype'):
            label = label.astype(np.uint8)
        else:
            label = np.array(label, dtype=np.uint8)
        label = (label > 0).astype(np.uint8)

        # 加载概率图（如果需要）
        pred_prob = None
        if needs_prob and prob_dir:
            prob_path = os.path.join(prob_dir, base_name + "_prob.npy")
            if os.path.exists(prob_path):
                pred_prob = load_probability(prob_path)
            else:
                print(f"Warning: probability map not found for {case_name}")

        # 评估
        metrics = evaluator.evaluate_single(prediction, label, pred_prob=pred_prob)
        per_image_results.append({"name": case_name, **metrics})

    if not per_image_results:
        print("No valid predictions found, skipping evaluation")
        return

    # 聚合
    perf = evaluator.aggregate(per_image_results)

    # 写入 output.txt
    if output_path is None:
        output_path = os.path.join(save_dir, "output.txt")
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    model_info = checkpoint_names[0] if checkpoint_names else "manual_evaluation"

    with open(output_path, 'w') as f:
        f.write(evaluator.format_summary(perf, model_info, split, len(per_image_results)))

        f.write("\nper-image results:\n")
        for item in per_image_results:
            case_name = item.pop("name")
            f.write(evaluator.format_per_image(case_name, item))
        f.write("\n")

    # 控制台输出
    print(f"Evaluation results ({len(per_image_results)} images):")
    for name, (mean_val, std_val) in perf.items():
        print(f"  {name}: {mean_val:.6f} +/- {std_val:.6f}")
    print(f"Results saved to {output_path}")


if __name__ == '__main__':
    args = parser.parse_args()

    # 加载配置
    cfg_path = args.dataset
    if not cfg_path.endswith('.yaml'):
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', cfg_path + '.yaml')
    config = load_dataset_config(cfg_path)

    # 目录推断
    pred_dir = args.pred_dir
    prob_dir = args.prob_dir
    if prob_dir is None:
        # 默认尝试同级 probabilities 目录
        candidate = os.path.join(os.path.dirname(pred_dir), "probabilities")
        if os.path.isdir(candidate):
            prob_dir = candidate

    save_dir = os.path.dirname(pred_dir)
    output_path = args.output or os.path.join(save_dir, "output.txt")

    run_evaluation(
        config=config,
        pred_dir=pred_dir,
        prob_dir=prob_dir,
        save_dir=save_dir,
        split=args.split,
        output_path=output_path,
    )
