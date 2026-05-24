"""
train_stats.py - 训练过程统计采集器。

独立于训练循环逻辑，在关键点采集指标，
低频写入 CSV 防崩溃丢失，训练结束后生成图表和汇总报告。
"""

import csv
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TrainStats:
    """训练过程统计采集器，独立于训练循环逻辑。

    用法:
        stats = TrainStats(snapshot_path)
        # 训练循环中:
        stats.log_train_step(iter_num, total_loss, ce_loss, dice_loss, lr)
        stats.log_val_step(iter_num, mean_dice, mean_iou)
        stats.log_epoch_time(epoch_num, seconds)
        # 训练结束:
        stats.plot_all(dataset_name, total_iters, total_epochs, total_time_s)
    """

    FLUSH_INTERVAL = 1000

    def __init__(self, snapshot_path: str):
        self.snapshot_path = snapshot_path
        self.plots_dir = os.path.join(snapshot_path, "training_plots")
        os.makedirs(self.plots_dir, exist_ok=True)

        self._csv_path = os.path.join(snapshot_path, "training_stats.csv")
        self._csv_header_written = os.path.exists(self._csv_path)

        # 内存累积
        self._train_rows = []   # (iter, total_loss, ce_loss, dice_loss, lr)
        self._val_rows = []     # (iter, mean_dice, mean_iou)
        self._epoch_rows = []   # (epoch, seconds)

    def log_train_step(self, iter_num, total_loss, ce_loss, dice_loss, lr):
        self._train_rows.append((iter_num, total_loss, ce_loss, dice_loss, lr))
        if iter_num % self.FLUSH_INTERVAL == 0:
            self._flush_csv()

    def log_val_step(self, iter_num, mean_dice, mean_iou):
        self._val_rows.append((iter_num, mean_dice, mean_iou))

    def log_epoch_time(self, epoch_num, seconds):
        self._epoch_rows.append((epoch_num, seconds))

    # ---- CSV 持久化 ----

    def _flush_csv(self):
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not self._csv_header_written:
                writer.writerow(["type", "iter", "value1", "value2", "value3", "value4"])
                self._csv_header_written = True
            for row in self._train_rows:
                writer.writerow(["train"] + list(row))
            for row in self._val_rows:
                writer.writerow(["val", row[0], row[1], row[2], "", ""])
            for row in self._epoch_rows:
                writer.writerow(["epoch", row[0], row[1], "", "", ""])
        # flush 后清空，避免重复写入
        self._train_rows.clear()
        self._val_rows.clear()
        self._epoch_rows.clear()

    # ---- 图表生成 ----

    def _plot_loss(self):
        iters = [r[0] for r in self._all_train]
        total = [r[1] for r in self._all_train]
        ce = [r[2] for r in self._all_train]
        dice = [r[3] for r in self._all_train]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(iters, total, label="Total Loss", linewidth=1.2)
        ax.plot(iters, ce, label="CE Loss", linewidth=0.8, alpha=0.7)
        ax.plot(iters, dice, label="Dice Loss", linewidth=0.8, alpha=0.7)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.plots_dir, "loss_curve.png"), dpi=150)
        plt.close(fig)

    def _plot_lr(self):
        iters = [r[0] for r in self._all_train]
        lrs = [r[4] for r in self._all_train]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(iters, lrs, linewidth=1.2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.plots_dir, "lr_curve.png"), dpi=150)
        plt.close(fig)

    def _plot_val(self):
        iters = [r[0] for r in self._all_val]
        dice = [r[1] for r in self._all_val]
        iou = [r[2] for r in self._all_val]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(iters, dice, label="Dice", linewidth=1.2)
        ax.plot(iters, iou, label="IoU", linewidth=1.2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Score")
        ax.set_title("Validation Metrics")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.plots_dir, "val_metrics.png"), dpi=150)
        plt.close(fig)

    def _plot_summary(self, dataset_name, total_iters, total_epochs, total_time_s):
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle(f"Training Summary - {dataset_name}", fontsize=14)

        # 1) Loss
        ax = axes[0, 0]
        iters = [r[0] for r in self._all_train]
        ax.plot(iters, [r[1] for r in self._all_train], label="Total", linewidth=1)
        ax.plot(iters, [r[2] for r in self._all_train], label="CE", linewidth=0.7, alpha=0.7)
        ax.plot(iters, [r[3] for r in self._all_train], label="Dice", linewidth=0.7, alpha=0.7)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 2) LR
        ax = axes[0, 1]
        ax.plot(iters, [r[4] for r in self._all_train], linewidth=1)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("LR")
        ax.set_title("Learning Rate")
        ax.grid(True, alpha=0.3)

        # 3) Val metrics
        ax = axes[1, 0]
        val_iters = [r[0] for r in self._all_val]
        ax.plot(val_iters, [r[1] for r in self._all_val], label="Dice", linewidth=1)
        ax.plot(val_iters, [r[2] for r in self._all_val], label="IoU", linewidth=1)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Score")
        ax.set_title("Validation")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 4) Epoch time
        ax = axes[1, 1]
        if self._all_epoch:
            epochs = [r[0] for r in self._all_epoch]
            times = [r[1] for r in self._all_epoch]
            ax.bar(epochs, times, color="steelblue", alpha=0.7)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Time (s)")
            ax.set_title("Epoch Duration")
            ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        fig.savefig(os.path.join(self.plots_dir, "training_summary.png"), dpi=150)
        plt.close(fig)

    # ---- 报告生成 ----

    def _write_report(self, dataset_name, total_iters, total_epochs,
                      total_time_s, best_dice, best_dice_iter,
                      best_iou, best_iou_iter, final_lr):
        report_path = os.path.join(self.snapshot_path, "training_report.txt")
        hrs = int(total_time_s // 3600)
        mins = int((total_time_s % 3600) // 60)
        secs = int(total_time_s % 60)
        avg_epoch = (total_time_s / total_epochs) if total_epochs > 0 else 0

        with open(report_path, "w") as f:
            f.write("=== Training Report ===\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Total iterations: {total_iters}\n")
            f.write(f"Total epochs: {total_epochs}\n")
            f.write(f"Total time: {hrs}h {mins}m {secs}s\n")
            if best_dice is not None:
                f.write(f"Best val Dice: {best_dice:.4f} @ iter {best_dice_iter}\n")
            if best_iou is not None:
                f.write(f"Best val IoU:  {best_iou:.4f} @ iter {best_iou_iter}\n")
            f.write(f"Final lr: {final_lr:.8f}\n")
            f.write(f"Avg epoch time: {avg_epoch:.1f}s\n")

    # ---- 主入口 ----

    def plot_all(self, dataset_name, total_iters, total_epochs, total_time_s):
        # 先 flush 剩余数据到 CSV
        if self._train_rows or self._val_rows or self._epoch_rows:
            self._flush_csv()

        # 从 CSV 加载全量数据（确保崩溃恢复后也能用）
        self._all_train = []
        self._all_val = []
        self._all_epoch = []

        if os.path.exists(self._csv_path):
            with open(self._csv_path, "r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)  # skip header
                for row in reader:
                    if not row:
                        continue
                    if row[0] == "train":
                        self._all_train.append(
                            (int(row[1]), float(row[2]), float(row[3]),
                             float(row[4]), float(row[5])))
                    elif row[0] == "val":
                        self._all_val.append(
                            (int(row[1]), float(row[2]), float(row[3])))
                    elif row[0] == "epoch":
                        self._all_epoch.append(
                            (int(row[1]), float(row[2])))

        if not self._all_train:
            return

        # 生成图表
        self._plot_loss()
        self._plot_lr()
        if self._all_val:
            self._plot_val()
        self._plot_summary(dataset_name, total_iters, total_epochs, total_time_s)

        # 提取 best 指标
        best_dice, best_dice_iter = None, None
        best_iou, best_iou_iter = None, None
        if self._all_val:
            best_entry = max(self._all_val, key=lambda r: r[1])
            best_dice, best_dice_iter = best_entry[1], best_entry[0]
            best_entry_iou = max(self._all_val, key=lambda r: r[2])
            best_iou, best_iou_iter = best_entry_iou[2], best_entry_iou[0]

        final_lr = self._all_train[-1][4] if self._all_train else 0

        self._write_report(
            dataset_name, total_iters, total_epochs, total_time_s,
            best_dice, best_dice_iter, best_iou, best_iou_iter, final_lr)
