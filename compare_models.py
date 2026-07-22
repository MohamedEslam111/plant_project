"""
Model comparison module for the Plant Health Checker project.

After each model has been trained and evaluated individually (see main.py),
this module builds a combined comparison: a results table (CSV + printed),
a bar chart comparing key metrics, an overlaid training-curve plot, and a
short written report -- everything needed for a graduation-project
results section.
"""

import csv

import matplotlib.pyplot as plt

import config


def build_comparison_table(results: dict):
    """
    Build a comparison table from per-model results.

    Args:
        results: dict mapping model_name -> {
            'metrics': dict from evaluate.compute_overall_metrics,
            'checkpoint': dict from train.train_model,
            'history': dict from train.train_model,
            'num_params': int,
            'avg_inference_time_ms': float,
        }

    Returns:
        rows: list of dicts, one per model, ready for CSV/printing.
    """
    rows = []
    for model_name, result in results.items():
        metrics = result['metrics']
        checkpoint = result['checkpoint']
        history = result['history']

        rows.append({
            'Model': model_name,
            'Test Accuracy (%)': round(metrics['accuracy'] * 100, 2),
            'Macro F1 (%)': round(metrics['f1_macro'] * 100, 2),
            'Weighted F1 (%)': round(metrics['f1_weighted'] * 100, 2),
            'Macro Precision (%)': round(metrics['precision_macro'] * 100, 2),
            'Macro Recall (%)': round(metrics['recall_macro'] * 100, 2),
            'Best Val Accuracy (%)': round(checkpoint['val_acc'], 2),
            'Best Val Loss': round(checkpoint['val_loss'], 4),
            'Best Epoch': checkpoint['epoch'] + 1,
            'Total Train Time (min)': round(history['total_train_time'] / 60, 2),
            'Avg Inference Time (ms/img)': round(result['avg_inference_time_ms'], 2),
            'Trainable Parameters (M)': round(result['num_params'] / 1e6, 2),
        })
    return rows


def save_comparison_csv(rows):
    """Save the comparison table to CSV."""
    if not rows:
        return
    with open(config.COMPARISON_TABLE_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Comparison table saved to: {config.COMPARISON_TABLE_PATH}")


def print_comparison_table(rows):
    """Print a nicely aligned comparison table to the console."""
    if not rows:
        print("No results to compare.")
        return

    headers = list(rows[0].keys())
    col_widths = [max(len(str(h)), max(len(str(r[h])) for r in rows)) + 2 for h in headers]

    print("\n" + "=" * sum(col_widths))
    print("MODEL COMPARISON TABLE")
    print("=" * sum(col_widths))

    header_line = "".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * sum(col_widths))

    for row in rows:
        line = "".join(f"{str(row[h]):<{w}}" for h, w in zip(headers, col_widths))
        print(line)
    print("=" * sum(col_widths))


def plot_comparison_chart(rows):
    """Plot a grouped bar chart comparing key metrics across models."""
    if not rows:
        return

    model_names = [r['Model'] for r in rows]
    metrics_to_plot = ['Test Accuracy (%)', 'Macro F1 (%)', 'Macro Precision (%)', 'Macro Recall (%)']

    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(model_names))
    width = 0.2
    colors = ['#2563eb', '#16a34a', '#f59e0b', '#dc2626']

    for i, metric in enumerate(metrics_to_plot):
        values = [r[metric] for r in rows]
        offsets = [xi + i * width for xi in x]
        ax.bar(offsets, values, width=width, label=metric, color=colors[i])

    ax.set_xticks([xi + width * 1.5 for xi in x])
    ax.set_xticklabels(model_names)
    ax.set_ylabel('Score (%)')
    ax.set_title('Model Comparison - Test Set Metrics', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(config.COMPARISON_CHART_PATH, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Comparison chart saved to: {config.COMPARISON_CHART_PATH}")


def plot_comparison_curves(results: dict):
    """Overlay validation loss and accuracy curves for all models on the same plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#2563eb', '#dc2626', '#16a34a', '#f59e0b']

    for i, (model_name, result) in enumerate(results.items()):
        history = result['history']
        epochs_range = range(1, len(history['val_losses']) + 1)
        color = colors[i % len(colors)]

        axes[0].plot(epochs_range, history['val_losses'], 'o-', label=f'{model_name}', color=color)
        axes[1].plot(epochs_range, history['val_accuracies'], 'o-', label=f'{model_name}', color=color)

    axes[0].set_title('Validation Loss Comparison', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title('Validation Accuracy Comparison', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(config.COMPARISON_CURVES_PATH, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Comparison curves saved to: {config.COMPARISON_CURVES_PATH}")


def save_comparison_report(rows):
    """Save a short text summary declaring the best model per key metric."""
    if not rows:
        return

    best_accuracy = max(rows, key=lambda r: r['Test Accuracy (%)'])
    best_f1 = max(rows, key=lambda r: r['Macro F1 (%)'])
    fastest_inference = min(rows, key=lambda r: r['Avg Inference Time (ms/img)'])
    smallest_model = min(rows, key=lambda r: r['Trainable Parameters (M)'])
    fastest_training = min(rows, key=lambda r: r['Total Train Time (min)'])

    lines = []
    lines.append("Plant Health Checker - Model Comparison Report")
    lines.append("=" * 60)
    lines.append("")
    for row in rows:
        lines.append(f"Model: {row['Model']}")
        for key, value in row.items():
            if key != 'Model':
                lines.append(f"  {key:<30s}: {value}")
        lines.append("")

    lines.append("-" * 60)
    lines.append("Key Takeaways")
    lines.append("-" * 60)
    lines.append(f"Highest Test Accuracy   : {best_accuracy['Model']} ({best_accuracy['Test Accuracy (%)']}%)")
    lines.append(f"Highest Macro F1-Score  : {best_f1['Model']} ({best_f1['Macro F1 (%)']}%)")
    lines.append(f"Fastest Inference       : {fastest_inference['Model']} "
                 f"({fastest_inference['Avg Inference Time (ms/img)']} ms/image)")
    lines.append(f"Smallest Model          : {smallest_model['Model']} "
                 f"({smallest_model['Trainable Parameters (M)']}M parameters)")
    lines.append(f"Fastest Training        : {fastest_training['Model']} "
                 f"({fastest_training['Total Train Time (min)']} minutes)")

    report_text = "\n".join(lines)

    with open(config.COMPARISON_REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"\nComparison report saved to: {config.COMPARISON_REPORT_PATH}")


def run_full_comparison(results: dict):
    """
    Convenience function: build table, save CSV, print table, plot chart,
    plot overlaid curves, and save the written report -- all in one call.
    """
    rows = build_comparison_table(results)
    save_comparison_csv(rows)
    print_comparison_table(rows)
    plot_comparison_chart(rows)
    plot_comparison_curves(results)
    save_comparison_report(rows)
    return rows
