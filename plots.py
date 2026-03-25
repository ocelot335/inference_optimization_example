import json
import matplotlib.pyplot as plt
import os


def draw_plots():
    if not os.path.exists("benchmark_results.json"):
        return

    with open("benchmark_results.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    PLOTS_DIR = "plots"
    os.makedirs(PLOTS_DIR, exist_ok=True)

    metrics_to_plot = {
        "throughput": (
            "Пропускная способность",
            "RPS (Запросов/сек)",
            "plot_throughput.png",
        ),
        "avg_latency": (
            "Средняя задержка",
            "Миллисекунды (ms)",
            "plot_avg_latency.png",
        ),
        "p95_latency": (
            "Задержка p95",
            "Миллисекунды (ms)",
            "plot_p95_latency.png",
        ),
        "p99_latency": (
            "Задержка p99",
            "Миллисекунды (ms)",
            "plot_p99_latency.png",
        ),
        "cpu_usage": ("Утилизация CPU", "Проценты (%)", "plot_cpu_usage.png"),
        "error_rate": (
            "Процент ошибок",
            "Проценты (%)",
            "plot_error_rate.png",
        ),
    }

    for metric_key, (title, ylabel, filename) in metrics_to_plot.items():
        plt.figure(figsize=(10, 5))

        for stage_name, metrics in data.items():
            concurrency_levels = list(metrics.keys())
            y_values = [m.get(metric_key, 0) for m in metrics.values()]

            plt.plot(
                concurrency_levels,
                y_values,
                marker="o",
                linewidth=2,
                label=stage_name,
            )

        plt.title(title)
        plt.xlabel("Одновременные запросы (Concurrency)")
        plt.ylabel(ylabel)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()

        filepath = os.path.join(PLOTS_DIR, filename)
        plt.savefig(filepath)
        plt.close()

    print(f"Графики сохранены в папку '{PLOTS_DIR}/'")


if __name__ == "__main__":
    draw_plots()
