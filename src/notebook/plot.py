import numpy as np
import matplotlib.pyplot as plt


def plot_bm25_vs_tfidf_metric(
    *,
    datasets,
    summary_df,
    metric_col: str,
    y_label: str,
    title: str,
    plot_path,
    dpi: int = 150,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(datasets))
    width = 0.35

    bm25_scores = [
        float(
            summary_df[
                (summary_df["dataset"] == d) & (summary_df["retrieval"] == "bm25")
            ][metric_col].iloc[0]
        )
        for d in datasets
    ]
    tfidf_scores = [
        float(
            summary_df[
                (summary_df["dataset"] == d) & (summary_df["retrieval"] == "tfidf")
            ][metric_col].iloc[0]
        )
        for d in datasets
    ]

    bars1 = ax.bar(x - width / 2, bm25_scores, width, label="BM25", alpha=0.8)
    bars2 = ax.bar(x + width / 2, tfidf_scores, width, label="TF-IDF", alpha=0.8)

    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(plot_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_method_metrics(
    *,
    datasets,
    retrieval_methods,
    baseline_summary_df,
    method_summary_df,
    method_label: str,
    output_dir,
    dpi: int = 150,
):
    metric_specs = [
        ("ndcg@10", "nDCG@10", "ndcg.png"),
        ("mrr", "MRR", "mrr.png"),
        ("recall@100", "Recall@100", "recall.png"),
        ("map", "MAP", "map.png"),
    ]

    figs = []
    for metric_col, y_label, filename in metric_specs:
        plot_path = output_dir / filename
        fig = plot_baseline_vs_method_metric(
            datasets=datasets,
            retrieval_methods=retrieval_methods,
            baseline_summary_df=baseline_summary_df,
            method_summary_df=method_summary_df,
            metric_col=metric_col,
            y_label=y_label,
            method_label=method_label,
            plot_path=plot_path,
            dpi=dpi,
        )
        figs.append(fig)

    return figs


def plot_eps_by_method(
    *,
    agg_df,
    plot_path,
    method_order=None,
    dpi: int = 150,
):
    import pandas as pd

    if method_order is None:
        method_order = ["baseline", "append", "reformulate", "agr"]

    agg_df_ordered = agg_df.copy()
    agg_df_ordered["method"] = pd.Categorical(
        agg_df_ordered["method"], categories=method_order, ordered=True
    )
    agg_df_ordered = agg_df_ordered.sort_values("method")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(agg_df_ordered))
    scores = agg_df_ordered["EPS_mean"].astype(float).tolist()
    bars = ax.bar(x, scores, alpha=0.8)

    ax.set_xlabel("Method", fontsize=12)
    ax.set_ylabel("EPS (weighted normalized)", fontsize=12)
    ax.set_title(
        "Expansion Performance Score (EPS) by Method", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(agg_df_ordered["method"].astype(str).tolist())
    ax.grid(axis="y", alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(plot_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_baseline_metrics(
    *,
    datasets,
    summary_df,
    output_dir,
    dpi: int = 150,
):
    metric_specs = [
        ("ndcg@10", "nDCG@10", "Baseline nDCG@10 (BM25 vs TF-IDF)", "ndcg.png"),
        ("mrr", "MRR", "Baseline MRR (BM25 vs TF-IDF)", "mrr.png"),
        (
            "recall@100",
            "Recall@100",
            "Baseline Recall@100 (BM25 vs TF-IDF)",
            "recall.png",
        ),
        ("map", "MAP", "Baseline MAP (BM25 vs TF-IDF)", "map.png"),
    ]

    figs = []
    for metric_col, y_label, title, filename in metric_specs:
        plot_path = output_dir / filename
        fig = plot_bm25_vs_tfidf_metric(
            datasets=datasets,
            summary_df=summary_df,
            metric_col=metric_col,
            y_label=y_label,
            title=title,
            plot_path=plot_path,
            dpi=dpi,
        )
        figs.append(fig)
    return figs


def plot_baseline_vs_method_metric(
    *,
    datasets,
    retrieval_methods,
    baseline_summary_df,
    method_summary_df,
    metric_col: str,
    y_label: str,
    method_label: str,
    plot_path,
    dpi: int = 150,
):
    fig, axes = plt.subplots(1, len(retrieval_methods), figsize=(14, 6), squeeze=False)
    fig.suptitle(
        f"{method_label} Method vs Baseline: {y_label}", fontsize=16, fontweight="bold"
    )

    for idx, retrieval in enumerate(retrieval_methods):
        ax = axes[0][idx]
        x = np.arange(len(datasets))
        width = 0.35

        baseline_scores = [
            float(
                baseline_summary_df[
                    (baseline_summary_df["dataset"] == d)
                    & (baseline_summary_df["retrieval"] == retrieval)
                ][metric_col].iloc[0]
            )
            for d in datasets
        ]
        method_scores = [
            float(
                method_summary_df[
                    (method_summary_df["dataset"] == d)
                    & (method_summary_df["retrieval"] == retrieval)
                ][metric_col].iloc[0]
            )
            for d in datasets
        ]

        bars1 = ax.bar(x - width / 2, baseline_scores, width, label="Baseline", alpha=0.8)
        bars2 = ax.bar(x + width / 2, method_scores, width, label=method_label, alpha=0.8)

        ax.set_xlabel("Dataset", fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(f"{retrieval.upper()}", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    plt.tight_layout()
    plt.savefig(plot_path, dpi=dpi, bbox_inches="tight")
    return fig
