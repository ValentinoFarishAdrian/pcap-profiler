#!/usr/bin/env python3

import argparse
import copy
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
    silhouette_score,
    pairwise_distances,
)
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.optimize import linear_sum_assignment


# ═══════════════════════════════════════════════════════════════════
# 1. FEATURE EXTRACTION  (identik dengan notebook Zaki)
# ═══════════════════════════════════════════════════════════════════
def extract_features(dataset: list):
    df = pd.DataFrame(dataset)

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    oui_vec = ohe.fit_transform(df[['oui']])
    n_oui_features = oui_vec.shape[1]

    mlb = MultiLabelBinarizer()
    svc_vec = mlb.fit_transform(df['svc'])

    X = np.hstack([oui_vec, svc_vec])
    return X, n_oui_features, ohe, mlb


# ═══════════════════════════════════════════════════════════════════
# 2. HYBRID DISTANCE METRIC (DYNAMIT) — identik dengan notebook Zaki
# ═══════════════════════════════════════════════════════════════════
def create_dynamit_metric(n_oui_features: int):
    def dynamit_metric(x, y):
        x_oui, y_oui = x[:n_oui_features], y[:n_oui_features]
        dist_oui = 0.0 if np.array_equal(x_oui, y_oui) else 1.0

        x_svc, y_svc = x[n_oui_features:], y[n_oui_features:]
        intersection = np.sum(np.minimum(x_svc, y_svc))
        union        = np.sum(np.maximum(x_svc, y_svc))
        dist_svc     = 0.0 if union == 0 else 1.0 - (intersection / union)

        return (dist_oui + dist_svc) / 2.0

    return dynamit_metric


# ═══════════════════════════════════════════════════════════════════
# 3. AHC RUNNERS
# ═══════════════════════════════════════════════════════════════════
def run_ahc_threshold(dist_matrix: np.ndarray, threshold: float):
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric='precomputed',
        linkage='average',
    )
    labels = model.fit_predict(dist_matrix)
    return model, labels


def run_ahc_k(dist_matrix: np.ndarray, k: int):
    model = AgglomerativeClustering(
        n_clusters=k,
        metric='precomputed',
        linkage='average',
    )
    labels = model.fit_predict(dist_matrix)
    return model, labels


# ═══════════════════════════════════════════════════════════════════
# 4A. THRESHOLD SWEEP
# ═══════════════════════════════════════════════════════════════════
def threshold_sweep(dist_matrix: np.ndarray, true_labels,
                    t_min: float, t_max: float, t_step: float):
    t_range = np.arange(t_min, t_max + t_step * 0.5, t_step)
    results = []

    for t in t_range:
        _, labels = run_ahc_threshold(dist_matrix, t)

        n_clusters = len(set(labels))   # AHC tidak punya noise
        ari = sil = np.nan

        if true_labels is not None and n_clusters > 1:
            ari = adjusted_rand_score(np.array(true_labels), labels)

        if n_clusters >= 2 and len(labels) >= 2:
            try:
                sil = silhouette_score(dist_matrix, labels,
                                       metric='precomputed')
            except Exception:
                sil = np.nan

        results.append({
            'threshold':  round(float(t), 4),
            'n_clusters': n_clusters,
            'ari':        ari,
            'silhouette': sil,
        })

    # Pilih yang terbaik
    if true_labels is not None:
        valid = [(r, r['ari']) for r in results if not np.isnan(r['ari'])]
        best_r = max(valid, key=lambda x: x[1])[0] if valid \
                 else results[len(results) // 2]
    else:
        valid = [(r, r['silhouette']) for r in results
                 if not np.isnan(r['silhouette'])]
        best_r = max(valid, key=lambda x: x[1])[0] if valid \
                 else results[len(results) // 2]

    _, best_labels = run_ahc_threshold(dist_matrix, best_r['threshold'])
    return results, best_r['threshold'], best_labels


# ═══════════════════════════════════════════════════════════════════
# 4B. K SWEEP
# ═══════════════════════════════════════════════════════════════════
def k_sweep(dist_matrix: np.ndarray, true_labels,
            k_min: int, k_max: int):
    n = len(dist_matrix)
    k_max = min(k_max, n - 1)
    results = []

    for k in range(k_min, k_max + 1):
        _, labels = run_ahc_k(dist_matrix, k)

        ari = sil = np.nan

        if true_labels is not None:
            ari = adjusted_rand_score(np.array(true_labels), labels)

        if k >= 2 and len(labels) >= 2:
            try:
                sil = silhouette_score(dist_matrix, labels,
                                       metric='precomputed')
            except Exception:
                sil = np.nan

        results.append({
            'k':          k,
            'ari':        ari,
            'silhouette': sil,
        })

    if true_labels is not None:
        valid = [(r, r['ari']) for r in results if not np.isnan(r['ari'])]
        best_r = max(valid, key=lambda x: x[1])[0] if valid \
                 else results[0]
    else:
        valid = [(r, r['silhouette']) for r in results
                 if not np.isnan(r['silhouette'])]
        best_r = max(valid, key=lambda x: x[1])[0] if valid \
                 else results[0]

    _, best_labels = run_ahc_k(dist_matrix, best_r['k'])
    return results, best_r['k'], best_labels


# ═══════════════════════════════════════════════════════════════════
# 5. METRICS
# ═══════════════════════════════════════════════════════════════════
def compute_metrics(labels: np.ndarray, dist_matrix: np.ndarray,
                    true_labels=None):
    unique    = set(labels)
    n_pred    = len(unique)   # AHC tidak menghasilkan noise (-1)
    n         = len(labels)

    result = {
        'n_clusters_pred': n_pred,
        'n_noise':         0,       # AHC tidak punya konsep noise
        'noise_pct':       0.0,
        'ari':        np.nan,
        'nmi':        np.nan,
        'fmi':        np.nan,
        'silhouette': np.nan,
    }

    if n_pred >= 2 and n >= 2:
        try:
            result['silhouette'] = silhouette_score(
                dist_matrix, labels, metric='precomputed'
            )
        except Exception:
            pass

    if true_labels is not None and len(labels) > 0:
        result['ari'] = adjusted_rand_score(np.array(true_labels), labels)
        result['nmi'] = normalized_mutual_info_score(
            np.array(true_labels), labels
        )
        result['fmi'] = fowlkes_mallows_score(
            np.array(true_labels), labels
        )

    return result


# ═══════════════════════════════════════════════════════════════════
# 6A. GRAPH 1A — CONFUSION MATRIX
# ═══════════════════════════════════════════════════════════════════
def plot_confusion_matrix(true_labels, predicted_labels, ax,
                          scenario_name: str = "",
                          class_names: list = None):
    true_labels      = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    true_unique = np.unique(true_labels)
    pred_unique = np.unique(predicted_labels)
    n_true = len(true_unique)
    n_pred = len(pred_unique)

    cm = np.zeros((n_true, n_pred), dtype=int)
    true_idx = {t: i for i, t in enumerate(true_unique)}
    pred_idx = {p: i for i, p in enumerate(pred_unique)}
    for t, p in zip(true_labels, predicted_labels):
        cm[true_idx[t], pred_idx[p]] += 1

    if cm.size > 0:
        row_ind, col_ind = linear_sum_assignment(-cm)
        aligned   = list(col_ind)
        remaining = [c for c in range(n_pred) if c not in aligned]
        aligned.extend(remaining)
        cm          = cm[:, aligned]
        pred_unique = pred_unique[aligned]

    x_labels = [f"Pred {i}" for i in pred_unique]
    y_labels = (
        [f"{class_names[t]}" for t in true_unique]
        if class_names and len(class_names) >= len(true_unique)
        else [f"True {t}" for t in true_unique]
    )

    sns.heatmap(
        cm, annot=True, fmt='d',
        cmap=sns.color_palette("Blues", as_cmap=True),
        xticklabels=x_labels, yticklabels=y_labels,
        linewidths=0.5, linecolor='#e0e0e0',
        cbar_kws={'shrink': 0.7},
        ax=ax
    )
    ax.set_xlabel("Predicted Cluster", fontsize=11, labelpad=8)
    ax.set_ylabel("True Cluster",      fontsize=11, labelpad=8)
    ax.set_title(f"Confusion Matrix — {scenario_name}",
                 fontsize=12, fontweight='bold', pad=12)
    ax.tick_params(axis='both', labelsize=9)


# ═══════════════════════════════════════════════════════════════════
# 6B. GRAPH 1B — CLUSTER MEMBERSHIP HEATMAP (tanpa GT)
# ═══════════════════════════════════════════════════════════════════
def plot_membership_heatmap(dataset: list, labels: np.ndarray, ax,
                             scenario_name: str = ""):
    df = pd.DataFrame(dataset).copy()
    df['cluster']   = labels
    df['label_str'] = df['cluster'].apply(lambda c: f"Cluster {c}")

    mlb     = MultiLabelBinarizer()
    svc_mat = mlb.fit_transform(df['svc'])
    svc_df  = pd.DataFrame(svc_mat, columns=mlb.classes_)

    order        = df.sort_values('cluster').index
    svc_sorted   = svc_df.loc[order]
    label_sorted = df.loc[order, 'label_str'].values
    ip_sorted    = (df.loc[order, 'ip'].values
                    if 'ip' in df.columns
                    else [f"H{i}" for i in range(len(df))])

    y_labels = [f"{ip} [{lbl}]" for ip, lbl in zip(ip_sorted, label_sorted)]

    sns.heatmap(
        svc_sorted, annot=False,
        cmap=sns.color_palette("YlOrRd", as_cmap=True),
        yticklabels=y_labels,
        xticklabels=mlb.classes_,
        linewidths=0.3, linecolor='#f0f0f0',
        cbar_kws={'shrink': 0.6, 'label': 'Service Present'},
        ax=ax
    )
    ax.set_title(f"Cluster Membership — {scenario_name}",
                 fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Service",        fontsize=10, labelpad=8)
    ax.set_ylabel("Host [Cluster]", fontsize=10, labelpad=8)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)


# ═══════════════════════════════════════════════════════════════════
# 7. GRAPH 2 — SCATTER PLOT (MDS 2D)
# ═══════════════════════════════════════════════════════════════════
def plot_scatter_mds(dist_matrix: np.ndarray, labels: np.ndarray,
                     ax, dataset: list = None,
                     true_labels=None, scenario_name: str = "",
                     file_labels: list = None):
    n = len(labels)

    if n < 3:
        ax.text(0.5, 0.5, f"Tidak cukup data untuk MDS (n={n}, min=3)",
                ha='center', va='center', transform=ax.transAxes,
                fontsize=11, color='#888888',
                bbox=dict(boxstyle='round', fc='#f5f5f5', ec='#cccccc'))
        ax.set_title(f"Scatter Plot — {scenario_name}",
                     fontsize=11, fontweight='bold', pad=10)
        return

    mds    = MDS(n_components=2, dissimilarity='precomputed',
                 random_state=42, normalized_stress='auto')
    coords = mds.fit_transform(dist_matrix)

    rng          = np.random.RandomState(42)
    coord_range  = coords.ptp(axis=0)
    jitter_scale = coord_range.mean() * 0.025
    coords_j     = coords + rng.randn(n, 2) * jitter_scale

    unique_clusters = sorted(set(labels))
    n_clusters      = len(unique_clusters)
    cmap_name       = 'tab10' if n_clusters <= 10 else 'tab20'
    palette         = matplotlib.colormaps[cmap_name]
    color_map       = {c: palette(i) for i, c in enumerate(unique_clusters)}

    for c in unique_clusters:
        mask         = labels == c
        n_in_cluster = int(mask.sum())

        sub = ""
        if dataset and file_labels:
            src_counts = {}
            for i, d in enumerate(dataset):
                if labels[i] == c:
                    s = d.get('source', '')
                    src_counts[s] = src_counts.get(s, 0) + 1
            if src_counts:
                dom = max(src_counts, key=src_counts.get)
                sub = f" ≈{dom[:18]}"

        ax.scatter(coords_j[mask, 0], coords_j[mask, 1],
                   color=color_map[c], s=80, alpha=0.85,
                   edgecolors='white', linewidths=0.8, zorder=3)
        ax.scatter([], [], color=color_map[c], s=80,
                   edgecolors='white', linewidths=0.8,
                   label=f"Cluster {c}{sub} (n={n_in_cluster})")

    if dataset and n <= 60:
        for i, d in enumerate(dataset):
            ip = d.get('ip', str(i))
            label_text = (f".{ip.split('.')[-1]}" if '.' in ip else f"H{i}")
            ax.annotate(label_text,
                        (coords_j[i, 0], coords_j[i, 1]),
                        fontsize=7, alpha=0.75, fontweight='500',
                        color=color_map.get(labels[i], '#888888'),
                        xytext=(3, 3), textcoords='offset points')

    stress_norm = (mds.stress_ / (np.sum(dist_matrix**2) / 2)
                   if mds.stress_ else 0)
    ax.set_title(
        f"Scatter Plot 2D (MDS) — {scenario_name}\n"
        f"Stress={stress_norm:.4f}  |  {n_clusters} clusters",
        fontsize=11, fontweight='bold', pad=10
    )
    ax.set_xlabel("MDS Dimension 1", fontsize=10)
    ax.set_ylabel("MDS Dimension 2", fontsize=10)
    ax.legend(loc='best', fontsize=8, framealpha=0.85,
              markerscale=0.9, title="Cluster", title_fontsize=9)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.tick_params(labelsize=9)


# ═══════════════════════════════════════════════════════════════════
# 8. GRAPH 3 — DENDROGRAM
# ═══════════════════════════════════════════════════════════════════
def plot_dendrogram(dist_matrix: np.ndarray, labels: np.ndarray,
                    ax, scenario_name: str = "",
                    cut_threshold: float = None,
                    dataset: list = None):
    n = len(dist_matrix)
    if n < 2:
        ax.text(0.5, 0.5, "Terlalu sedikit host untuk dendrogram",
                ha='center', va='center', transform=ax.transAxes)
        return

    # Konversi square matrix → condensed form untuk scipy linkage
    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method='average')

    # Label tiap leaf: IP pendek atau indeks
    leaf_labels = []
    if dataset:
        for d in dataset:
            ip = d.get('ip', '')
            lbl = (f".{ip.split('.')[-1]}" if '.' in ip
                   else d.get('source', '')[:12] or f"H{len(leaf_labels)}")
            leaf_labels.append(lbl)
    else:
        leaf_labels = [str(i) for i in range(n)]

    # Warna per cluster untuk leaf
    unique_clusters = sorted(set(labels))
    cmap_name = 'tab10' if len(unique_clusters) <= 10 else 'tab20'
    palette   = matplotlib.colormaps[cmap_name]
    color_map = {c: palette(i) for i, c in enumerate(unique_clusters)}

    # Gambar dendrogram
    ddata = dendrogram(
        Z,
        ax=ax,
        labels=leaf_labels,
        leaf_rotation=90,
        leaf_font_size=7,
        color_threshold=cut_threshold if cut_threshold is not None else 0,
        above_threshold_color='#aaaaaa',
    )

    # Garis cut
    if cut_threshold is not None:
        ax.axhline(y=cut_threshold, color='#E91E63', linewidth=1.8,
                   linestyle='--',
                   label=f'Cut threshold = {cut_threshold:.3f}')
        ax.legend(loc='upper right', fontsize=8)
    else:
        # Hitung tinggi cut yang menghasilkan n cluster yang sama
        n_clusters = len(unique_clusters)
        if len(Z) >= n_clusters - 1:
            # Tinggi merge ke-N dari bawah (yang menghasilkan n_clusters)
            merge_heights = sorted(Z[:, 2])
            if n_clusters <= len(merge_heights):
                cut_h = merge_heights[-(n_clusters - 1)] - 1e-9
                ax.axhline(y=cut_h, color='#E91E63', linewidth=1.8,
                           linestyle='--',
                           label=f'Cut @ {n_clusters} clusters')
                ax.legend(loc='upper right', fontsize=8)

    ax.set_title(f"Dendrogram (Average Linkage) — {scenario_name}",
                 fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel("Host", fontsize=10)
    ax.set_ylabel("Distance (DYNAMIT)", fontsize=10)
    ax.tick_params(axis='x', labelsize=7)
    ax.tick_params(axis='y', labelsize=9)


# ═══════════════════════════════════════════════════════════════════
# 9. GRAPH 4 — PARAMETER SWEEP PLOT
# ═══════════════════════════════════════════════════════════════════
def plot_threshold_sweep(sweep_results: list, best_val: float,
                          ax, scenario_name: str = "",
                          has_gt: bool = False):
    t_vals  = [r['threshold']  for r in sweep_results]
    nc_vals = [r['n_clusters'] for r in sweep_results]
    ari_vals = [r['ari']        for r in sweep_results]
    sil_vals = [r['silhouette'] for r in sweep_results]

    ax2 = ax.twinx()
    ax.bar(t_vals, nc_vals, width=0.025, alpha=0.2,
           color='steelblue', label='# Clusters')

    if has_gt:
        valid = [(t, v) for t, v in zip(t_vals, ari_vals)
                 if not np.isnan(v)]
        if valid:
            tx, ty = zip(*valid)
            ax2.plot(tx, ty, color='#2196F3', lw=2, marker='o',
                     markersize=4, label='ARI')
        metric_label = 'ARI'
        metric_color = '#2196F3'
    else:
        valid = [(t, v) for t, v in zip(t_vals, sil_vals)
                 if not np.isnan(v)]
        if valid:
            tx, ty = zip(*valid)
            ax2.plot(tx, ty, color='#4CAF50', lw=2, marker='s',
                     markersize=4, label='Silhouette')
        metric_label = 'Silhouette'
        metric_color = '#4CAF50'

    ax2.axvline(best_val, color='#E91E63', lw=1.8, linestyle='--',
                label=f'Best={best_val:.3f}')
    ax2.set_ylim(-0.1, 1.1)

    ax.set_xlabel("distance_threshold", fontsize=10)
    ax.set_ylabel("# Clusters", fontsize=9, color='#555555')
    ax2.set_ylabel(metric_label, fontsize=10, color=metric_color)

    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2,
              loc='upper right', fontsize=8, framealpha=0.85)
    ax.set_title(
        f"Threshold Sweep — {scenario_name}\n"
        f"Best threshold={best_val:.3f} "
        f"(by {'ARI' if has_gt else 'Silhouette'})",
        fontsize=11, fontweight='bold', pad=10
    )
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.tick_params(labelsize=9)


def plot_k_sweep(sweep_results: list, best_k: int,
                 ax, scenario_name: str = "",
                 has_gt: bool = False):
    k_vals   = [r['k']          for r in sweep_results]
    ari_vals = [r['ari']        for r in sweep_results]
    sil_vals = [r['silhouette'] for r in sweep_results]

    ax2 = ax.twinx()

    if has_gt:
        valid = [(k, v) for k, v in zip(k_vals, ari_vals)
                 if not np.isnan(v)]
        if valid:
            kx, ky = zip(*valid)
            ax2.plot(kx, ky, color='#2196F3', lw=2, marker='o',
                     markersize=5, label='ARI')
        metric_label = 'ARI'
        metric_color = '#2196F3'
    else:
        valid = [(k, v) for k, v in zip(k_vals, sil_vals)
                 if not np.isnan(v)]
        if valid:
            kx, ky = zip(*valid)
            ax2.plot(kx, ky, color='#4CAF50', lw=2, marker='s',
                     markersize=5, label='Silhouette')
        metric_label = 'Silhouette'
        metric_color = '#4CAF50'

    ax2.axvline(best_k, color='#E91E63', lw=1.8, linestyle='--',
                label=f'Best K={best_k}')
    ax2.set_ylim(-0.1, 1.1)

    ax.set_xlabel("n_clusters (K)", fontsize=10)
    ax.set_ylabel("—", fontsize=9, color='#ffffff')   # no bar, just placeholder
    ax2.set_ylabel(metric_label, fontsize=10, color=metric_color)

    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2,
              loc='upper right', fontsize=8, framealpha=0.85)
    ax.set_title(
        f"K Sweep — {scenario_name}\n"
        f"Best K={best_k} "
        f"(by {'ARI' if has_gt else 'Silhouette'})",
        fontsize=11, fontweight='bold', pad=10
    )
    ax.set_xticks(k_vals)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.tick_params(labelsize=9)


# ═══════════════════════════════════════════════════════════════════
# 10. PRINT SUMMARY
# ═══════════════════════════════════════════════════════════════════
def print_summary(metrics: dict, mode_str: str,
                  n_true: int = None, scenario: str = ""):
    w = 54
    print(f"\n{'='*w}")
    print(f"  AHC RESULT — {scenario}")
    print(f"{'='*w}")
    print(f"  {mode_str}")
    if n_true:
        print(f"  True Clusters (Ground)  : {n_true}")
    print(f"  Predicted Clusters      : {metrics['n_clusters_pred']}")
    print(f"  {'─'*46}")
    if not np.isnan(metrics['ari']):
        print(f"  ARI  (Accuracy vs GT)   : {metrics['ari']:.4f}")
        print(f"  NMI  (Mutual Info)      : {metrics['nmi']:.4f}")
        print(f"  FMI  (Fowlkes-Mallows)  : {metrics['fmi']:.4f}")
    else:
        print(f"  ARI / NMI / FMI         : N/A (no ground truth)")
    if not np.isnan(metrics['silhouette']):
        print(f"  Silhouette Score        : {metrics['silhouette']:.4f}")
    else:
        print(f"  Silhouette Score        : N/A")
    print(f"{'='*w}\n")


# ═══════════════════════════════════════════════════════════════════
# 11. FIGURE BUILDER
# ═══════════════════════════════════════════════════════════════════
def _build_figure(dataset, true_labels, file_labels,
                  dist_matrix, labels, metrics,
                  mode_str, sweep_results, sweep_type,
                  best_param, cut_threshold,
                  scenario, args):
    has_gt = true_labels is not None
    do_sweep = sweep_results is not None

    if do_sweep:
        fig = plt.figure(figsize=(18, 13))
        fig.patch.set_facecolor('#fafafa')
        gs  = gridspec.GridSpec(2, 2, figure=fig,
                                hspace=0.50, wspace=0.35,
                                height_ratios=[1.1, 1.0])
        ax_tl = fig.add_subplot(gs[0, 0])
        ax_tr = fig.add_subplot(gs[0, 1])
        ax_bl = fig.add_subplot(gs[1, 0])
        ax_br = fig.add_subplot(gs[1, 1])
    else:
        fig = plt.figure(figsize=(21, 7))
        fig.patch.set_facecolor('#fafafa')
        gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)
        ax_tl = fig.add_subplot(gs[0, 0])
        ax_tr = fig.add_subplot(gs[0, 1])
        ax_bl = fig.add_subplot(gs[0, 2])
        ax_br = None

    # Panel kiri atas — confusion matrix / membership heatmap
    if has_gt:
        plot_confusion_matrix(true_labels, labels, ax_tl,
                              scenario_name=scenario,
                              class_names=file_labels)
    else:
        plot_membership_heatmap(dataset, labels, ax_tl,
                                scenario_name=scenario)

    # Panel kanan atas — MDS scatter
    plot_scatter_mds(dist_matrix, labels, ax_tr,
                     dataset=dataset, true_labels=true_labels,
                     scenario_name=scenario, file_labels=file_labels)

    # Panel kiri bawah (atau kanan dalam 1×3) — dendrogram
    plot_dendrogram(dist_matrix, labels, ax_bl,
                    scenario_name=scenario,
                    cut_threshold=cut_threshold,
                    dataset=dataset)

    # Panel kanan bawah — sweep (hanya jika ada)
    if do_sweep and ax_br is not None:
        if sweep_type == 'threshold':
            plot_threshold_sweep(sweep_results, best_param, ax_br,
                                 scenario_name=scenario, has_gt=has_gt)
        else:
            plot_k_sweep(sweep_results, best_param, ax_br,
                         scenario_name=scenario, has_gt=has_gt)

    # Metric annotation
    lines = [f"{mode_str}  |  Clusters Found: {metrics['n_clusters_pred']}"]
    if not np.isnan(metrics['ari']):
        lines.append(
            f"ARI={metrics['ari']:.4f}  NMI={metrics['nmi']:.4f}  "
            f"FMI={metrics['fmi']:.4f}  Silhouette={metrics['silhouette']:.4f}"
        )
    else:
        sil_str = (f"{metrics['silhouette']:.4f}"
                   if not np.isnan(metrics['silhouette']) else "N/A")
        lines.append(f"Silhouette={sil_str}  (No ground truth)")

    fig.text(0.5, 0.005, '\n'.join(lines),
             ha='center', va='bottom', fontsize=10, color='#444444',
             bbox=dict(boxstyle='round,pad=0.4', fc='#f0f4f8',
                       ec='#cccccc', lw=0.8))

    plt.suptitle(f"AHC Clustering — {scenario}",
                 fontsize=14, fontweight='bold', y=1.01, color='#222222')

    try:
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            safe = (scenario
                    .replace(' ', '_').replace('/', '-')
                    .replace('x', 'x').replace(':', '-')
                    .replace('(', '').replace(')', '')
                    .replace(',', '')[:80])
            out_path = os.path.join(args.save_dir, f"ahc_{safe}.png")
            plt.savefig(out_path, dpi=150, bbox_inches='tight',
                        facecolor='#fafafa')
            if os.path.isfile(out_path):
                print(f"[+] Plot disimpan: {out_path}  "
                      f"({os.path.getsize(out_path)/1024:.1f} KB)")
            else:
                print(f"[!] File tidak terbuat: {out_path}")
        else:
            plt.show()
    except Exception as e:
        print(f"[!] Gagal menyimpan plot: {e}")
    finally:
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# 12. DATASET LOADERS  (identik dengan dbscan-clustering.py)
# ═══════════════════════════════════════════════════════════════════
def load_from_json(json_path: str):
    with open(json_path, 'r') as f:
        data = json.load(f)
    profiles = data.get('profiles_notebook', [])
    if not profiles:
        raise ValueError(f"Key 'profiles_notebook' kosong di {json_path}")
    dataset = [{'oui': p['oui'], 'svc': list(p['svc']), 'ip': p.get('ip', '')}
               for p in profiles]
    meta     = data.get('metadata', {})
    src      = ', '.join(meta.get('source_files', [json_path]))
    scenario = meta.get('scenario_name', f"PCAP: {src}")
    print(f"  [+] Loaded {meta.get('total_hosts', len(dataset))} hosts dari: {src}")
    return dataset, None, scenario


def load_synthetic_example(num_host: int = 5, num_cluster: int = 8):
    import random
    random.seed(42)
    SERVICES_POOL = [
        ['HTTP', 'HTTPS', 'DNS', 'DHCP'], ['SSH', 'TELNET', 'FTP', 'SMTP'],
        ['RDP', 'VNC', 'SMB', 'NETBIOS'], ['NTP', 'SNMP', 'SYSLOG', 'TFTP'],
        ['MQTT', 'UPNP', 'MDNS', 'XMPP'], ['IMAP', 'POP3', 'LDAP', 'KERBEROS'],
        ['IPP', 'AFP', 'RTSP', 'RTP'],    ['DHCP', 'NTP', 'HTTP', 'UPNP'],
    ]
    oui_base = [f"{i:02x}:{(i*7)%256:02x}:{(i*13)%256:02x}"
                for i in range(1, num_cluster + 1)]
    dataset, true_labels = [], []
    for c in range(num_cluster):
        base_svc = SERVICES_POOL[c % len(SERVICES_POOL)]
        for h in range(num_host):
            svc = base_svc.copy()
            extra = random.choice(['HTTP', 'DNS', 'NTP', 'SSH', 'FTP'])
            if extra not in svc:
                svc.append(extra)
            if len(svc) > 2 and random.random() > 0.5:
                svc.pop(random.randrange(len(svc)))
            oui = oui_base[c] if random.random() > 0.3 else \
                  f"{(c+10)%256:02x}:{(h+5)%256:02x}:{c:02x}"
            dataset.append({'oui': oui, 'svc': sorted(set(svc))})
            true_labels.append(c)
    return dataset, true_labels, f"Synthetic {num_host}H x {num_cluster}C"


def collect_json_files(dirpath: str) -> list:
    files = [os.path.join(dirpath, e) for e in sorted(os.listdir(dirpath))
             if os.path.isfile(os.path.join(dirpath, e))
             and e.lower().endswith('.json')]
    if not files:
        raise FileNotFoundError(f"Tidak ada .json di: {dirpath}")
    return files


def _stem(filepath: str, data: dict = None) -> str:
    if data:
        meta = data.get('metadata', {})
        if meta.get('label'):
            return meta['label']
        src_files = meta.get('source_files', [])
        if src_files:
            parent = os.path.basename(os.path.dirname(src_files[0]))
            if parent and parent not in ('.', ''):
                return parent
    return os.path.splitext(os.path.basename(filepath))[0]


def load_from_dir_merged(dirpath: str) -> tuple:
    files = collect_json_files(dirpath)
    dataset, true_labels, file_labels = [], [], []
    print(f"  [dir-merge] {len(files)} file JSON ditemukan:")
    for idx, fpath in enumerate(files):
        with open(fpath) as f:
            data = json.load(f)
        profiles = data.get('profiles_notebook', [])
        detail   = {d['ip']: d for d in data.get('profiles_detail', [])}
        label    = _stem(fpath, data)
        file_labels.append(label)
        n_before = len(dataset)
        for p in profiles:
            dataset.append({
                'oui':    p['oui'],
                'svc':    list(p['svc']),
                'ip':     detail.get(p.get('ip', ''), {}).get('ip', p.get('ip', '')),
                'source': label,
            })
            true_labels.append(idx)
        print(f"    [{idx:2d}] {label:45s} -> {len(dataset)-n_before} hosts")
    scenario = (f"DIR-MERGE: {os.path.basename(dirpath)} "
                f"({len(files)} files, {len(dataset)} hosts)")
    return dataset, true_labels, scenario, file_labels


def load_from_dir_batch(dirpath: str) -> list:
    files = collect_json_files(dirpath)
    runs  = []
    print(f"  [dir-batch] {len(files)} file JSON -> {len(files)} run:")
    for fpath in files:
        try:
            dataset, _, scenario = load_from_json(fpath)
            runs.append((dataset, None, _stem(fpath, None), fpath))
        except Exception as e:
            print(f"    [SKIP] {os.path.basename(fpath)}: {e}")
    return runs


# ═══════════════════════════════════════════════════════════════════
# 13. SINGLE-RUN HELPER
# ═══════════════════════════════════════════════════════════════════
def _single_run(dataset, true_labels, scenario, file_labels, args):
    print(f"\n  >> Run: {scenario}  ({len(dataset)} hosts)")
    if len(dataset) < 2:
        print(f"  [SKIP] Terlalu sedikit host ({len(dataset)}), min=2")
        return

    X, n_oui, ohe, mlb = extract_features(dataset)
    hybrid_metric       = create_dynamit_metric(n_oui)

    print("  Computing hybrid distance matrix...")
    dist_matrix = pairwise_distances(X, metric=hybrid_metric)

    sweep_results = None
    sweep_type    = None
    best_param    = None
    cut_threshold = None  # untuk dendrogram cut line (threshold mode saja)

    # ── Mode selection ────────────────────────────────────────────
    if args.threshold_search:
        print(f"  Sweeping threshold [{args.t_min:.2f} -> {args.t_max:.2f}, "
              f"step={args.t_step:.2f}]...")
        sweep_results, best_param, labels = threshold_sweep(
            dist_matrix, true_labels,
            args.t_min, args.t_max, args.t_step,
        )
        cut_threshold = best_param
        sweep_type    = 'threshold'
        print(f"  Best threshold selected: {best_param:.4f}")

    elif args.threshold is not None:
        best_param    = args.threshold
        cut_threshold = best_param
        _, labels     = run_ahc_threshold(dist_matrix, best_param)
        print(f"  threshold = {best_param:.4f}")

    elif args.k_search:
        k_max = args.k_max if args.k_max else max(2, len(dataset) // 2)
        print(f"  Sweeping K [{args.k_min} -> {k_max}]...")
        sweep_results, best_param, labels = k_sweep(
            dist_matrix, true_labels, args.k_min, k_max
        )
        sweep_type = 'k'
        print(f"  Best K selected: {best_param}")

    else:  # fixed k
        best_param = args.k
        _, labels  = run_ahc_k(dist_matrix, best_param)
        print(f"  K = {best_param}")

    labels    = np.array(labels)
    metrics   = compute_metrics(labels, dist_matrix, true_labels)
    n_true    = len(set(true_labels)) if true_labels is not None else None

    # mode description string for summary
    if args.threshold_search or args.threshold is not None:
        mode_str = f"Mode: distance_threshold={cut_threshold:.3f}  linkage=average"
    else:
        mode_str = f"Mode: n_clusters={best_param}  linkage=average"

    print_summary(metrics, mode_str, n_true=n_true, scenario=scenario)

    _build_figure(dataset, true_labels, file_labels,
                  dist_matrix, labels, metrics,
                  mode_str, sweep_results, sweep_type,
                  best_param, cut_threshold,
                  scenario, args)

    return metrics


# ═══════════════════════════════════════════════════════════════════
# 14. MAIN
# ═══════════════════════════════════════════════════════════════════
def main(args):
    print(f"\n{'='*54}")
    print(f"  AHC CLUSTERING — Resource-Aware Honeynet")
    print(f"  Metric : DYNAMIT (Manhattan OUI + Jaccard SVC)")
    print(f"  Linkage: average (UPGMA)")
    print(f"{'='*54}")

    file_labels = None

    if args.dir:
        if not os.path.isdir(args.dir):
            print(f"[ERROR] Direktori tidak ditemukan: {args.dir}")
            sys.exit(1)
        if args.batch:
            runs = load_from_dir_batch(args.dir)
            print(f"\n  Menjalankan {len(runs)} run...")
            for dataset_b, _, scenario_b, _ in runs:
                _single_run(dataset_b, None, scenario_b, None,
                            copy.copy(args))
            return
        dataset, true_labels, scenario, file_labels = \
            load_from_dir_merged(args.dir)
        if args.label:
            scenario = args.label

    elif args.json:
        if not os.path.isfile(args.json):
            print(f"[ERROR] File tidak ditemukan: {args.json}")
            sys.exit(1)
        dataset, true_labels, scenario = load_from_json(args.json)
        if args.label:
            scenario = args.label

    else:
        dataset, true_labels, scenario = load_synthetic_example(
            num_host=args.num_host, num_cluster=args.num_cluster
        )

    print(f"  Scenario : {scenario}")
    print(f"  Dataset  : {len(dataset)} hosts")
    if len(dataset) < 2:
        print(f"[ERROR] Dataset terlalu kecil ({len(dataset)} host). Min=2.")
        return None
    if true_labels:
        print(f"  GT Clust : {len(set(true_labels))}")

    _single_run(dataset, true_labels, scenario, file_labels, args)


# ═══════════════════════════════════════════════════════════════════
# 15. CLI
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='AHC Clustering + Hybrid Metric (DYNAMIT)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Mode threshold (algoritma tentukan K — default, sesuai notebook Zaki):
  python ahc_clustering.py --dir output/ --threshold-search --save-dir plots/
  python ahc_clustering.py --json output/gns3-simulation-v3.json --threshold 0.5 --save-dir plots/

Mode fixed-K (jumlah cluster manual atau sweep):
  python ahc_clustering.py --dir output/ --k-search --save-dir plots/
  python ahc_clustering.py --dir output/ --k 6 --save-dir plots/

Dataset sintetis:
  python ahc_clustering.py --inline --num-host 5 --num-cluster 8 --threshold-search

Rentang threshold kustom:
  python ahc_clustering.py --dir output/ --threshold-search \\
      --t-min 0.2 --t-max 0.8 --t-step 0.02 --save-dir plots/
        """
    )

    # Input source
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--json',   type=str,
                     help='Path ke satu file JSON dari pcap_host_profiler.py')
    src.add_argument('--dir',    type=str,
                     help='Direktori berisi banyak .json\n'
                          '  Default: MERGE semua, GT dari nama file\n'
                          '  Tambah --batch untuk run per file terpisah')
    src.add_argument('--inline', action='store_true',
                     help='Gunakan dataset sintetis bawaan')

    parser.add_argument('--batch', action='store_true',
                        help='[--dir] Jalankan tiap .json sebagai run terpisah')
    parser.add_argument('--label', type=str, default=None,
                        help='Override nama skenario pada judul grafik')
    parser.add_argument('--num-host',    type=int, default=5)
    parser.add_argument('--num-cluster', type=int, default=8)

    # AHC mode — mutually exclusive
    mode_grp = parser.add_mutually_exclusive_group()
    mode_grp.add_argument('--threshold', type=float, default=None,
                           metavar='FLOAT',
                           help='Threshold cut dendrogram (mode A, fixed).\n'
                                'Contoh: --threshold 0.6')
    mode_grp.add_argument('--threshold-search', action='store_true',
                           help='Sweep distance_threshold otomatis (mode A, search).\n'
                                'Pilih terbaik berdasarkan ARI (GT) atau Silhouette.')
    mode_grp.add_argument('--k', type=int, default=None,
                           metavar='INT',
                           help='Jumlah cluster tetap (mode B, fixed).\n'
                                'Contoh: --k 5')
    mode_grp.add_argument('--k-search', action='store_true',
                           help='Sweep n_clusters otomatis (mode B, search).\n'
                                'Pilih terbaik berdasarkan ARI (GT) atau Silhouette.')

    # Threshold sweep params
    parser.add_argument('--t-min',  type=float, default=0.1,
                        help='[--threshold-search] Batas bawah (default: 0.1)')
    parser.add_argument('--t-max',  type=float, default=1.0,
                        help='[--threshold-search] Batas atas (default: 1.0)')
    parser.add_argument('--t-step', type=float, default=0.05,
                        help='[--threshold-search] Step (default: 0.05)')

    # K sweep params
    parser.add_argument('--k-min', type=int, default=2,
                        help='[--k-search] K minimum (default: 2)')
    parser.add_argument('--k-max', type=int, default=None,
                        help='[--k-search] K maksimum (default: n_hosts // 2)')

    parser.add_argument('--save-dir', type=str, default=None,
                        help='Direktori simpan plot PNG (default: tampilkan)')

    args = parser.parse_args()

    # Default mode jika tidak ada yang dipilih: threshold-search
    if (args.threshold is None and not args.threshold_search
            and args.k is None and not args.k_search):
        args.threshold_search = True

    main(args)
