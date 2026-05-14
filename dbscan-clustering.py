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

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
    silhouette_score,
    pairwise_distances,
)
from sklearn.manifold import MDS
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
# 3. DBSCAN RUNNER
# ═══════════════════════════════════════════════════════════════════
def run_dbscan(dist_matrix: np.ndarray, eps: float, min_samples: int = 1):
    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric='precomputed',
        n_jobs=-1,
    )
    labels = dbscan.fit_predict(dist_matrix)
    return dbscan, labels


def eps_sweep(dist_matrix: np.ndarray, true_labels,
              eps_min: float, eps_max: float, eps_step: float,
              min_samples: int = 1):
    eps_range = np.arange(eps_min, eps_max + eps_step * 0.5, eps_step)
    results = []

    for eps in eps_range:
        _, labels = run_dbscan(dist_matrix, eps, min_samples)

        n_clusters = len(set(labels) - {-1})
        n_noise    = int(np.sum(labels == -1))

        ari = sil = np.nan

        if true_labels is not None:
            mask = labels != -1
            if mask.sum() > 1 and len(set(np.array(true_labels)[mask])) > 1:
                ari = adjusted_rand_score(
                    np.array(true_labels)[mask], labels[mask]
                )

        # Silhouette butuh >= 2 cluster dan >= 2 non-noise points
        n_valid = int((labels != -1).sum())
        if n_clusters >= 2 and n_valid >= 2:
            try:
                sil = silhouette_score(
                    dist_matrix[np.ix_(labels != -1, labels != -1)],
                    labels[labels != -1],
                    metric='precomputed',
                )
            except Exception:
                sil = np.nan

        results.append({
            'eps':        round(float(eps), 4),
            'n_clusters': n_clusters,
            'n_noise':    n_noise,
            'ari':        ari,
            'silhouette': sil,
        })

    # Pilih eps terbaik
    if true_labels is not None:
        valid = [(r, r['ari']) for r in results if not np.isnan(r['ari'])]
        if valid:
            best_r = max(valid, key=lambda x: x[1])[0]
        else:
            best_r = results[len(results) // 2]
    else:
        valid = [(r, r['silhouette']) for r in results
                 if not np.isnan(r['silhouette'])]
        if valid:
            best_r = max(valid, key=lambda x: x[1])[0]
        else:
            best_r = results[len(results) // 2]

    _, best_labels = run_dbscan(dist_matrix, best_r['eps'], min_samples)
    return results, best_r['eps'], best_labels


# ═══════════════════════════════════════════════════════════════════
# 4. METRICS
# ═══════════════════════════════════════════════════════════════════
def compute_metrics(labels: np.ndarray, dist_matrix: np.ndarray,
                    true_labels=None):
    unique    = set(labels)
    n_pred    = len(unique) - (1 if -1 in unique else 0)
    n_noise   = int(np.sum(labels == -1))
    noise_pct = n_noise / len(labels) * 100

    result = {
        'n_clusters_pred': n_pred,
        'n_noise':         n_noise,
        'noise_pct':       noise_pct,
        'ari':        np.nan,
        'nmi':        np.nan,
        'fmi':        np.nan,
        'silhouette': np.nan,
    }

    # Silhouette (pengganti DBCV untuk DBSCAN)
    n_valid = int((labels != -1).sum())
    if n_pred >= 2 and n_valid >= 2:
        try:
            result['silhouette'] = silhouette_score(
                dist_matrix[np.ix_(labels != -1, labels != -1)],
                labels[labels != -1],
                metric='precomputed',
            )
        except Exception:
            pass

    if true_labels is not None:
        mask = labels != -1
        if mask.sum() > 0:
            tl = np.array(true_labels)[mask]
            pl = labels[mask]
            result['ari'] = adjusted_rand_score(tl, pl)
            result['nmi'] = normalized_mutual_info_score(tl, pl)
            result['fmi'] = fowlkes_mallows_score(tl, pl)

    return result


# ═══════════════════════════════════════════════════════════════════
# 5A. GRAPH 1A — CONFUSION MATRIX
# ═══════════════════════════════════════════════════════════════════
def plot_confusion_matrix(true_labels, predicted_labels, ax,
                          scenario_name: str = "",
                          class_names: list = None):
    true_labels      = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    noise_mask = predicted_labels == -1
    n_noise    = int(noise_mask.sum())

    t_clean = true_labels[~noise_mask]
    p_clean = predicted_labels[~noise_mask]

    true_unique = np.unique(true_labels)
    pred_unique = np.unique(p_clean) if len(p_clean) > 0 else np.array([])

    if len(p_clean) > 0:
        n_true = len(true_unique)
        n_pred = len(pred_unique)
        cm = np.zeros((n_true, n_pred), dtype=int)
        true_idx = {t: i for i, t in enumerate(true_unique)}
        pred_idx = {p: i for i, p in enumerate(pred_unique)}
        for t, p in zip(t_clean, p_clean):
            if t in true_idx and p in pred_idx:
                cm[true_idx[t], pred_idx[p]] += 1

        if cm.size > 0:
            row_ind, col_ind = linear_sum_assignment(-cm)
            aligned   = list(col_ind)
            remaining = [c for c in range(n_pred) if c not in aligned]
            aligned.extend(remaining)
            cm          = cm[:, aligned]
            pred_unique = pred_unique[aligned]
            x_labels    = [f"Pred {i}" for i in pred_unique]
    else:
        cm       = np.zeros((len(true_unique), 0), dtype=int)
        x_labels = []

    if n_noise > 0:
        noise_col = np.array([
            int(((true_labels == t) & noise_mask).sum()) for t in true_unique
        ]).reshape(-1, 1)
        cm       = np.hstack([cm, noise_col])
        x_labels = x_labels + [f"Noise\n({n_noise})"]

    if class_names and len(class_names) >= len(true_unique):
        y_labels = [f"{class_names[t]}" for t in true_unique]
    else:
        y_labels = [f"True {t}" for t in true_unique]

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
# 5B. GRAPH 1B — CLUSTER MEMBERSHIP HEATMAP (tanpa GT)
# ═══════════════════════════════════════════════════════════════════
def plot_membership_heatmap(dataset: list, labels: np.ndarray, ax,
                             scenario_name: str = ""):
    df = pd.DataFrame(dataset).copy()
    df['cluster']   = labels
    df['label_str'] = df['cluster'].apply(
        lambda c: "Noise" if c == -1 else f"Cluster {c}"
    )

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
    ax.set_xlabel("Service",       fontsize=10, labelpad=8)
    ax.set_ylabel("Host [Cluster]", fontsize=10, labelpad=8)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)


# ═══════════════════════════════════════════════════════════════════
# 6. GRAPH 2 — SCATTER PLOT (MDS 2D)
# ═══════════════════════════════════════════════════════════════════
def plot_scatter_mds(dist_matrix: np.ndarray, labels: np.ndarray,
                     ax, dataset: list = None,
                     true_labels=None, scenario_name: str = "",
                     file_labels: list = None):
    n       = len(labels)
    n_valid = int((labels != -1).sum())

    if n_valid < 3:
        ax.text(0.5, 0.5,
                f"Tidak cukup data untuk MDS\n"
                f"({n_valid} titik non-noise, min=3)\n"
                f"Turunkan --eps atau --min-pkts",
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

    unique_clusters = sorted(set(labels) - {-1})
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

    noise_mask = labels == -1
    if noise_mask.any():
        ax.scatter(coords_j[noise_mask, 0], coords_j[noise_mask, 1],
                   color='#aaaaaa', marker='X', s=80, alpha=0.7,
                   label=f"Noise (n={noise_mask.sum()})", zorder=2)

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
        f"Stress={stress_norm:.4f}  |  {n_clusters} clusters  |  "
        f"Noise={int(noise_mask.sum())}",
        fontsize=11, fontweight='bold', pad=10
    )
    ax.set_xlabel("MDS Dimension 1", fontsize=10)
    ax.set_ylabel("MDS Dimension 2", fontsize=10)
    ax.legend(loc='best', fontsize=8, framealpha=0.85,
              markerscale=0.9, title="Cluster", title_fontsize=9)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.tick_params(labelsize=9)


# ═══════════════════════════════════════════════════════════════════
# 7. GRAPH 3 — EPS SWEEP PLOT
# ═══════════════════════════════════════════════════════════════════
def plot_eps_sweep(sweep_results: list, best_eps: float, ax,
                   scenario_name: str = "", has_gt: bool = False):
    eps_vals  = [r['eps']        for r in sweep_results]
    ari_vals  = [r['ari']        for r in sweep_results]
    sil_vals  = [r['silhouette'] for r in sweep_results]
    nc_vals   = [r['n_clusters'] for r in sweep_results]
    noise_vals = [r['n_noise']   for r in sweep_results]

    ax2 = ax.twinx()

    # Cluster count + noise as background bars
    ax.bar(eps_vals, nc_vals, width=0.03, alpha=0.18,
           color='steelblue', label='# Clusters')
    ax.bar(eps_vals, noise_vals, width=0.03, alpha=0.12,
           color='salmon', label='# Noise', bottom=0)

    # Primary metric line
    if has_gt:
        valid_ari = [(e, v) for e, v in zip(eps_vals, ari_vals)
                     if not np.isnan(v)]
        if valid_ari:
            ex, ey = zip(*valid_ari)
            ax2.plot(ex, ey, color='#2196F3', lw=2, marker='o',
                     markersize=4, label='ARI', zorder=5)
        primary_label = 'ARI'
    else:
        valid_sil = [(e, v) for e, v in zip(eps_vals, sil_vals)
                     if not np.isnan(v)]
        if valid_sil:
            ex, ey = zip(*valid_sil)
            ax2.plot(ex, ey, color='#4CAF50', lw=2, marker='s',
                     markersize=4, label='Silhouette', zorder=5)
        primary_label = 'Silhouette'

    # Mark best eps
    ax2.axvline(best_eps, color='#E91E63', lw=1.8, linestyle='--',
                label=f'Best eps={best_eps:.2f}', zorder=6)

    ax.set_xlabel("eps (DBSCAN neighborhood radius)", fontsize=10)
    ax.set_ylabel("Count (Clusters / Noise)", fontsize=9, color='#555555')
    ax2.set_ylabel(primary_label, fontsize=10, color='#2196F3'
                   if has_gt else '#4CAF50')
    ax2.set_ylim(-0.1, 1.1)

    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2,
              loc='upper right', fontsize=8, framealpha=0.85)

    ax.set_title(f"Eps Sweep — {scenario_name}\n"
                 f"Best eps={best_eps:.2f} (by {'ARI' if has_gt else 'Silhouette'})",
                 fontsize=11, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.tick_params(labelsize=9)


# ═══════════════════════════════════════════════════════════════════
# 8. PRINT SUMMARY
# ═══════════════════════════════════════════════════════════════════
def print_summary(metrics: dict, eps: float, n_true: int = None,
                  scenario: str = ""):
    w = 54
    print(f"\n{'='*w}")
    print(f"  DBSCAN RESULT — {scenario}")
    print(f"{'='*w}")
    print(f"  eps used                : {eps:.4f}")
    if n_true:
        print(f"  True Clusters (Ground)  : {n_true}")
    print(f"  Predicted Clusters      : {metrics['n_clusters_pred']}")
    print(f"  Noise Points Detected   : {metrics['n_noise']} ({metrics['noise_pct']:.1f}%)")
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
# 9. DATASET LOADERS  (identik dengan hdbscan-clustering.py)
# ═══════════════════════════════════════════════════════════════════
def load_from_json(json_path: str):
    with open(json_path, 'r') as f:
        data = json.load(f)

    profiles = data.get('profiles_notebook', [])
    if not profiles:
        raise ValueError(f"Key 'profiles_notebook' kosong di {json_path}")

    dataset = []
    for p in profiles:
        dataset.append({
            'oui': p['oui'],
            'svc': list(p['svc']),
            'ip':  p.get('ip', ''),
        })

    meta     = data.get('metadata', {})
    src      = ', '.join(meta.get('source_files', [json_path]))
    scenario = meta.get('scenario_name', f"PCAP: {src}")
    n_hosts  = meta.get('total_hosts', len(dataset))

    print(f"  [+] Loaded {n_hosts} hosts dari: {src}")
    return dataset, None, scenario


def load_synthetic_example(num_host: int = 5, num_cluster: int = 8):
    import random
    random.seed(42)

    SERVICES_POOL = [
        ['HTTP', 'HTTPS', 'DNS', 'DHCP'],
        ['SSH', 'TELNET', 'FTP', 'SMTP'],
        ['RDP', 'VNC', 'SMB', 'NETBIOS'],
        ['NTP', 'SNMP', 'SYSLOG', 'TFTP'],
        ['MQTT', 'UPNP', 'MDNS', 'XMPP'],
        ['IMAP', 'POP3', 'LDAP', 'KERBEROS'],
        ['IPP', 'AFP', 'RTSP', 'RTP'],
        ['DHCP', 'NTP', 'HTTP', 'UPNP'],
    ]
    oui_base = [f"{i:02x}:{(i*7)%256:02x}:{(i*13)%256:02x}"
                for i in range(1, num_cluster + 1)]

    dataset, true_labels = [], []
    for c in range(num_cluster):
        base_svc = SERVICES_POOL[c % len(SERVICES_POOL)]
        base_oui = oui_base[c]
        for h in range(num_host):
            svc   = base_svc.copy()
            extra = random.choice(['HTTP', 'DNS', 'NTP', 'SSH', 'FTP'])
            if extra not in svc:
                svc.append(extra)
            if len(svc) > 2 and random.random() > 0.5:
                svc.pop(random.randrange(len(svc)))
            oui = base_oui if random.random() > 0.3 else \
                  f"{(c+10)%256:02x}:{(h+5)%256:02x}:{c:02x}"
            dataset.append({'oui': oui, 'svc': sorted(set(svc))})
            true_labels.append(c)

    return dataset, true_labels, f"Synthetic {num_host}H × {num_cluster}C"


def collect_json_files(dirpath: str) -> list:
    files = []
    for entry in sorted(os.listdir(dirpath)):
        full = os.path.join(dirpath, entry)
        if os.path.isfile(full) and entry.lower().endswith('.json'):
            files.append(full)
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
    files       = collect_json_files(dirpath)
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
        print(f"    [{idx:2d}] {label:45s} → {len(dataset)-n_before} hosts")

    scenario = (f"DIR-MERGE: {os.path.basename(dirpath)} "
                f"({len(files)} files, {len(dataset)} hosts)")
    return dataset, true_labels, scenario, file_labels


def load_from_dir_batch(dirpath: str) -> list:
    files = collect_json_files(dirpath)
    runs  = []
    print(f"  [dir-batch] {len(files)} file JSON → {len(files)} clustering run:")
    for fpath in files:
        try:
            dataset, _, scenario = load_from_json(fpath)
            runs.append((dataset, None, _stem(fpath, None), fpath))
        except Exception as e:
            print(f"    [SKIP] {os.path.basename(fpath)}: {e}")
    return runs


# ═══════════════════════════════════════════════════════════════════
# 10. FIGURE BUILDER
# ═══════════════════════════════════════════════════════════════════
def _build_figure(dataset, true_labels, file_labels,
                  dist_matrix, labels, metrics,
                  eps_used, sweep_results,
                  scenario, args):
    has_gt    = true_labels is not None
    do_sweep  = sweep_results is not None

    if do_sweep:
        # Layout 2 baris: [CM/heatmap | scatter] atas, [eps sweep] bawah
        fig = plt.figure(figsize=(16, 12))
        fig.patch.set_facecolor('#fafafa')
        gs  = gridspec.GridSpec(2, 2, figure=fig,
                                hspace=0.45, wspace=0.35,
                                height_ratios=[1.1, 0.8])
        ax_left  = fig.add_subplot(gs[0, 0])
        ax_right = fig.add_subplot(gs[0, 1])
        ax_sweep = fig.add_subplot(gs[1, :])
    else:
        fig = plt.figure(figsize=(16, 7))
        fig.patch.set_facecolor('#fafafa')
        gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
        ax_left  = fig.add_subplot(gs[0, 0])
        ax_right = fig.add_subplot(gs[0, 1])
        ax_sweep = None

    # Panel kiri
    if has_gt:
        plot_confusion_matrix(true_labels, labels, ax_left,
                              scenario_name=scenario,
                              class_names=file_labels)
    else:
        plot_membership_heatmap(dataset, labels, ax_left,
                                scenario_name=scenario)

    # Panel kanan — scatter MDS
    plot_scatter_mds(dist_matrix, labels, ax_right,
                     dataset=dataset, true_labels=true_labels,
                     scenario_name=scenario, file_labels=file_labels)

    # Panel bawah — eps sweep
    if do_sweep and ax_sweep is not None:
        plot_eps_sweep(sweep_results, eps_used, ax_sweep,
                       scenario_name=scenario, has_gt=has_gt)

    # Metric annotation
    lines = [
        f"eps={eps_used:.3f}  |  "
        f"Clusters Found: {metrics['n_clusters_pred']}  |  "
        f"Noise: {metrics['n_noise']} ({metrics['noise_pct']:.1f}%)"
    ]
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

    plt.suptitle(f"DBSCAN Clustering — {scenario}",
                 fontsize=14, fontweight='bold', y=1.01, color='#222222')

    try:
        rect = [0, 0.05, 1, 1] if do_sweep else [0, 0.06, 1, 1]
        plt.tight_layout(rect=rect)

        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            safe = (scenario
                    .replace(' ', '_').replace('/', '-')
                    .replace('×', 'x').replace(':', '-')
                    .replace('(', '').replace(')', '')
                    .replace(',', '')[:80])
            out_path = os.path.join(args.save_dir, f"dbscan_{safe}.png")
            plt.savefig(out_path, dpi=150, bbox_inches='tight',
                        facecolor='#fafafa')
            if os.path.isfile(out_path):
                print(f"[+] Plot disimpan: {out_path}  "
                      f"({os.path.getsize(out_path)/1024:.1f} KB)")
            else:
                print(f"[!] savefig dipanggil tapi file tidak ditemukan: {out_path}")
        else:
            plt.show()
    except Exception as e:
        print(f"[!] Gagal menyimpan plot: {e}")
    finally:
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# 11. SINGLE-RUN HELPER (batch mode)
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

    # Eps selection
    sweep_results = None
    if args.eps_search:
        print(f"  Sweeping eps [{args.eps_min:.2f} → {args.eps_max:.2f}, "
              f"step={args.eps_step:.2f}]...")
        sweep_results, eps_used, labels = eps_sweep(
            dist_matrix, true_labels,
            args.eps_min, args.eps_max, args.eps_step,
            args.min_samples,
        )
        print(f"  Best eps selected: {eps_used:.4f}")
    else:
        eps_used = args.eps
        _, labels = run_dbscan(dist_matrix, eps_used, args.min_samples)

    metrics   = compute_metrics(labels, dist_matrix, true_labels)
    n_true    = len(set(true_labels)) if true_labels is not None else None
    print_summary(metrics, eps_used, n_true=n_true, scenario=scenario)

    _build_figure(dataset, true_labels, file_labels,
                  dist_matrix, labels, metrics,
                  eps_used, sweep_results,
                  scenario, args)

    return metrics


# ═══════════════════════════════════════════════════════════════════
# 12. MAIN
# ═══════════════════════════════════════════════════════════════════
def main(args):
    print(f"\n{'='*54}")
    print(f"  DBSCAN CLUSTERING — Resource-Aware Honeynet")
    print(f"  Metric: DYNAMIT (Manhattan OUI + Jaccard SVC)")
    print(f"{'='*54}")

    file_labels = None

    if args.dir:
        if not os.path.isdir(args.dir):
            print(f"[ERROR] Direktori tidak ditemukan: {args.dir}")
            sys.exit(1)

        if args.batch:
            runs = load_from_dir_batch(args.dir)
            print(f"\n  Menjalankan {len(runs)} clustering run...")
            for dataset_b, _, scenario_b, _ in runs:
                args_b = copy.copy(args)
                args_b.dir = None
                _single_run(dataset_b, None, scenario_b, None, args_b)
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
# 13. CLI
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DBSCAN Clustering + Hybrid Metric (DYNAMIT)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh:
  # GNS3 honeynet — merge semua JSON, GT dari nama file, eps-search:
  python dbscan_clustering.py --dir output/ --eps-search --save-dir plots/

  # Satu file JSON, eps manual:
  python dbscan_clustering.py --json output/gns3-simulation-v3.json --eps 0.45 --save-dir plots/

  # Tiap JSON sebagai run terpisah + eps-search:
  python dbscan_clustering.py --dir output/ --batch --eps-search --save-dir plots/

  # Dataset sintetis:
  python dbscan_clustering.py --inline --num-host 5 --num-cluster 8 --eps-search

  # Eps sweep dengan rentang kustom:
  python dbscan_clustering.py --dir output/ --eps-search \\
      --eps-min 0.2 --eps-max 0.8 --eps-step 0.02 --save-dir plots/
        """
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--json',   type=str,
                     help='Path ke satu file JSON dari pcap_host_profiler.py')
    src.add_argument('--dir',    type=str,
                     help='Direktori berisi banyak .json\n'
                          '  Default: MERGE semua → 1 run, GT dari nama file\n'
                          '  Tambah --batch untuk run per file terpisah')
    src.add_argument('--inline', action='store_true',
                     help='Gunakan dataset sintetis bawaan')

    parser.add_argument('--batch', action='store_true',
                        help='[--dir] Jalankan tiap .json sebagai run terpisah')
    parser.add_argument('--label', type=str, default=None,
                        help='Nama skenario untuk judul grafik')
    parser.add_argument('--num-host',    type=int, default=5,
                        help='[--inline] Jumlah host per cluster (default: 5)')
    parser.add_argument('--num-cluster', type=int, default=8,
                        help='[--inline] Jumlah cluster (default: 8)')

    # DBSCAN eps params
    eps_grp = parser.add_mutually_exclusive_group()
    eps_grp.add_argument('--eps', type=float, default=0.5,
                         help='DBSCAN eps radius (default: 0.5). '
                              'Diabaikan jika --eps-search aktif.')
    eps_grp.add_argument('--eps-search', action='store_true',
                         help='Sweep eps otomatis, pilih terbaik berdasarkan\n'
                              'ARI (jika GT tersedia) atau Silhouette.')

    parser.add_argument('--eps-min',  type=float, default=0.1,
                        help='[--eps-search] Batas bawah eps (default: 0.1)')
    parser.add_argument('--eps-max',  type=float, default=1.0,
                        help='[--eps-search] Batas atas eps (default: 1.0)')
    parser.add_argument('--eps-step', type=float, default=0.05,
                        help='[--eps-search] Step eps sweep (default: 0.05)')
    parser.add_argument('--min-samples', type=int, default=1,
                        help='DBSCAN min_samples (default: 1)')

    parser.add_argument('--save-dir', type=str, default=None,
                        help='Direktori untuk menyimpan plot PNG (default: tampilkan)')

    args = parser.parse_args()
    main(args)
