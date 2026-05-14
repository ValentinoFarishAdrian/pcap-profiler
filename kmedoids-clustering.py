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

try:
    import kmedoids
except ImportError:
    print("[ERROR] kmedoids belum terinstall. Jalankan: pip install kmedoids")
    sys.exit(1)

from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
    pairwise_distances,
)
from sklearn.manifold import MDS
from scipy.optimize import linear_sum_assignment


def extract_features(dataset: list):
    df = pd.DataFrame(dataset)

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    oui_vec = ohe.fit_transform(df[['oui']])
    n_oui_features = oui_vec.shape[1]

    mlb = MultiLabelBinarizer()
    svc_vec = mlb.fit_transform(df['svc'])

    X = np.hstack([oui_vec, svc_vec])
    return X, n_oui_features, ohe, mlb


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


def run_kmedoids(dist_matrix: np.ndarray, k: int,
                 max_iter: int = 500, random_state: int = 42):
    result = kmedoids.fasterpam(
        dist_matrix,
        medoids=k,
        max_iter=max_iter,
        random_state=random_state,
    )
    return result


def _sil(dist_matrix, labels):
    try:
        return float(kmedoids.silhouette(dist_matrix, labels, False)[0])
    except Exception:
        return 0.0


def _msil(dist_matrix, labels):
    try:
        return float(kmedoids.medoid_silhouette(dist_matrix, labels, False)[0])
    except Exception:
        return 0.0


def k_sweep(dist_matrix: np.ndarray, k_min: int, k_max: int,
            max_iter: int = 500, random_state: int = 42):
    n     = len(dist_matrix)
    k_max = min(k_max, n - 1)
    iter_range = range(k_min, k_max + 1)

    result_loss  = []
    result_sill  = []
    result_msill = []

    optimal_cluster = 0
    prev_sill = 0.0
    prev_loss = 0.0

    for i in iter_range:
        res    = run_kmedoids(dist_matrix, i, max_iter, random_state)
        labels = np.array(res.labels)
        loss   = float(res.loss)
        sill   = _sil(dist_matrix, labels)
        msill  = _msil(dist_matrix, labels)

        result_loss.append(loss)
        result_sill.append(sill)
        result_msill.append(msill)

        if i == iter_range[0]:
            prev_sill = sill
            prev_loss = loss
            continue

        if (optimal_cluster == 0
                and sill - prev_sill < 0.03
                and prev_loss - loss <= 1.8):
            optimal_cluster = i - 1

        prev_sill = sill
        prev_loss = loss

    if optimal_cluster == 0:
        sill_arr = np.array(result_sill)
        if len(sill_arr) > 1:
            optimal_cluster = list(iter_range)[int(np.argmax(sill_arr[1:])) + 1]
        else:
            optimal_cluster = k_min

    res_best    = run_kmedoids(dist_matrix, optimal_cluster, max_iter, random_state)
    best_labels = np.array(res_best.labels)

    return (list(iter_range), result_loss, result_sill, result_msill,
            optimal_cluster, best_labels)


def compute_metrics(labels: np.ndarray, dist_matrix: np.ndarray,
                    true_labels=None):
    n_pred = len(set(labels))
    sill   = _sil(dist_matrix, labels)
    msill  = _msil(dist_matrix, labels)

    result = {
        'n_clusters_pred': n_pred,
        'sill':            sill,
        'msill':           msill,
        'ari':  np.nan,
        'nmi':  np.nan,
        'fmi':  np.nan,
    }

    if true_labels is not None and len(labels) > 0:
        result['ari'] = adjusted_rand_score(np.array(true_labels), labels)
        result['nmi'] = normalized_mutual_info_score(np.array(true_labels), labels)
        result['fmi'] = fowlkes_mallows_score(np.array(true_labels), labels)

    return result


def plot_confusion_matrix(true_labels, predicted_labels, ax,
                          scenario_name: str = "",
                          class_names: list = None):
    true_labels      = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    true_unique = np.unique(true_labels)
    pred_unique = np.unique(predicted_labels)
    n_true = len(true_unique)
    n_pred = len(pred_unique)

    cm       = np.zeros((n_true, n_pred), dtype=int)
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
        cbar_kws={'shrink': 0.7}, ax=ax
    )
    ax.set_xlabel("Predicted Cluster", fontsize=11, labelpad=8)
    ax.set_ylabel("True Cluster",      fontsize=11, labelpad=8)
    ax.set_title(f"Confusion Matrix — {scenario_name}",
                 fontsize=12, fontweight='bold', pad=12)
    ax.tick_params(axis='both', labelsize=9)


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
        yticklabels=y_labels, xticklabels=mlb.classes_,
        linewidths=0.3, linecolor='#f0f0f0',
        cbar_kws={'shrink': 0.6, 'label': 'Service Present'}, ax=ax
    )
    ax.set_title(f"Cluster Membership — {scenario_name}",
                 fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Service",        fontsize=10, labelpad=8)
    ax.set_ylabel("Host [Cluster]", fontsize=10, labelpad=8)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)


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
                sub = f" ~{dom[:16]}"
        ax.scatter(coords_j[mask, 0], coords_j[mask, 1],
                   color=color_map[c], s=80, alpha=0.85,
                   edgecolors='white', linewidths=0.8, zorder=3)
        ax.scatter([], [], color=color_map[c], s=80,
                   edgecolors='white', linewidths=0.8,
                   label=f"Cluster {c}{sub} (n={n_in_cluster})")

    if dataset and n <= 60:
        for i, d in enumerate(dataset):
            ip = d.get('ip', str(i))
            lbl_text = (f".{ip.split('.')[-1]}" if '.' in ip else f"H{i}")
            ax.annotate(lbl_text,
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


def plot_loss_curve(k_range: list, result_loss: list,
                    optimal_cluster: int, ax,
                    scenario_name: str = "", gt_k: int = None):
    ax.plot(k_range, result_loss, marker='o',
            label='Clustering Loss at k', color='#1565C0', lw=1.8)

    if gt_k is not None:
        ax.axvline(x=gt_k, color='red', linestyle='--',
                   label=f'True K = {gt_k}')

    if optimal_cluster > 0 and optimal_cluster in k_range:
        ax.axvline(x=optimal_cluster, color='#E91E63', linestyle=':',
                   lw=1.8, label=f'Optimal K = {optimal_cluster}')

    ax.set_xlabel('K (n_clusters)', fontsize=10)
    ax.set_ylabel('Loss',           fontsize=10)
    ax.set_title(f"Loss Curve (Elbow) — {scenario_name}",
                 fontsize=11, fontweight='bold', pad=10)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.85)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.tick_params(labelsize=9)


def plot_silhouette_curve(k_range: list,
                           result_sill: list, result_msill: list,
                           optimal_cluster: int, ax,
                           scenario_name: str = "", gt_k: int = None):
    k_plot    = k_range[1:]
    sil_plot  = result_sill[1:]
    msil_plot = result_msill[1:]

    if not k_plot:
        ax.text(0.5, 0.5, "Tidak cukup data untuk kurva silhouette",
                ha='center', va='center', transform=ax.transAxes)
        return

    ax.plot(k_plot, sil_plot, marker='^', color='green', lw=1.8,
            label='Silhouette at k')
    ax.plot(k_plot, msil_plot, marker='s', color='#FF6F00', lw=1.5,
            linestyle='--', label='Medoid Silhouette at k')

    if gt_k is not None:
        ax.axvline(x=gt_k, color='red', linestyle='--',
                   label=f'True K = {gt_k}')

    if optimal_cluster > 0 and optimal_cluster in k_range:
        ax.axvline(x=optimal_cluster, color='#E91E63', linestyle=':',
                   lw=1.8, label=f'Optimal K = {optimal_cluster}')

    ax.set_xlabel('K (n_clusters)', fontsize=10)
    ax.set_ylabel('Silhouette',     fontsize=10)
    ax.set_title(f"Silhouette Curve — {scenario_name}",
                 fontsize=11, fontweight='bold', pad=10)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.85)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.tick_params(labelsize=9)


def print_summary(metrics: dict, best_k: int, optimal_k: int,
                  best_loss: float, n_true: int = None, scenario: str = ""):
    w = 54
    print(f"\n{'='*w}")
    print(f"  K-MEDOIDS RESULT — {scenario}")
    print(f"{'='*w}")
    _kver = getattr(kmedoids, "__version__", "0.5.x")
    print(f"  Algorithm           : FasterPAM (kmedoids {_kver})")
    if n_true:
        print(f"  True Clusters (GT)  : {n_true}")
    print(f"  Optimal K (heuristik): {optimal_k}")
    print(f"  Predicted Clusters  : {metrics['n_clusters_pred']}")
    print(f"  Loss (at best K)    : {best_loss:.4f}")
    print(f"  {'─'*46}")
    print(f"  Silhouette          : {metrics['sill']:.4f}")
    print(f"  Medoid Silhouette   : {metrics['msill']:.4f}")
    print(f"  {'─'*46}")
    if not np.isnan(metrics['ari']):
        print(f"  ARI  (vs GT)        : {metrics['ari']:.4f}")
        print(f"  NMI  (Mutual Info)  : {metrics['nmi']:.4f}")
        print(f"  FMI  (Fowlkes-Mal.) : {metrics['fmi']:.4f}")
    else:
        print(f"  ARI / NMI / FMI     : N/A (no ground truth)")
    print(f"{'='*w}\n")


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
    files = [os.path.join(dirpath, e)
             for e in sorted(os.listdir(dirpath))
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


def scenario_name_short(s: str, max_len: int = 25) -> str:
    return s if len(s) <= max_len else s[:max_len] + '...'


def _build_figure(dataset, true_labels, file_labels,
                  dist_matrix, labels, metrics,
                  k_range, result_loss, result_sill, result_msill,
                  optimal_cluster, best_loss,
                  scenario, args):
    has_gt = true_labels is not None
    gt_k   = len(set(true_labels)) if has_gt else None

    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor('#fafafa')
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            hspace=0.48, wspace=0.35,
                            height_ratios=[1.1, 1.0])
    ax_tl = fig.add_subplot(gs[0, 0])
    ax_tr = fig.add_subplot(gs[0, 1])
    ax_bl = fig.add_subplot(gs[1, 0])
    ax_br = fig.add_subplot(gs[1, 1])

    if has_gt:
        plot_confusion_matrix(true_labels, labels, ax_tl,
                              scenario_name=scenario,
                              class_names=file_labels)
    else:
        plot_membership_heatmap(dataset, labels, ax_tl,
                                scenario_name=scenario)

    plot_scatter_mds(dist_matrix, labels, ax_tr,
                     dataset=dataset, true_labels=true_labels,
                     scenario_name=scenario, file_labels=file_labels)

    if k_range is not None:
        plot_loss_curve(k_range, result_loss, optimal_cluster,
                        ax_bl, scenario_name=scenario, gt_k=gt_k)
    else:
        n_k = metrics['n_clusters_pred']
        ax_bl.text(0.5, 0.5,
                   f"K tetap = {n_k}\n(K-sweep dinonaktifkan)\n"
                   f"Loss = {best_loss:.4f}",
                   ha='center', va='center', transform=ax_bl.transAxes,
                   fontsize=12, color='#555555',
                   bbox=dict(boxstyle='round', fc='#f0f4f8', ec='#cccccc'))
        ax_bl.set_title(f"Loss — {scenario_name_short(scenario)}",
                        fontsize=11, fontweight='bold', pad=10)

    if k_range is not None:
        plot_silhouette_curve(k_range, result_sill, result_msill,
                              optimal_cluster, ax_br,
                              scenario_name=scenario, gt_k=gt_k)
    else:
        ax_br.text(0.5, 0.5,
                   f"Silhouette = {metrics['sill']:.4f}\n"
                   f"Medoid Sil = {metrics['msill']:.4f}",
                   ha='center', va='center', transform=ax_br.transAxes,
                   fontsize=12, color='#555555',
                   bbox=dict(boxstyle='round', fc='#f0f4f8', ec='#cccccc'))
        ax_br.set_title(f"Silhouette — {scenario_name_short(scenario)}",
                        fontsize=11, fontweight='bold', pad=10)

    opt_str = f"Optimal K={optimal_cluster}" if k_range else f"K={metrics['n_clusters_pred']}"
    lines = [
        f"FasterPAM  |  {opt_str}  |  "
        f"Loss={best_loss:.4f}  |  Sil={metrics['sill']:.4f}  |  "
        f"MedSil={metrics['msill']:.4f}"
    ]
    if not np.isnan(metrics['ari']):
        lines.append(
            f"ARI={metrics['ari']:.4f}  NMI={metrics['nmi']:.4f}  "
            f"FMI={metrics['fmi']:.4f}"
        )
    else:
        lines.append("ARI / NMI / FMI : N/A (no ground truth)")

    fig.text(0.5, 0.005, '\n'.join(lines),
             ha='center', va='bottom', fontsize=10, color='#444444',
             bbox=dict(boxstyle='round,pad=0.4', fc='#f0f4f8',
                       ec='#cccccc', lw=0.8))

    plt.suptitle(f"K-Medoids (FasterPAM) Clustering — {scenario}",
                 fontsize=14, fontweight='bold', y=1.01, color='#222222')

    try:
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            safe = (scenario
                    .replace(' ', '_').replace('/', '-')
                    .replace(':', '-').replace('(', '').replace(')', '')
                    .replace(',', '')[:80])
            out_path = os.path.join(args.save_dir, f"kmedoids_{safe}.png")
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


def _single_run(dataset, true_labels, scenario, file_labels, args):
    print(f"\n  >> Run: {scenario}  ({len(dataset)} hosts)")
    if len(dataset) < 2:
        print(f"  [SKIP] Terlalu sedikit host ({len(dataset)}), min=2")
        return

    X, n_oui, ohe, mlb = extract_features(dataset)
    hybrid_metric       = create_dynamit_metric(n_oui)

    print("  Computing hybrid distance matrix...")
    dist_matrix = pairwise_distances(X, metric=hybrid_metric)

    if args.k is not None:
        print(f"  K tetap = {args.k}")
        res             = run_kmedoids(dist_matrix, args.k, args.max_iter,
                                       args.random_state)
        labels          = np.array(res.labels)
        best_loss       = float(res.loss)
        optimal_cluster = args.k
        k_range = result_loss = result_sill = result_msill = None

    else:
        k_max = args.k_max if args.k_max else min(len(dataset) - 1, 30)
        print(f"  Sweeping K [{args.k_min} -> {k_max}]  "
              f"(max_iter={args.max_iter}, random_state={args.random_state})...")

        (k_range, result_loss, result_sill, result_msill,
         optimal_cluster, labels) = k_sweep(
            dist_matrix, args.k_min, k_max,
            args.max_iter, args.random_state
        )

        opt_idx   = list(k_range).index(optimal_cluster)
        best_loss = result_loss[opt_idx]
        print(f"  Optimal K (heuristik): {optimal_cluster}  "
              f"(loss={best_loss:.4f})")

    metrics = compute_metrics(labels, dist_matrix, true_labels)
    metrics['loss'] = best_loss

    n_true = len(set(true_labels)) if true_labels is not None else None
    print_summary(metrics, optimal_cluster, optimal_cluster,
                  best_loss, n_true=n_true, scenario=scenario)

    _build_figure(dataset, true_labels, file_labels,
                  dist_matrix, labels, metrics,
                  k_range, result_loss, result_sill, result_msill,
                  optimal_cluster, best_loss,
                  scenario, args)

    return metrics


def main(args):
    print(f"\n{'='*54}")
    print(f"  K-MEDOIDS CLUSTERING — Resource-Aware Honeynet")
    _kver = getattr(kmedoids, "__version__", "0.5.x")
    print(f"  Algorithm: FasterPAM  (kmedoids {_kver})")
    print(f"  Metric   : DYNAMIT (Manhattan OUI + Jaccard SVC)")
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='K-Medoids Clustering (FasterPAM) + Hybrid Metric (DYNAMIT)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh:
  # GNS3 honeynet — K-sweep otomatis (default):
  python kmedoids_clustering.py --dir output/ --save-dir plots/

  # Satu JSON, K-sweep:
  python kmedoids_clustering.py --json output/gns3-simulation-v3.json --save-dir plots/

  # K tetap (skip sweep):
  python kmedoids_clustering.py --json output/gns3-simulation-v3.json --k 5 --save-dir plots/

  # Dataset sintetis (replikasi notebook 5H x 8C):
  python kmedoids_clustering.py --inline --num-host 5 --num-cluster 8

  # Rentang K kustom + parameter fasterpam:
  python kmedoids_clustering.py --dir output/ \\
      --k-min 2 --k-max 15 --max-iter 500 --random-state 11 --save-dir plots/

  # Tiap JSON sebagai run terpisah:
  python kmedoids_clustering.py --dir output/ --batch --save-dir plots/
        """
    )

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
    parser.add_argument('--label',       type=str, default=None,
                        help='Override nama skenario pada judul grafik')
    parser.add_argument('--num-host',    type=int, default=5,
                        help='[--inline] Jumlah host per cluster (default: 5)')
    parser.add_argument('--num-cluster', type=int, default=8,
                        help='[--inline] Jumlah cluster (default: 8)')

    parser.add_argument('--k',           type=int, default=None,
                        help='K tetap — skip K-sweep. Jika tidak diset: sweep otomatis.')
    parser.add_argument('--k-min',       type=int, default=1,
                        help='[K-sweep] Batas bawah K (default: 1)')
    parser.add_argument('--k-max',       type=int, default=None,
                        help='[K-sweep] Batas atas K (default: min(n_hosts-1, 30))')
    parser.add_argument('--max-iter',    type=int, default=500,
                        help='FasterPAM max_iter (default: 500)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='FasterPAM random_state (default: 42)')
    parser.add_argument('--save-dir',    type=str, default=None,
                        help='Direktori simpan plot PNG (default: tampilkan)')

    args = parser.parse_args()
    main(args)
