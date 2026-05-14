#!/usr/bin/env python3

import argparse
import json
import sys
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
    pairwise_distances,
)
from sklearn.manifold import MDS
from scipy.optimize import linear_sum_assignment

try:
    import hdbscan
except ImportError:
    print("[ERROR] hdbscan belum terinstall. Jalankan: pip install hdbscan")
    sys.exit(1)


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


def run_hdbscan(X: np.ndarray, hybrid_metric,
                min_cluster_size: int = 2,
                min_samples: int = 1,
                cluster_selection_epsilon: float = 0.6):
    clusterer = hdbscan.HDBSCAN(
        metric=hybrid_metric,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        allow_single_cluster=True,
        gen_min_span_tree=True,
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(X)
    return clusterer, labels


def compute_metrics(labels: np.ndarray, clusterer, true_labels=None):
    unique    = set(labels)
    n_pred    = len(unique) - (1 if -1 in unique else 0)
    n_noise   = int(np.sum(labels == -1))
    noise_pct = n_noise / len(labels) * 100

    result = {
        'n_clusters_pred': n_pred,
        'n_noise':         n_noise,
        'noise_pct':       noise_pct,
        'ari':  np.nan,
        'nmi':  np.nan,
        'fmi':  np.nan,
        'dbcv': np.nan,
    }

    if hasattr(clusterer, 'relative_validity_'):
        v = clusterer.relative_validity_
        if v is not None and not np.isnan(float(v)):
            result['dbcv'] = float(v)

    if true_labels is not None:
        mask = labels != -1
        if mask.sum() > 0:
            tl = np.array(true_labels)[mask]
            pl = labels[mask]
            result['ari'] = adjusted_rand_score(tl, pl)
            result['nmi'] = normalized_mutual_info_score(tl, pl)
            result['fmi'] = fowlkes_mallows_score(tl, pl)

    return result


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
    ax.set_title(f"Confusion Matrix — {scenario_name}", fontsize=12,
                 fontweight='bold', pad=12)
    ax.tick_params(axis='both', labelsize=9)


def plot_membership_heatmap(dataset: list, labels: np.ndarray, ax,
                             scenario_name: str = ""):
    df = pd.DataFrame(dataset).copy()
    df['cluster']   = labels
    df['label_str'] = df['cluster'].apply(
        lambda c: "Noise" if c == -1 else f"Cluster {c}"
    )
    df['ip_short'] = df.get('ip', pd.Series([f"Host {i}" for i in range(len(df))],
                                             index=df.index))

    mlb = MultiLabelBinarizer()
    svc_mat = mlb.fit_transform(df['svc'])
    svc_df  = pd.DataFrame(svc_mat, columns=mlb.classes_)

    order        = df.sort_values('cluster').index
    svc_sorted   = svc_df.loc[order]
    label_sorted = df.loc[order, 'label_str'].values
    ip_sorted    = (df.loc[order, 'ip_short'].values if 'ip' in df.columns
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
    ax.set_title(f"Cluster Membership — {scenario_name}", fontsize=12,
                 fontweight='bold', pad=12)
    ax.set_xlabel("Service",        fontsize=10, labelpad=8)
    ax.set_ylabel("Host [Cluster]", fontsize=10, labelpad=8)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)


def plot_scatter_mds(dist_matrix: np.ndarray, labels: np.ndarray,
                     ax, dataset: list = None,
                     true_labels=None, scenario_name: str = "",
                     file_labels: list = None):
    n = len(labels)

    n_valid = int((labels != -1).sum())
    if n_valid < 3:
        ax.text(0.5, 0.5,
                f"Tidak cukup data untuk MDS\n"
                f"({n_valid} titik non-noise, min=3)\n"
                f"Tambah host atau turunkan --min-pkts",
                ha='center', va='center', transform=ax.transAxes,
                fontsize=11, color='#888888',
                bbox=dict(boxstyle='round', fc='#f5f5f5', ec='#cccccc'))
        ax.set_title(f"Scatter Plot — {scenario_name}", fontsize=11,
                     fontweight='bold', pad=10)
        return

    mds    = MDS(n_components=2, dissimilarity='precomputed',
                 random_state=42, normalized_stress='auto')
    coords = mds.fit_transform(dist_matrix)

    # Jitter kecil agar host dengan profil identik (jarak=0) tidak menumpuk di satu titik
    rng          = np.random.RandomState(42)
    coord_range  = coords.ptp(axis=0)
    jitter_scale = coord_range.mean() * 0.025
    coords_jittered = coords + rng.randn(n, 2) * jitter_scale

    unique_clusters = sorted(set(labels) - {-1})
    n_clusters      = len(unique_clusters)
    cmap_name       = 'tab10' if n_clusters <= 10 else 'tab20'
    palette         = matplotlib.colormaps[cmap_name]
    color_map       = {c: palette(i) for i, c in enumerate(unique_clusters)}

    for c in unique_clusters:
        mask         = labels == c
        n_in_cluster = int(mask.sum())

        cluster_sources = []
        if dataset and file_labels:
            src_counts = {}
            for i, d in enumerate(dataset):
                if labels[i] == c:
                    s = d.get('source', '')
                    src_counts[s] = src_counts.get(s, 0) + 1
            if src_counts:
                dom = max(src_counts, key=src_counts.get)
                cluster_sources = [f"≈{dom[:18]}"]
        sub = f" {cluster_sources[0]}" if cluster_sources else ""

        ax.scatter(
            coords_jittered[mask, 0], coords_jittered[mask, 1],
            color=color_map[c], s=80, alpha=0.85,
            edgecolors='white', linewidths=0.8, zorder=3
        )
        ax.scatter([], [],
            color=color_map[c], s=80,
            edgecolors='white', linewidths=0.8,
            label=f"Cluster {c}{sub} (n={n_in_cluster})"
        )

    noise_mask = labels == -1
    if noise_mask.any():
        ax.scatter(
            coords_jittered[noise_mask, 0], coords_jittered[noise_mask, 1],
            color='#aaaaaa', marker='X', s=80, alpha=0.7,
            label=f"Noise (n={noise_mask.sum()})",
            zorder=2
        )

    if dataset and n <= 60:
        for i, d in enumerate(dataset):
            ip = d.get('ip', str(i))
            if ip:
                short_ip   = ip.split('.')[-1] if '.' in ip else str(i)
                label_text = f".{short_ip}"
            else:
                label_text = f"H{i}"
            ax.annotate(
                label_text,
                (coords_jittered[i, 0], coords_jittered[i, 1]),
                fontsize=7, alpha=0.75, fontweight='500',
                color=color_map.get(labels[i], '#888888'),
                xytext=(3, 3), textcoords='offset points'
            )

    stress_norm = mds.stress_ / (np.sum(dist_matrix**2) / 2) if mds.stress_ else 0

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


def print_summary(metrics: dict, n_true: int = None, scenario: str = ""):
    w = 54
    print(f"\n{'='*w}")
    print(f"  HDBSCAN RESULT — {scenario}")
    print(f"{'='*w}")
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
    if not np.isnan(metrics['dbcv']):
        print(f"  DBCV (Density Validity) : {metrics['dbcv']:.4f}")
    else:
        print(f"  DBCV                    : N/A")
    print(f"{'='*w}\n")


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

    oui_base = [f"{i:02x}:{(i*7)%256:02x}:{(i*13)%256:02x}" for i in range(1, num_cluster+1)]

    dataset     = []
    true_labels = []
    for c in range(num_cluster):
        base_svc = SERVICES_POOL[c % len(SERVICES_POOL)]
        base_oui = oui_base[c]
        for h in range(num_host):
            svc = base_svc.copy()
            extra = random.choice(['HTTP', 'DNS', 'NTP', 'SSH', 'FTP'])
            if extra not in svc:
                svc.append(extra)
            if len(svc) > 2 and random.random() > 0.5:
                svc.pop(random.randrange(len(svc)))
            oui = base_oui if random.random() > 0.3 else \
                  f"{(c+10)%256:02x}:{(h+5)%256:02x}:{c:02x}"
            dataset.append({'oui': oui, 'svc': sorted(set(svc))})
            true_labels.append(c)

    scenario = f"Synthetic {num_host}H × {num_cluster}C"
    return dataset, true_labels, scenario


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
    dataset     = []
    true_labels = []
    file_labels = []

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

    scenario = f"DIR-MERGE: {os.path.basename(dirpath)} ({len(files)} files, {len(dataset)} hosts)"
    return dataset, true_labels, scenario, file_labels


def load_from_dir_batch(dirpath: str) -> list:
    files = collect_json_files(dirpath)
    runs  = []
    print(f"  [dir-batch] {len(files)} file JSON → {len(files)} clustering run:")
    for fpath in files:
        try:
            dataset, _, scenario = load_from_json(fpath)
            runs.append((dataset, None, _stem(fpath), fpath))
        except Exception as e:
            print(f"    [SKIP] {os.path.basename(fpath)}: {e}")
    return runs


def _batch_run(dataset, true_labels, scenario, args):
    print(f"\n  → Run: {scenario}  ({len(dataset)} hosts)")
    if len(dataset) < 2:
        print(f"  [SKIP] Terlalu sedikit host ({len(dataset)}), min=2")
        return

    X, n_oui, ohe, mlb = extract_features(dataset)
    hybrid_metric       = create_dynamit_metric(n_oui)
    dist_matrix         = pairwise_distances(X, metric=hybrid_metric)

    clusterer, labels = run_hdbscan(
        X, hybrid_metric,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_epsilon=args.epsilon,
    )
    metrics = compute_metrics(labels, clusterer, true_labels)
    print_summary(metrics, n_true=None, scenario=scenario)

    fig = plt.figure(figsize=(16, 7))
    fig.patch.set_facecolor('#fafafa')
    gs       = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
    ax_left  = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    plot_membership_heatmap(dataset, labels, ax_left, scenario_name=scenario)
    plot_scatter_mds(dist_matrix, labels, ax_right, dataset=dataset,
                     true_labels=None, scenario_name=scenario)

    lines = [
        f"Clusters Found: {metrics['n_clusters_pred']}  |  Noise: {metrics['n_noise']} ({metrics['noise_pct']:.1f}%)",
        f"DBCV={metrics['dbcv']:.4f}  (no ground truth)"
    ]
    fig.text(0.5, 0.01, '\n'.join(lines), ha='center', va='bottom',
             fontsize=10, color='#444444',
             bbox=dict(boxstyle='round,pad=0.4', fc='#f0f4f8', ec='#cccccc', lw=0.8))
    plt.suptitle(f"HDBSCAN — {scenario}", fontsize=13, fontweight='bold',
                 y=1.01, color='#222222')
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        safe = scenario.replace(' ', '_').replace('/', '-').replace('×', 'x')
        out  = os.path.join(args.save_dir, f"hdbscan_batch_{safe}.png")
        plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#fafafa')
        if os.path.isfile(out):
            print(f"  [+] Plot: {out}  ({os.path.getsize(out)/1024:.1f} KB)")
        else:
            print(f"  [!] File tidak terbuat: {out}")
    else:
        plt.show()
    plt.close(fig)


def main(args):
    print(f"\n{'='*54}")
    print(f"  HDBSCAN CLUSTERING — Resource-Aware Honeynet")
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
            import copy
            for dataset_b, _, scenario_b, _ in runs:
                args_b        = copy.copy(args)
                args_b.dir    = None
                args_b.json   = '__batch__'
                args_b.inline = False
                args_b.label  = scenario_b
                _batch_run(dataset_b, None, scenario_b, args_b)
            return
        else:
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
        print(f"        Tambah lebih banyak PCAP atau turunkan --min-pkts")
        return None
    if true_labels:
        n_true = len(set(true_labels))
        print(f"  GT Clust : {n_true}")

    X, n_oui, ohe, mlb = extract_features(dataset)
    hybrid_metric = create_dynamit_metric(n_oui)
    print(f"  Features : {X.shape[1]} dims ({n_oui} OUI + {X.shape[1]-n_oui} SVC)")

    print("  Computing hybrid distance matrix...")
    dist_matrix = pairwise_distances(X, metric=hybrid_metric)

    print(f"  Running HDBSCAN (min_cluster_size={args.min_cluster_size}, "
          f"min_samples={args.min_samples}, eps={args.epsilon})...")
    clusterer, labels = run_hdbscan(
        X, hybrid_metric,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_epsilon=args.epsilon,
    )

    metrics    = compute_metrics(labels, clusterer, true_labels)
    n_true_val = len(set(true_labels)) if true_labels else None
    print_summary(metrics, n_true=n_true_val, scenario=scenario)

    fig = plt.figure(figsize=(16, 7))
    fig.patch.set_facecolor('#fafafa')
    gs       = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
    ax_left  = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    if true_labels is not None:
        plot_confusion_matrix(
            true_labels, labels, ax_left,
            scenario_name=scenario,
            class_names=file_labels,
        )
    else:
        plot_membership_heatmap(
            dataset, labels, ax_left, scenario_name=scenario
        )

    plot_scatter_mds(
        dist_matrix, labels, ax_right,
        dataset=dataset, true_labels=true_labels,
        scenario_name=scenario, file_labels=file_labels,
    )

    lines = [
        f"Clusters Found: {metrics['n_clusters_pred']}  |  "
        f"Noise: {metrics['n_noise']} ({metrics['noise_pct']:.1f}%)"
    ]
    if not np.isnan(metrics['ari']):
        lines.append(
            f"ARI={metrics['ari']:.4f}  NMI={metrics['nmi']:.4f}  "
            f"FMI={metrics['fmi']:.4f}  DBCV={metrics['dbcv']:.4f}"
        )
    else:
        lines.append(
            f"DBCV={metrics['dbcv']:.4f}  "
            f"(No ground truth — ARI/NMI/FMI not available)"
        )

    fig.text(
        0.5, 0.01, '\n'.join(lines),
        ha='center', va='bottom', fontsize=10,
        color='#444444',
        bbox=dict(boxstyle='round,pad=0.4', fc='#f0f4f8', ec='#cccccc', lw=0.8)
    )
    plt.suptitle(
        f"HDBSCAN Clustering — {scenario}",
        fontsize=14, fontweight='bold', y=1.01, color='#222222'
    )

    try:
        plt.tight_layout(rect=[0, 0.06, 1, 1])
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            safe_name = (scenario.replace(' ', '_').replace('/', '-')
                         .replace('×', 'x').replace(':', '-')
                         .replace('(', '').replace(')', '')
                         .replace(',', '')[:80])
            out_path = os.path.join(args.save_dir, f"hdbscan_{safe_name}.png")
            plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#fafafa')
            if os.path.isfile(out_path):
                size_kb = os.path.getsize(out_path) / 1024
                print(f"[+] Plot disimpan: {out_path}  ({size_kb:.1f} KB)")
            else:
                print(f"[!] savefig dipanggil tapi file tidak ditemukan: {out_path}")
        else:
            plt.show()
    except Exception as e:
        print(f"[!] Gagal menyimpan plot: {e}")
    finally:
        plt.close(fig)

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='HDBSCAN Clustering + Hybrid Metric (DYNAMIT)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh:
  # Data nyata dari pcap-profiler:
  python hdbscan_clustering.py --json philips_hue_bridge_profile.json

  # Dengan label kustom dan simpan grafik:
  python hdbscan_clustering.py --json hasil.json \\
      --label "CTU-Honeypot-7-1 Somfy" --save-dir ./plots

  # Dataset sintetis (untuk testing / replikasi eksperimen):
  python hdbscan_clustering.py --inline --num-host 5 --num-cluster 8

  # Semua .json dalam direktori → 1 clustering gabungan:
  python hdbscan_clustering.py --dir ./output/ --save-dir ./plots

  # Setiap file diproses terpisah (N plot independen):
  python hdbscan_clustering.py --dir ./output/ --batch --save-dir ./plots

  # Tune parameter HDBSCAN:
  python hdbscan_clustering.py --json hasil.json \\
      --min-cluster-size 3 --min-samples 2 --epsilon 0.5
        """
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--json',   type=str,
                     help='Path ke satu file JSON dari pcap_host_profiler.py')
    src.add_argument('--dir',    type=str,
                     help='Path direktori berisi banyak file .json\n'
                          '  Default: mode MERGE (gabung semua, GT dari nama file)\n'
                          '  Tambah --batch untuk jalankan tiap file terpisah')
    src.add_argument('--inline', action='store_true',
                     help='Gunakan dataset sintetis bawaan (untuk testing)')

    parser.add_argument('--batch', action='store_true',
                        help='[--dir] Jalankan tiap .json sebagai run terpisah')
    parser.add_argument('--label',       type=str, default=None,
                        help='Nama skenario untuk judul grafik')
    parser.add_argument('--num-host',    type=int, default=5,
                        help='[--inline] Jumlah host per cluster (default: 5)')
    parser.add_argument('--num-cluster', type=int, default=8,
                        help='[--inline] Jumlah cluster (default: 8)')

    parser.add_argument('--min-cluster-size', type=int, default=2,
                        help='HDBSCAN min_cluster_size (default: 2)')
    parser.add_argument('--min-samples',      type=int, default=1,
                        help='HDBSCAN min_samples (default: 1)')
    parser.add_argument('--epsilon',          type=float, default=0.6,
                        help='HDBSCAN cluster_selection_epsilon (default: 0.6)')
    parser.add_argument('--save-dir',         type=str, default=None,
                        help='Direktori untuk menyimpan plot PNG (default: tampilkan)')

    args = parser.parse_args()
    main(args)
