#!/usr/bin/env python3
"""
Download and preprocess REAL datasets for Network IDS project.

Datasets:
  1. KDD Cup 99 (10% subset ~5MB, full ~18MB compressed)
  2. CICIDS2017 (~6.4GB total, downloads per-day CSVs)

Usage:
    python download_real_data.py kdd          # KDD Cup 99 only
    python download_real_data.py cicids        # CICIDS2017 only  
    python download_real_data.py all           # Both datasets
    python download_real_data.py kdd --full    # KDD full dataset (4.9M records)

Output:
    data/kdd/train_data.csv, train_labels.csv, test_data.csv, test_labels.csv
    data/cicids/train_data.csv, train_labels.csv, test_data.csv, test_labels.csv
"""

import numpy as np
import pandas as pd
import os
import sys
import time
import urllib.request
import gzip
import shutil
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ===================== KDD CUP 99 =====================
KDD_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label'
]

# Map attack names to 5 categories
ATTACK_MAP = {
    'normal.': 0,
    # DoS
    'back.': 1, 'land.': 1, 'neptune.': 1, 'pod.': 1, 'smurf.': 1,
    'teardrop.': 1, 'mailbomb.': 1, 'apache2.': 1, 'processtable.': 1,
    'udpstorm.': 1,
    # Probe
    'satan.': 2, 'ipsweep.': 2, 'nmap.': 2, 'portsweep.': 2,
    'mscan.': 2, 'saint.': 2,
    # R2L
    'guess_passwd.': 3, 'ftp_write.': 3, 'imap.': 3, 'phf.': 3,
    'multihop.': 3, 'warezmaster.': 3, 'warezclient.': 3, 'spy.': 3,
    'xlock.': 3, 'xsnoop.': 3, 'snmpguess.': 3, 'snmpgetattack.': 3,
    'httptunnel.': 3, 'sendmail.': 3, 'named.': 3, 'worm.': 3,
    # U2R
    'buffer_overflow.': 4, 'loadmodule.': 4, 'rootkit.': 4, 'perl.': 4,
    'sqlattack.': 4, 'xterm.': 4, 'ps.': 4, 'httptunnel.': 4,
}

CLASS_NAMES = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']


def download_file(url, filepath):
    """Download file with progress bar."""
    if os.path.exists(filepath):
        print(f"  Already exists: {filepath}")
        return
    print(f"  Downloading: {url}")
    print(f"  Saving to: {filepath}")
    
    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / 1e6
            total_mb = total_size / 1e6
            print(f"\r  Progress: {pct}% ({mb:.1f}/{total_mb:.1f} MB)", end="", flush=True)
    
    urllib.request.urlretrieve(url, filepath, reporthook=progress)
    print()


def download_kdd(use_full=False):
    """Download and preprocess KDD Cup 99 dataset."""
    print("\n" + "="*60)
    print("  KDD Cup 99 Dataset")
    print("="*60)
    
    os.makedirs("data/kdd", exist_ok=True)
    
    if use_full:
        url = "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz"
        gz_file = "data/kdd/kddcup.data.gz"
        csv_file = "data/kdd/kddcup.data.csv"
        print("  Mode: FULL dataset (~4.9M records)")
    else:
        url = "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz"
        gz_file = "data/kdd/kddcup.data_10_percent.gz"
        csv_file = "data/kdd/kddcup.data_10_percent.csv"
        print("  Mode: 10% subset (~494K records)")
    
    # Download
    download_file(url, gz_file)
    
    # Extract gz
    if not os.path.exists(csv_file):
        print("  Extracting gzip...")
        with gzip.open(gz_file, 'rb') as f_in:
            with open(csv_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    
    # Load and process
    print("  Loading CSV...")
    t_start = time.time()
    df = pd.read_csv(csv_file, header=None, names=KDD_COLUMNS)
    print(f"  Loaded {len(df)} records in {time.time()-t_start:.1f}s")
    
    # Map labels to 5 categories
    print("  Mapping attack labels to 5 categories...")
    df['category'] = df['label'].map(ATTACK_MAP)
    # Unknown attacks -> assign to closest category (U2R as default for unknown)
    df['category'] = df['category'].fillna(4).astype(int)
    
    # Print class distribution
    print("\n  Class Distribution:")
    for i, name in enumerate(CLASS_NAMES):
        count = (df['category'] == i).sum()
        pct = 100 * count / len(df)
        print(f"    {name:8s}: {count:>8,} ({pct:5.1f}%)")
    
    # Encode categorical features
    print("\n  Encoding categorical features...")
    cat_cols = ['protocol_type', 'service', 'flag']
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
    
    # Select features (drop label columns)
    feature_cols = [c for c in df.columns if c not in ['label', 'category']]
    X = df[feature_cols].values.astype(np.float64)
    y = df['category'].values.astype(int)
    
    # Normalize
    print("  Min-Max normalizing...")
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    # Remove constant features
    var = X.var(axis=0)
    non_const = var > 1e-10
    X = X[:, non_const]
    print(f"  Features after removing constants: {X.shape[1]}")
    
    # Split 70/30
    print("  Splitting 70/30 train/test...")
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split = int(len(X) * 0.7)
    
    train_X = X[indices[:split]]
    train_y = y[indices[:split]]
    test_X = X[indices[split:]]
    test_y = y[indices[split:]]
    
    # Save
    print("  Saving processed data...")
    np.savetxt("data/kdd/train_data.csv", train_X, delimiter=",", fmt="%.8f")
    np.savetxt("data/kdd/train_labels.csv", train_y, delimiter=",", fmt="%d")
    np.savetxt("data/kdd/test_data.csv", test_X, delimiter=",", fmt="%.8f")
    np.savetxt("data/kdd/test_labels.csv", test_y, delimiter=",", fmt="%d")
    
    print(f"\n  Done! Train: {len(train_X)} | Test: {len(test_X)} | Features: {X.shape[1]}")
    print(f"  Files saved to data/kdd/")
    
    # Also copy to data/ root for default usage
    for f in ['train_data.csv', 'train_labels.csv', 'test_data.csv', 'test_labels.csv']:
        shutil.copy(f"data/kdd/{f}", f"data/{f}")
    print("  Also copied to data/ (default location)")
    
    return train_X, train_y, test_X, test_y


def download_cicids():
    """Download and preprocess CICIDS2017 dataset."""
    print("\n" + "="*60)
    print("  CICIDS2017 Dataset")
    print("="*60)
    
    os.makedirs("data/cicids", exist_ok=True)
    
    # CICIDS2017 CSV files from UNB (hosted on various mirrors)
    # Using the CIC mirror / Kaggle-compatible format
    CICIDS_URLS = {
        "Monday": "https://iscxdownloads.cs.unb.ca/iscxdownloads/CIC-IDS-2017/GeneratedLabelledFlows/TrafficLabelling/Monday-WorkingHours.pcap_ISCX.csv",
        "Tuesday": "https://iscxdownloads.cs.unb.ca/iscxdownloads/CIC-IDS-2017/GeneratedLabelledFlows/TrafficLabelling/Tuesday-WorkingHours.pcap_ISCX.csv",
        "Wednesday": "https://iscxdownloads.cs.unb.ca/iscxdownloads/CIC-IDS-2017/GeneratedLabelledFlows/TrafficLabelling/Wednesday-workingHours.pcap_ISCX.csv",
        "Thursday-Morning": "https://iscxdownloads.cs.unb.ca/iscxdownloads/CIC-IDS-2017/GeneratedLabelledFlows/TrafficLabelling/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "Thursday-Afternoon": "https://iscxdownloads.cs.unb.ca/iscxdownloads/CIC-IDS-2017/GeneratedLabelledFlows/TrafficLabelling/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "Friday-Morning": "https://iscxdownloads.cs.unb.ca/iscxdownloads/CIC-IDS-2017/GeneratedLabelledFlows/TrafficLabelling/Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "Friday-Afternoon-PortScan": "https://iscxdownloads.cs.unb.ca/iscxdownloads/CIC-IDS-2017/GeneratedLabelledFlows/TrafficLabelling/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Friday-Afternoon-DDos": "https://iscxdownloads.cs.unb.ca/iscxdownloads/CIC-IDS-2017/GeneratedLabelledFlows/TrafficLabelling/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    }
    
    print(f"  CICIDS2017 has {len(CICIDS_URLS)} CSV files (~6.4GB total)")
    print("  NOTE: Download may take 10-30 minutes depending on connection")
    print("  TIP: If download fails, you can manually download from:")
    print("       https://www.unb.ca/cic/datasets/ids-2017.html")
    print("       or Kaggle: https://www.kaggle.com/datasets/cicdataset/cicids2017")
    print()
    
    all_dfs = []
    
    for day_name, url in CICIDS_URLS.items():
        csv_file = f"data/cicids/{day_name}.csv"
        
        try:
            download_file(url, csv_file)
            
            print(f"  Loading {day_name}...")
            # CICIDS has some encoding issues, handle them
            df = pd.read_csv(csv_file, encoding='utf-8', low_memory=False)
            
            # Clean column names (strip whitespace)
            df.columns = df.columns.str.strip()
            
            print(f"    {len(df)} records loaded")
            all_dfs.append(df)
            
        except Exception as e:
            print(f"  WARNING: Failed to download {day_name}: {e}")
            print(f"  Skipping this file. Download manually if needed.")
            continue
    
    if not all_dfs:
        print("\n  ERROR: No CICIDS files could be downloaded.")
        print("  Please download manually from:")
        print("  https://www.kaggle.com/datasets/cicdataset/cicids2017")
        print("  Place CSV files in data/cicids/ and run again.")
        return None, None, None, None
    
    # Combine all days
    print(f"\n  Combining {len(all_dfs)} files...")
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"  Total records: {len(df)}")
    
    # Clean data
    print("  Cleaning data...")
    
    # Get label column (might be ' Label' or 'Label')
    label_col = None
    for col in df.columns:
        if 'label' in col.lower():
            label_col = col
            break
    
    if label_col is None:
        print("  ERROR: Cannot find label column")
        return None, None, None, None
    
    # Map CICIDS labels to numeric categories
    # 0=Benign, 1-N=Attack types
    df[label_col] = df[label_col].str.strip()
    unique_labels = df[label_col].unique()
    print(f"  Unique labels: {len(unique_labels)}")
    for lbl in sorted(unique_labels):
        count = (df[label_col] == lbl).sum()
        print(f"    {lbl}: {count:,}")
    
    # Encode labels
    le = LabelEncoder()
    df['category'] = le.fit_transform(df[label_col])
    
    # Select numeric features only
    feature_cols = []
    for col in df.columns:
        if col not in [label_col, 'category']:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                    feature_cols.append(col)
            except:
                pass
    
    print(f"  Numeric features: {len(feature_cols)}")
    
    # Drop rows with NaN/Inf
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols)
    print(f"  Records after cleaning: {len(df)}")
    
    X = df[feature_cols].values.astype(np.float64)
    y = df['category'].values.astype(int)
    
    # Normalize
    print("  Min-Max normalizing...")
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    # Remove constant features
    var = X.var(axis=0)
    non_const = var > 1e-10
    X = X[:, non_const]
    print(f"  Features after removing constants: {X.shape[1]}")
    
    # Split 70/30
    print("  Splitting 70/30 train/test...")
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split = int(len(X) * 0.7)
    
    train_X = X[indices[:split]]
    train_y = y[indices[:split]]
    test_X = X[indices[split:]]
    test_y = y[indices[split:]]
    
    # Save
    print("  Saving processed data...")
    np.savetxt("data/cicids/train_data.csv", train_X, delimiter=",", fmt="%.8f")
    np.savetxt("data/cicids/train_labels.csv", train_y, delimiter=",", fmt="%d")
    np.savetxt("data/cicids/test_data.csv", test_X, delimiter=",", fmt="%.8f")
    np.savetxt("data/cicids/test_labels.csv", test_y, delimiter=",", fmt="%d")
    
    print(f"\n  Done! Train: {len(train_X)} | Test: {len(test_X)} | Features: {X.shape[1]}")
    print(f"  Files saved to data/cicids/")
    
    # Save label mapping
    with open("data/cicids/label_mapping.txt", "w") as f:
        for i, name in enumerate(le.classes_):
            f.write(f"{i}: {name}\n")
    
    return train_X, train_y, test_X, test_y


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "all"
    use_full = "--full" in sys.argv
    
    print("="*60)
    print("  HPCE Network IDS - Real Dataset Downloader")
    print("="*60)
    
    if dataset in ["kdd", "all"]:
        download_kdd(use_full)
    
    if dataset in ["cicids", "all"]:
        download_cicids()
    
    if dataset not in ["kdd", "cicids", "all"]:
        print(f"Unknown dataset: {dataset}")
        print("Usage: python download_real_data.py [kdd|cicids|all] [--full]")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("  All done!")
    print("  To use KDD:    cp data/kdd/* data/")
    print("  To use CICIDS: cp data/cicids/* data/")
    print("  Then run: make run")
    print("="*60)


if __name__ == "__main__":
    main()
