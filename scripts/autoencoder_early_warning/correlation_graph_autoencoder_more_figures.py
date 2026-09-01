#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Experiments B/C: turbine-specific Graph Autoencoder for SCADA anomaly detection.

Run the SAME script twice:
  B: --graph-method spearman   (correlation graph)
  C: --graph-method mi         (mutual-information graph)

The evaluation protocol is intentionally aligned with Experiment A:
- one independent model per event CSV/turbine;
- only earlier normal history is used for feature selection, imputation, scaling,
  graph construction, model fitting and threshold calibration;
- 24 h windows (144 x 10 min);
- dense validation/test scoring;
- global reconstruction error + local sensor confirmation;
- causal k-of-n persistence + episode filtering;
- 7/14/30-day early-warning recall, interval detection recall, normal-event
  specificity/FAR, validation FAR and median lead time.

The ONLY intended experimental change relative to A is the representation model:
A = convolutional Autoencoder
B = Graph Autoencoder with correlation graph
C = Graph Autoencoder with mutual-information graph
"""

from __future__ import annotations

import argparse, json, math, os, random, re, traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import mutual_info_regression


@dataclass
class Config:
    datasets_dir: str
    metadata_file: str
    feature_description_file: str
    output_dir: str
    graph_method: str = "spearman"       # spearman | pearson | mi

    timestamp_column: str = "time_stamp"
    asset_id_column: str = "asset_id"
    event_id_column: str = "event_id"
    label_column: str = "event_label"
    start_column: str = "event_start"
    end_column: str = "event_end"
    description_column: str = "event_description"

    sampling_minutes: int = 10
    feature_mode: str = "avg_std"
    top_k_features: int = 30
    max_missing_fraction: float = 0.20
    max_constant_fraction: float = 0.995
    max_abs_correlation: float = 0.98

    window_size: int = 144
    train_step: int = 72
    validation_step: int = 72
    test_step: int = 6
    min_window_valid_fraction: float = 0.98

    anomaly_pre_exclusion_days: float = 14.0
    normal_pre_exclusion_days: float = 1.0
    train_fraction_before_validation: float = 0.80
    split_gap_hours: float = 24.0
    test_lookback_days: float = 30.0
    post_interval_test_days: float = 2.0
    warning_match_days: float = 30.0

    seed: int = 42
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    graph_hidden_dim: int = 64
    latent_dim: int = 16
    dropout: float = 0.20
    l2: float = 5e-4
    patience: int = 12

    graph_top_k_neighbors: int = 5
    graph_min_edge_weight: float = 0.0
    mi_max_samples: int = 30000

    threshold_quantile: float = 0.99
    target_validation_far: float = 0.02
    threshold_quantile_grid: str = "0.95,0.975,0.99,0.995,0.9975"
    score_smoothing_windows: int = 3
    persistence_lookback_windows: int = 12
    persistence_min_positive: int = 8
    local_threshold_quantile: float = 0.99
    minimum_local_sensor_count: int = 1
    minimum_episode_hours: float = 6.0
    max_validation_episodes_per_30d: float = 1.0
    merge_gap_hours: float = 2.0

    include_labels: str = "all"
    event_ids: str = ""
    overwrite: bool = False

    # Figure generation
    generate_plots: bool = True
    plot_dpi: int = 180
    plot_top_n: int = 12


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets-dir", required=True)
    p.add_argument("--metadata-file", required=True)
    p.add_argument("--feature-description-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--graph-method", choices=["spearman", "pearson", "mi"], required=True)
    p.add_argument("--feature-mode", choices=["avg", "avg_std", "all_numeric"], default="avg_std")
    p.add_argument("--top-k-features", type=int, default=30)
    p.add_argument("--graph-top-k-neighbors", type=int, default=5)
    p.add_argument("--mi-max-samples", type=int, default=30000)
    p.add_argument("--anomaly-pre-exclusion-days", type=float, default=14.0)
    p.add_argument("--normal-pre-exclusion-days", type=float, default=1.0)
    p.add_argument("--test-lookback-days", type=float, default=30.0)
    p.add_argument("--post-interval-test-days", type=float, default=2.0)
    p.add_argument("--warning-match-days", type=float, default=30.0)
    p.add_argument("--threshold-quantile", type=float, default=0.99)
    p.add_argument("--target-validation-far", type=float, default=0.02)
    p.add_argument("--threshold-quantile-grid", default="0.95,0.975,0.99,0.995,0.9975")
    p.add_argument("--score-smoothing-windows", type=int, default=3)
    p.add_argument("--persistence-lookback-windows", type=int, default=12)
    p.add_argument("--persistence-min-positive", type=int, default=8)
    p.add_argument("--local-threshold-quantile", type=float, default=0.99)
    p.add_argument("--minimum-local-sensor-count", type=int, default=1)
    p.add_argument("--minimum-episode-hours", type=float, default=6.0)
    p.add_argument("--max-validation-episodes-per-30d", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--graph-hidden-dim", type=int, default=64)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--include-labels", choices=["all", "anomaly", "normal"], default="all")
    p.add_argument("--event-ids", default="")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-plots", action="store_true", help="Disable PNG figure generation.")
    p.add_argument("--plot-dpi", type=int, default=180)
    p.add_argument("--plot-top-n", type=int, default=12)
    ns = p.parse_args()
    cfg = Config(ns.datasets_dir, ns.metadata_file, ns.feature_description_file,
                 ns.output_dir, ns.graph_method)
    for k, v in vars(ns).items():
        if hasattr(cfg, k) and k not in {"datasets_dir","metadata_file","feature_description_file","output_dir","graph_method","no_plots"}:
            setattr(cfg, k, v)
    cfg.generate_plots = not bool(ns.no_plots)
    return cfg


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
        try: tf.config.experimental.enable_op_determinism()
        except Exception: pass
    except ImportError: pass


def tfmod():
    try: import tensorflow as tf
    except ImportError as e: raise ImportError("Install TensorFlow: python -m pip install tensorflow") from e
    return tf


def read_csv_auto(path: Path):
    if not path.exists(): raise FileNotFoundError(path)
    err = None
    for sep in (";", ",", "\t"):
        try:
            df = pd.read_csv(path, sep=sep, low_memory=False)
            if df.shape[1] > 1: return df
        except Exception as e: err = e
    raise RuntimeError(f"Could not read {path}: {err}")


def norm_name(x): return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def find_column(cols: Iterable[str], requested: str, alternatives: Sequence[str]):
    m = {norm_name(c): c for c in cols}
    for c in [requested, *alternatives]:
        if norm_name(c) in m: return m[norm_name(c)]
    raise KeyError(f"Cannot find {requested!r}")


NON_SENSOR = {"time_stamp","timestamp","time","datetime","date_time","asset_id","event_id","id","train_test","status_type_id","label","target","event_label","event_start","event_end","description"}
LEAKAGE_TOKENS = {"fault","failure","anomaly","label","target","prediction","event_"}


def feature_allowed(name, mode):
    n = norm_name(name)
    if n in NON_SENSOR or any(t in n for t in LEAKAGE_TOKENS): return False
    if mode == "avg": return n.endswith("_avg")
    if mode == "avg_std": return n.endswith("_avg") or n.endswith("_std")
    return True


def canonical_label(v):
    s = norm_name(v)
    if s in {"anomaly","abnormal","fault","failure","1","true"}: return "anomaly"
    if s in {"normal","healthy","0","false"}: return "normal"
    return s


def load_metadata(cfg):
    m = read_csv_auto(Path(cfg.metadata_file))
    ic = find_column(m.columns, cfg.event_id_column, ["id","source_id"])
    lc = find_column(m.columns, cfg.label_column, ["label","event_type"])
    sc = find_column(m.columns, cfg.start_column, ["start","start_time"])
    ec = find_column(m.columns, cfg.end_column, ["end","end_time"])
    try: dc = find_column(m.columns, cfg.description_column, ["description","event_desc"])
    except KeyError: dc = None
    out = pd.DataFrame({
        "event_id": m[ic].astype(str).str.strip(),
        "label": m[lc].map(canonical_label),
        "interval_start": pd.to_datetime(m[sc], errors="coerce"),
        "interval_end": pd.to_datetime(m[ec], errors="coerce"),
        "event_description": "" if dc is None else m[dc].fillna("").astype(str)
    }).dropna(subset=["interval_start","interval_end"])
    out = out[out.interval_end > out.interval_start].drop_duplicates("event_id", keep="last")
    if cfg.include_labels != "all": out = out[out.label == cfg.include_labels]
    if cfg.event_ids.strip():
        wanted = {x.strip() for x in cfg.event_ids.split(",") if x.strip()}
        out = out[out.event_id.isin(wanted)]
    return out.reset_index(drop=True)


def load_feature_description(path):
    df = read_csv_auto(path); c = {norm_name(x): x for x in df.columns}
    sc = c.get("sensor_name") or c.get("feature") or c.get("name")
    dc = c.get("description") or c.get("feature_description")
    uc = c.get("unit"); st = c.get("statistics_type") or c.get("statistics")
    if sc is None or dc is None: raise ValueError("Need sensor_name and description columns")
    out = pd.DataFrame({"base_signal":df[sc].astype(str).str.strip(),
                        "description":df[dc].fillna("").astype(str).str.strip(),
                        "unit":"" if uc is None else df[uc].fillna("").astype(str),
                        "statistics_type":"" if st is None else df[st].fillna("").astype(str)})
    out["base_signal_norm"] = out.base_signal.map(norm_name)
    return out.drop_duplicates("base_signal_norm", keep="first")


def parse_measurement_name(f):
    n = norm_name(f)
    for suf, stat in {"_avg":"average","_std":"std_dev","_min":"minimum","_max":"maximum"}.items():
        if n.endswith(suf): return n[:-len(suf)], stat
    return n, "raw"


def enrich_features(features, fd):
    lk = fd.set_index("base_signal_norm").to_dict("index"); rows=[]
    for f in features:
        b, s = parse_measurement_name(f); info=lk.get(b,{})
        rows.append({"feature":f,"base_signal":b,"statistic":s,
                     "physical_description":info.get("description","UNKNOWN"),
                     "unit":info.get("unit","")})
    return pd.DataFrame(rows)


def load_and_prepare(path, cfg):
    raw = read_csv_auto(path)
    tc = find_column(raw.columns, cfg.timestamp_column, ["timestamp","datetime","date_time","time"])
    try: ac = find_column(raw.columns, cfg.asset_id_column, ["asset","turbine_id","wt_id"])
    except KeyError: ac = None
    raw[tc] = pd.to_datetime(raw[tc], errors="coerce")
    raw = raw.dropna(subset=[tc]).sort_values(tc).drop_duplicates(tc, keep="last").reset_index(drop=True)
    asset = None
    if ac is not None:
        vals = raw[ac].dropna().astype(str).unique().tolist()
        if len(vals) > 1: raise ValueError(f"Multiple assets in {path.name}: {vals[:10]}")
        asset = vals[0] if vals else None
    return raw.rename(columns={tc:"__timestamp__"}), asset


def split_contiguous_segments(df, sampling_minutes):
    expected = pd.Timedelta(minutes=sampling_minutes)
    breaks = df.__timestamp__.diff().fillna(expected) > expected*1.5
    return [g.reset_index(drop=True) for _,g in df.groupby(breaks.cumsum()) if len(g)]


def robust_variability(s):
    x = pd.to_numeric(s, errors="coerce").dropna().to_numpy(float)
    if len(x)<10: return 0.0
    q25,q75=np.nanpercentile(x,[25,75]); mad=np.nanmedian(np.abs(x-np.nanmedian(x)))
    return max(float(q75-q25), float(1.4826*mad))


def select_features(train_rows, cfg):
    rows=[]; cand=[]
    for col in train_rows.columns:
        if col=="__timestamp__" or not feature_allowed(col,cfg.feature_mode): continue
        x=pd.to_numeric(train_rows[col],errors="coerce")
        if x.notna().sum()<max(100,cfg.window_size): continue
        miss=float(x.isna().mean())
        if miss>cfg.max_missing_fraction: continue
        vc=x.dropna().value_counts(normalize=True); const=float(vc.iloc[0]) if len(vc) else 1.0
        if const>=cfg.max_constant_fraction: continue
        var=robust_variability(x)
        if not np.isfinite(var) or var<=0: continue
        cand.append(col); rows.append({"feature":col,"missing_fraction":miss,"largest_value_fraction":const,"robust_variability":var})
    if not cand: raise ValueError("No usable features")
    ranking=pd.DataFrame(rows).sort_values(["robust_variability","missing_fraction"],ascending=[False,True]).reset_index(drop=True)
    s=train_rows[cand].apply(pd.to_numeric,errors="coerce")
    if len(s)>50000: s=s.sample(50000,random_state=cfg.seed)
    s=s.interpolate(limit_direction="both").fillna(s.median())
    corr=s.corr(method="spearman").abs(); selected=[]
    for col in ranking.feature:
        if all(float(corr.loc[col,k])<cfg.max_abs_correlation for k in selected): selected.append(col)
        if len(selected)>=cfg.top_k_features: break
    if len(selected)<3: raise ValueError("Fewer than 3 features survived")
    ranking["selected"]=ranking.feature.isin(selected)
    return selected, ranking


def fit_scaler(train_rows, features):
    x=train_rows[features].apply(pd.to_numeric,errors="coerce"); med=x.median()
    x=x.interpolate(limit_direction="both").fillna(med)
    sc=RobustScaler(quantile_range=(25,75)); sc.fit(x.to_numpy(np.float32))
    return sc,med


def transform_segment(seg, features, scaler, medians):
    x=seg[features].apply(pd.to_numeric,errors="coerce"); valid=x.notna().mean(axis=1).to_numpy(float)
    x=x.interpolate(limit_direction="both").fillna(medians)
    return scaler.transform(x.to_numpy(np.float32)).astype(np.float32),valid


def make_windows(df, features, scaler, medians, cfg, start, end, step):
    sub=df[(df.__timestamp__>=start)&(df.__timestamp__<=end)].copy(); W=[]; M=[]
    for seg in split_contiguous_segments(sub,cfg.sampling_minutes):
        if len(seg)<cfg.window_size: continue
        arr,valid=transform_segment(seg,features,scaler,medians)
        for i in range(0,len(seg)-cfg.window_size+1,step):
            j=i+cfg.window_size
            if float(np.mean(valid[i:j]))<cfg.min_window_valid_fraction: continue
            W.append(arr[i:j]); M.append({"window_start":seg.__timestamp__.iloc[i],"window_end":seg.__timestamp__.iloc[j-1]})
    if not W: return np.empty((0,cfg.window_size,len(features)),np.float32),pd.DataFrame(M)
    return np.stack(W).astype(np.float32),pd.DataFrame(M)


def topk_sparsify(w,k,minw=0.0):
    w=np.asarray(w,float).copy(); np.fill_diagonal(w,0); n=w.shape[0]; k=max(1,min(int(k),n-1)); s=np.zeros_like(w)
    for i in range(n):
        for j in np.argsort(w[i])[::-1][:k]:
            if w[i,j]>=minw: s[i,j]=w[i,j]
    s=np.maximum(s,s.T); np.fill_diagonal(s,0); return s


def build_adjacency(train_rows, features, medians, cfg):
    """Build graph from NORMAL TRAINING rows only."""
    x=train_rows[features].apply(pd.to_numeric,errors="coerce").interpolate(limit_direction="both").fillna(medians)
    n=len(features)
    if cfg.graph_method in {"spearman","pearson"}:
        w=np.abs(x.corr(method=cfg.graph_method).to_numpy(float)); w=np.nan_to_num(w)
    else:
        if len(x)>cfg.mi_max_samples: x=x.sample(cfg.mi_max_samples,random_state=cfg.seed).sort_index()
        a=x.to_numpy(float); w=np.zeros((n,n),float)
        for i in range(n):
            for j in range(i+1,n):
                try:
                    mij=mutual_info_regression(a[:,[j]],a[:,i],random_state=cfg.seed)[0]
                    mji=mutual_info_regression(a[:,[i]],a[:,j],random_state=cfg.seed)[0]
                    w[i,j]=w[j,i]=max(0.0,float((mij+mji)/2))
                except Exception: pass
        pos=w[w>0]
        if pos.size:
            scale=float(np.quantile(pos,0.95)) or float(pos.max())
            if scale>0: w=np.clip(w/scale,0,1)
    raw=w.copy(); sparse=topk_sparsify(w,cfg.graph_top_k_neighbors,cfg.graph_min_edge_weight)
    at=sparse+np.eye(n); deg=at.sum(axis=1); dinv=np.diag(1/np.sqrt(np.maximum(deg,1e-12)))
    return raw.astype(np.float32),sparse.astype(np.float32),(dinv@at@dinv).astype(np.float32)


def save_graph(features, raw, sparse, anorm, out):
    pd.DataFrame(raw,index=features,columns=features).to_csv(out/"graph_raw_weights.csv")
    pd.DataFrame(sparse,index=features,columns=features).to_csv(out/"graph_sparse_adjacency.csv")
    np.save(out/"graph_normalized_adjacency.npy",anorm)
    edges=[]
    for i in range(len(features)):
        for j in range(i+1,len(features)):
            if sparse[i,j]>0: edges.append({"sensor_i":features[i],"sensor_j":features[j],"edge_weight":float(sparse[i,j])})
    pd.DataFrame(edges).sort_values("edge_weight",ascending=False).to_csv(out/"graph_edges.csv",index=False)


def build_model(cfg,n_sensors,a_norm):
    tf=tfmod(); reg=tf.keras.regularizers.l2(cfg.l2); A=tf.constant(a_norm,dtype=tf.float32)
    class GCN(tf.keras.layers.Layer):
        def __init__(self,units,activation=None,**kw): super().__init__(**kw); self.units=units; self.act=tf.keras.activations.get(activation)
        def build(self,shape):
            self.W=self.add_weight(name="W",shape=(int(shape[-1]),self.units),initializer="glorot_uniform",regularizer=reg)
            self.b=self.add_weight(name="b",shape=(self.units,),initializer="zeros")
        def call(self,h):
            h=tf.einsum("ij,bjf->bif",A,h); h=tf.einsum("bif,fo->bio",h,self.W)+self.b
            return self.act(h) if self.act is not None else h
    inp=tf.keras.Input((n_sensors,cfg.window_size),name="sensor_graph_window")
    x=GCN(cfg.graph_hidden_dim,"relu",name="gcn_enc1")(inp); x=tf.keras.layers.LayerNormalization()(x); x=tf.keras.layers.Dropout(cfg.dropout)(x)
    z=GCN(cfg.latent_dim,"relu",name="graph_latent")(x); z=tf.keras.layers.LayerNormalization()(z)
    x=GCN(cfg.graph_hidden_dim,"relu",name="gcn_dec1")(z); x=tf.keras.layers.LayerNormalization()(x); x=tf.keras.layers.Dropout(cfg.dropout)(x)
    out=GCN(cfg.window_size,None,name="reconstruction")(x)
    m=tf.keras.Model(inp,out,name=f"{cfg.graph_method}_graph_autoencoder"); m.compile(tf.keras.optimizers.Adam(cfg.learning_rate),loss="mae")
    return m


def as_graph(x): return np.transpose(x,(0,2,1)).astype(np.float32)


def reconstruction_scores(model,x,batch):
    if len(x)==0: return np.empty(0),np.empty((0,x.shape[-1])),np.empty((0,x.shape[1]))
    pred=np.transpose(model.predict(as_graph(x),batch_size=batch,verbose=0),(0,2,1)); err=np.abs(x-pred)
    return err.mean((1,2)),err.mean(1),err.mean(2)


def rolling_median(v,n): return pd.Series(np.asarray(v,float)).rolling(max(1,int(n)),min_periods=1).median().to_numpy(float)

def apply_k_of_n(flags,n,k):
    n=max(1,int(n)); k=min(max(1,int(k)),n); c=pd.Series(np.asarray(flags,bool).astype(int)).rolling(n,min_periods=n).sum().to_numpy()
    return np.nan_to_num(c,nan=0)>=k


def count_episodes(times,flags,cfg):
    a=pd.DataFrame({"t":pd.to_datetime(times),"f":np.asarray(flags,bool)}); a=a[a.f].sort_values("t")
    if a.empty: return 0
    maxgap=pd.Timedelta(hours=cfg.merge_gap_hours+cfg.test_step*cfg.sampling_minutes/60); n=1; prev=a.t.iloc[0]
    for t in a.t.iloc[1:]:
        if t-prev>maxgap: n+=1
        prev=t
    return n


def calibrate(gs,ss,times,sensor_thr,cfg):
    qs=[]
    for tok in cfg.threshold_quantile_grid.split(","):
        try: qs.append(float(tok))
        except: pass
    qs=sorted({*qs,cfg.threshold_quantile,1-cfg.target_validation_far})
    smooth=rolling_median(gs,cfg.score_smoothing_windows); lc=(ss>sensor_thr[None,:]).sum(1)
    days=max(1e-9,(pd.to_datetime(times).max()-pd.to_datetime(times).min()).total_seconds()/86400)
    rows=[]; chosen=None
    for q in qs:
        thr=float(np.quantile(gs,min(max(q,.5),.9999))); cand=(smooth>thr)&(lc>=cfg.minimum_local_sensor_count)
        conf=apply_k_of_n(cand,cfg.persistence_lookback_windows,cfg.persistence_min_positive); far=float(conf.mean())
        ep=count_episodes(times,conf,cfg); ep30=float(ep*30/days); row={"quantile":q,"threshold":thr,"confirmed_window_far":far,"warning_episodes":ep,"warning_episodes_per_30d":ep30}; rows.append(row)
        if chosen is None and far<=cfg.target_validation_far and ep30<=cfg.max_validation_episodes_per_30d: chosen=row
    if chosen is None: chosen=rows[-1]
    chosen["calibration_candidates"]=rows; return chosen


def build_episodes(scored,cfg):
    a=scored[scored.confirmed_warning].sort_values("window_end")
    cols=["episode_start","episode_end","n_windows","max_score","max_score_ratio","dominant_top_sensor","dominant_sensor_description"]
    if a.empty: return pd.DataFrame(columns=cols)
    gap=pd.Timedelta(hours=cfg.merge_gap_hours+cfg.test_step*cfg.sampling_minutes/60); groups=[]; cur=[]; prev=None
    for _,r in a.iterrows():
        t=pd.Timestamp(r.window_end)
        if prev is None or t-prev<=gap: cur.append(r)
        else: groups.append(cur); cur=[r]
        prev=t
    if cur: groups.append(cur)
    rows=[]
    for g in groups:
        f=pd.DataFrame(g); rows.append({"episode_start":pd.to_datetime(f.window_end).min(),"episode_end":pd.to_datetime(f.window_end).max(),"n_windows":len(f),"max_score":float(f.reconstruction_error.max()),"max_score_ratio":float(f.score_ratio.max()),"dominant_top_sensor":f.top_sensor.mode().iloc[0],"dominant_sensor_description":f.top_sensor_description.mode().iloc[0]})
    return pd.DataFrame(rows)


# =============================================================================
# FIGURE GENERATION
# =============================================================================

def _save_figure(fig, path: Path, dpi: int = 180):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_guard(name, fn, *args, **kwargs):
    """Create a figure without allowing a plotting problem to stop model training."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        print(f"[PLOT WARNING] {name}: {exc}")
        return None


def plot_training_history(history_df: pd.DataFrame, fig_dir: Path, cfg, event_id: str):
    if history_df.empty or "loss" not in history_df.columns:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    epochs = np.arange(1, len(history_df) + 1)
    ax.plot(epochs, history_df["loss"], label="Training loss")
    if "val_loss" in history_df.columns:
        ax.plot(epochs, history_df["val_loss"], label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE reconstruction loss")
    ax.set_title(f"Event {event_id}: graph autoencoder training history")
    ax.grid(alpha=0.25)
    ax.legend()
    _save_figure(fig, fig_dir / "01_training_history.png", cfg.plot_dpi)


def plot_graph_matrix(matrix, features, title, path, cfg):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return
    fig, ax = plt.subplots(figsize=(9.5, 8.0))
    im = ax.imshow(matrix, aspect="auto")
    n = len(features)
    if n <= 35:
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(features, rotation=90, fontsize=7)
        ax.set_yticklabels(features, fontsize=7)
    else:
        ax.set_xlabel("Sensor index")
        ax.set_ylabel("Sensor index")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Edge weight")
    fig.tight_layout()
    _save_figure(fig, path, cfg.plot_dpi)


def plot_graph_degree(sparse, features, fig_dir, cfg, event_id):
    sparse = np.asarray(sparse, dtype=float)
    if sparse.size == 0:
        return
    weighted_degree = sparse.sum(axis=1)
    order = np.argsort(weighted_degree)[::-1]
    top_n = min(cfg.plot_top_n, len(features))
    idx = order[:top_n]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(np.arange(top_n), weighted_degree[idx])
    ax.set_xticks(np.arange(top_n))
    ax.set_xticklabels([features[i] for i in idx], rotation=60, ha="right")
    ax.set_ylabel("Weighted graph degree")
    ax.set_title(f"Event {event_id}: highest-connectivity sensors")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, fig_dir / "04_graph_weighted_degree.png", cfg.plot_dpi)


def plot_top_graph_edges(sparse, features, fig_dir, cfg, event_id):
    sparse = np.asarray(sparse, dtype=float)
    rows = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            if sparse[i, j] > 0:
                rows.append((float(sparse[i, j]), f"{features[i]} ↔ {features[j]}"))
    if not rows:
        return
    rows.sort(reverse=True, key=lambda x: x[0])
    rows = rows[: min(cfg.plot_top_n, len(rows))]
    values = [r[0] for r in rows][::-1]
    labels = [r[1] for r in rows][::-1]
    fig, ax = plt.subplots(figsize=(10, max(5.0, 0.38 * len(rows) + 2)))
    ax.barh(np.arange(len(rows)), values)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Edge weight")
    ax.set_title(f"Event {event_id}: strongest graph edges")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, fig_dir / "05_strongest_graph_edges.png", cfg.plot_dpi)


def plot_threshold_calibration(cal_df, fig_dir, cfg, event_id):
    if cal_df.empty:
        return
    cal_df = cal_df.sort_values("quantile").copy()

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(cal_df["quantile"], cal_df["confirmed_window_far"], marker="o")
    ax.axhline(cfg.target_validation_far, linestyle="--", label="Target validation FAR")
    ax.set_xlabel("Threshold quantile")
    ax.set_ylabel("Confirmed-window validation FAR")
    ax.set_title(f"Event {event_id}: threshold calibration – validation FAR")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, fig_dir / "06_threshold_calibration_far.png", cfg.plot_dpi)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(cal_df["quantile"], cal_df["warning_episodes_per_30d"], marker="o")
    ax.axhline(
        cfg.max_validation_episodes_per_30d,
        linestyle="--",
        label="Maximum validation episodes / 30 d",
    )
    ax.set_xlabel("Threshold quantile")
    ax.set_ylabel("Validation warning episodes per 30 days")
    ax.set_title(f"Event {event_id}: threshold calibration – episode burden")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, fig_dir / "07_threshold_calibration_episode_rate.png", cfg.plot_dpi)


def plot_validation_score_distribution(validation_scores, threshold, fig_dir, cfg, event_id):
    values = np.asarray(validation_scores, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.hist(values, bins=min(40, max(10, int(np.sqrt(len(values))))))
    ax.axvline(threshold, linestyle="--", linewidth=2, label=f"Threshold = {threshold:.4g}")
    ax.set_xlabel("Validation reconstruction error")
    ax.set_ylabel("Window count")
    ax.set_title(f"Event {event_id}: validation reconstruction-error distribution")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, fig_dir / "08_validation_score_distribution.png", cfg.plot_dpi)


def plot_test_score_timeline(scored, start, end, fig_dir, cfg, event_id):
    if scored.empty:
        return
    d = scored.copy()
    d["window_end"] = pd.to_datetime(d["window_end"], errors="coerce")
    d = d.dropna(subset=["window_end"]).sort_values("window_end")
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(d["window_end"], d["smoothed_reconstruction_error"], label="Smoothed reconstruction error")
    ax.plot(d["window_end"], d["threshold"], linestyle="--", label="Global threshold")
    ax.axvspan(start, end, alpha=0.12, label="Metadata interval")
    confirmed = d[d["confirmed_warning"]]
    if not confirmed.empty:
        ax.scatter(
            confirmed["window_end"],
            confirmed["smoothed_reconstruction_error"],
            s=22,
            label="Confirmed warning",
            zorder=4,
        )
    ax.set_xlabel("Time")
    ax.set_ylabel("Reconstruction error")
    ax.set_title(f"Event {event_id}: reconstruction score and confirmed warnings")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    _save_figure(fig, fig_dir / "09_test_reconstruction_timeline.png", cfg.plot_dpi)


def plot_local_sensor_exceedance(scored, start, end, fig_dir, cfg, event_id):
    if scored.empty:
        return
    d = scored.copy()
    d["window_end"] = pd.to_datetime(d["window_end"], errors="coerce")
    d = d.dropna(subset=["window_end"]).sort_values("window_end")
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.plot(d["window_end"], d["local_sensor_exceedance_count"])
    ax.axhline(
        cfg.minimum_local_sensor_count,
        linestyle="--",
        label="Minimum local sensor count",
    )
    ax.axvspan(start, end, alpha=0.12, label="Metadata interval")
    ax.set_xlabel("Time")
    ax.set_ylabel("Sensors above local threshold")
    ax.set_title(f"Event {event_id}: local sensor confirmation over time")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    _save_figure(fig, fig_dir / "10_local_sensor_exceedance_timeline.png", cfg.plot_dpi)


def plot_top_sensor_errors(scored, fig_dir, cfg, event_id):
    if scored.empty or "top_sensor" not in scored.columns:
        return
    grouped = (
        scored.groupby("top_sensor", dropna=True)["top_sensor_error"]
        .agg(["mean", "max", "count"])
        .sort_values(["max", "mean"], ascending=False)
        .head(cfg.plot_top_n)
    )
    if grouped.empty:
        return
    grouped = grouped.sort_values("max")
    fig, ax = plt.subplots(figsize=(10, max(5.0, 0.38 * len(grouped) + 2)))
    y = np.arange(len(grouped))
    ax.barh(y, grouped["max"].to_numpy())
    ax.set_yticks(y)
    ax.set_yticklabels(grouped.index.astype(str), fontsize=8)
    ax.set_xlabel("Maximum sensor reconstruction error")
    ax.set_title(f"Event {event_id}: sensors with largest reconstruction errors")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, fig_dir / "11_top_sensor_reconstruction_errors.png", cfg.plot_dpi)


def plot_warning_episode_timeline(eps, start, end, fig_dir, cfg, event_id):
    if eps is None or eps.empty:
        return
    d = eps.copy()
    d["episode_start"] = pd.to_datetime(d["episode_start"], errors="coerce")
    d["episode_end"] = pd.to_datetime(d["episode_end"], errors="coerce")
    d = d.dropna(subset=["episode_start", "episode_end"]).sort_values("episode_start")
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(13, max(4.5, min(9.0, 2.5 + 0.18 * len(d)))))
    y = np.arange(len(d))
    for pos, (_, r) in enumerate(d.iterrows()):
        ax.hlines(pos, r["episode_start"], r["episode_end"], linewidth=3)
        ax.plot(r["episode_start"], pos, marker="|", markersize=10)
        ax.plot(r["episode_end"], pos, marker="|", markersize=10)
    ax.axvspan(start, end, alpha=0.12, label="Metadata interval")
    ax.set_xlabel("Time")
    ax.set_ylabel("Warning episode index")
    ax.set_title(f"Event {event_id}: warning-episode timeline")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="best")
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    _save_figure(fig, fig_dir / "12_warning_episode_timeline.png", cfg.plot_dpi)


def create_event_figures(
    *,
    out,
    cfg,
    event_id,
    start,
    end,
    history_df,
    raw_weights,
    sparse_adjacency,
    features,
    calibration_df,
    validation_scores,
    threshold,
    scored,
    episodes,
):
    if not cfg.generate_plots:
        return
    fig_dir = Path(out) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    _plot_guard("training history", plot_training_history, history_df, fig_dir, cfg, event_id)
    _plot_guard(
        "raw graph heatmap",
        plot_graph_matrix,
        raw_weights,
        features,
        f"Event {event_id}: raw {cfg.graph_method} graph weights",
        fig_dir / "02_graph_raw_weights_heatmap.png",
        cfg,
    )
    _plot_guard(
        "sparse graph heatmap",
        plot_graph_matrix,
        sparse_adjacency,
        features,
        f"Event {event_id}: sparse {cfg.graph_method} graph adjacency",
        fig_dir / "03_graph_sparse_adjacency_heatmap.png",
        cfg,
    )
    _plot_guard("graph degree", plot_graph_degree, sparse_adjacency, features, fig_dir, cfg, event_id)
    _plot_guard("top graph edges", plot_top_graph_edges, sparse_adjacency, features, fig_dir, cfg, event_id)
    _plot_guard("threshold calibration", plot_threshold_calibration, calibration_df, fig_dir, cfg, event_id)
    _plot_guard(
        "validation score distribution",
        plot_validation_score_distribution,
        validation_scores,
        threshold,
        fig_dir,
        cfg,
        event_id,
    )
    _plot_guard("test score timeline", plot_test_score_timeline, scored, start, end, fig_dir, cfg, event_id)
    _plot_guard(
        "local sensor exceedance",
        plot_local_sensor_exceedance,
        scored,
        start,
        end,
        fig_dir,
        cfg,
        event_id,
    )
    _plot_guard("top sensor errors", plot_top_sensor_errors, scored, fig_dir, cfg, event_id)
    _plot_guard(
        "warning episode timeline",
        plot_warning_episode_timeline,
        episodes,
        start,
        end,
        fig_dir,
        cfg,
        event_id,
    )


def create_aggregate_figures(df: pd.DataFrame, metrics: dict, out: Path, cfg):
    if not cfg.generate_plots or df.empty:
        return
    fig_dir = Path(out) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1) Main event-level performance metrics.
    metric_items = [
        ("30 d EW recall", metrics.get("anomaly_early_warning_recall_30d")),
        ("14 d EW recall", metrics.get("anomaly_early_warning_recall_14d")),
        ("7 d EW recall", metrics.get("anomaly_early_warning_recall_7d")),
        ("Interval recall", metrics.get("anomaly_interval_detection_recall")),
        ("Normal specificity", metrics.get("normal_interval_specificity")),
    ]
    metric_items = [(k, v) for k, v in metric_items if v is not None and np.isfinite(v)]
    if metric_items:
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        labels = [x[0] for x in metric_items]
        vals = [float(x[1]) for x in metric_items]
        bars = ax.bar(np.arange(len(vals)), vals)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Rate")
        ax.set_title(f"{cfg.graph_method.upper()} GAE: aggregate detection performance")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{100*v:.1f}%", ha="center", va="bottom")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        _save_figure(fig, fig_dir / "A01_aggregate_detection_metrics.png", cfg.plot_dpi)

    # 2) Explicitly separate Event-level FAR from validation window-level FAR.
    event_far = metrics.get("normal_interval_false_alarm_event_rate")
    val_far = metrics.get("mean_validation_false_alarm_rate")
    vals = []
    labs = []
    if event_far is not None and np.isfinite(event_far):
        labs.append("Normal Event false-positive rate\n(Event-level)")
        vals.append(float(event_far))
    if val_far is not None and np.isfinite(val_far):
        labs.append("Validation false-alarm rate\n(Window/scoring-point level)")
        vals.append(float(val_far))
    if vals:
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        bars = ax.bar(np.arange(len(vals)), vals)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labs)
        ax.set_ylim(0, max(0.05, min(1.0, max(vals) * 1.25)))
        ax.set_ylabel("Rate")
        ax.set_title(f"{cfg.graph_method.upper()} GAE: false-alarm metrics at different aggregation levels")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + max(vals)*0.03 + 0.002, f"{100*v:.2f}%", ha="center")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        _save_figure(fig, fig_dir / "A02_false_alarm_metrics_different_levels.png", cfg.plot_dpi)

    anomaly = df[df["label"] == "anomaly"].copy()

    # 3) Lead-time distribution.
    lead = pd.to_numeric(anomaly.get("lead_time_days"), errors="coerce").dropna()
    if not lead.empty:
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        ax.hist(lead, bins=min(15, max(5, int(np.sqrt(len(lead))))))
        median = float(lead.median())
        ax.axvline(median, linestyle="--", label=f"Median = {median:.2f} d")
        ax.set_xlabel("Lead time (days before metadata Event start)")
        ax.set_ylabel("Detected anomaly Events")
        ax.set_title(f"{cfg.graph_method.upper()} GAE: early-warning lead-time distribution")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        _save_figure(fig, fig_dir / "A03_lead_time_distribution.png", cfg.plot_dpi)

        ranking = anomaly.dropna(subset=["lead_time_days"]).copy()
        ranking["lead_time_days"] = pd.to_numeric(ranking["lead_time_days"], errors="coerce")
        ranking = ranking.dropna(subset=["lead_time_days"]).sort_values("lead_time_days", ascending=False)
        if not ranking.empty:
            fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(ranking) + 4), 5.5))
            x = np.arange(len(ranking))
            ax.bar(x, ranking["lead_time_days"])
            ax.set_xticks(x)
            ax.set_xticklabels(ranking["event_id"].astype(str), rotation=90)
            ax.set_xlabel("Anomaly Event ID")
            ax.set_ylabel("Lead time (days)")
            ax.set_title(f"{cfg.graph_method.upper()} GAE: lead time by detected anomaly Event")
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            _save_figure(fig, fig_dir / "A04_lead_time_by_event.png", cfg.plot_dpi)

    # 4) Graph density distribution.
    density = pd.to_numeric(df.get("graph_density"), errors="coerce").dropna()
    if not density.empty:
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        ax.hist(density, bins=min(15, max(5, int(np.sqrt(len(density))))))
        ax.set_xlabel("Graph density")
        ax.set_ylabel("Event models")
        ax.set_title(f"{cfg.graph_method.upper()} GAE: graph-density distribution")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        _save_figure(fig, fig_dir / "A05_graph_density_distribution.png", cfg.plot_dpi)

    # 5) Graph edge count per Event.
    if "graph_edge_count" in df.columns:
        tmp = df.copy()
        tmp["graph_edge_count"] = pd.to_numeric(tmp["graph_edge_count"], errors="coerce")
        tmp = tmp.dropna(subset=["graph_edge_count"]).sort_values("graph_edge_count", ascending=False)
        if not tmp.empty:
            fig, ax = plt.subplots(figsize=(max(10, 0.35 * len(tmp) + 4), 5.5))
            x = np.arange(len(tmp))
            ax.bar(x, tmp["graph_edge_count"])
            ax.set_xticks(x)
            ax.set_xticklabels(tmp["event_id"].astype(str), rotation=90)
            ax.set_xlabel("Event ID")
            ax.set_ylabel("Sparse graph edge count")
            ax.set_title(f"{cfg.graph_method.upper()} GAE: graph edge count by Event")
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            _save_figure(fig, fig_dir / "A06_graph_edge_count_by_event.png", cfg.plot_dpi)

    # 6) Validation FAR by Event.
    if "validation_false_alarm_rate" in df.columns:
        tmp = df.copy()
        tmp["validation_false_alarm_rate"] = pd.to_numeric(tmp["validation_false_alarm_rate"], errors="coerce")
        tmp = tmp.dropna(subset=["validation_false_alarm_rate"]).sort_values("validation_false_alarm_rate", ascending=False)
        if not tmp.empty:
            fig, ax = plt.subplots(figsize=(max(10, 0.35 * len(tmp) + 4), 5.5))
            x = np.arange(len(tmp))
            ax.bar(x, tmp["validation_false_alarm_rate"])
            ax.axhline(cfg.target_validation_far, linestyle="--", label="Target validation FAR")
            ax.set_xticks(x)
            ax.set_xticklabels(tmp["event_id"].astype(str), rotation=90)
            ax.set_xlabel("Event ID")
            ax.set_ylabel("Validation FAR (window/scoring-point level)")
            ax.set_title(f"{cfg.graph_method.upper()} GAE: validation false-alarm rate by Event")
            ax.grid(axis="y", alpha=0.25)
            ax.legend()
            fig.tight_layout()
            _save_figure(fig, fig_dir / "A07_validation_far_by_event.png", cfg.plot_dpi)

    # 7) Warning episode burden by Event.
    if "total_warning_episode_count" in df.columns:
        tmp = df.copy()
        tmp["total_warning_episode_count"] = pd.to_numeric(tmp["total_warning_episode_count"], errors="coerce")
        tmp = tmp.dropna(subset=["total_warning_episode_count"]).sort_values("total_warning_episode_count", ascending=False)
        if not tmp.empty:
            fig, ax = plt.subplots(figsize=(max(10, 0.35 * len(tmp) + 4), 5.5))
            x = np.arange(len(tmp))
            ax.bar(x, tmp["total_warning_episode_count"])
            ax.set_xticks(x)
            ax.set_xticklabels(tmp["event_id"].astype(str), rotation=90)
            ax.set_xlabel("Event ID")
            ax.set_ylabel("Warning episode count")
            ax.set_title(f"{cfg.graph_method.upper()} GAE: warning-episode burden by Event")
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            _save_figure(fig, fig_dir / "A08_warning_episode_count_by_event.png", cfg.plot_dpi)

    # 8) Event-level outcome counts.
    a = df[df["label"] == "anomaly"]
    n = df[df["label"] == "normal"]
    if len(a) or len(n):
        anomaly_detected = int(a["anomaly_interval_detected"].fillna(False).astype(bool).sum()) if len(a) else 0
        anomaly_missed = int(len(a) - anomaly_detected)
        normal_clean = int(n["normal_interval_clean"].fillna(False).astype(bool).sum()) if len(n) else 0
        normal_false_alarm = int(len(n) - normal_clean)
        labels = ["Anomaly detected", "Anomaly missed", "Normal clean", "Normal false alarm"]
        vals = [anomaly_detected, anomaly_missed, normal_clean, normal_false_alarm]
        fig, ax = plt.subplots(figsize=(9, 5.0))
        bars = ax.bar(np.arange(len(vals)), vals)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Number of Events")
        ax.set_title(f"{cfg.graph_method.upper()} GAE: Event-level outcome counts")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.3, str(v), ha="center")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        _save_figure(fig, fig_dir / "A09_event_level_outcome_counts.png", cfg.plot_dpi)


def write_figure_manifest(out: Path):
    rows = []
    for path in sorted(Path(out).rglob("*.png")):
        rel = path.relative_to(out)
        rows.append({
            "scope": "aggregate" if str(rel).startswith("figures") else "event",
            "figure_file": str(rel),
        })
    pd.DataFrame(rows).to_csv(Path(out) / "figure_manifest.csv", index=False)


def process_event(row,cfg,fd):
    eid=str(row.event_id); label=str(row.label); start=pd.Timestamp(row.interval_start); end=pd.Timestamp(row.interval_end); desc=str(row.event_description)
    csv=Path(cfg.datasets_dir)/f"{eid}.csv"; out=Path(cfg.output_dir)/f"event_{eid}_{label}"; out.mkdir(parents=True,exist_ok=True)
    sp=out/"summary.json"
    if sp.exists() and not cfg.overwrite: return json.loads(sp.read_text(encoding="utf-8"))
    df,asset=load_and_prepare(csv,cfg); excl=cfg.anomaly_pre_exclusion_days if label=="anomaly" else cfg.normal_pre_exclusion_days
    cutoff=start-pd.Timedelta(days=excl); pre=df[df.__timestamp__<cutoff].copy()
    if len(pre)<cfg.window_size*10: raise ValueError("Insufficient pre-event history")
    earliest,latest=pre.__timestamp__.min(),pre.__timestamp__.max(); split=earliest+(latest-earliest)*cfg.train_fraction_before_validation; gap=pd.Timedelta(hours=cfg.split_gap_hours)
    train_end=split-gap/2; val_start=split+gap/2; val_end=cutoff
    tr=df[(df.__timestamp__>=earliest)&(df.__timestamp__<=train_end)].copy(); va=df[(df.__timestamp__>=val_start)&(df.__timestamp__<val_end)].copy()
    if len(tr)<cfg.window_size*5 or len(va)<cfg.window_size*2: raise ValueError("Train/validation periods too short")
    features,ranking=select_features(tr,cfg); mapping=enrich_features(features,fd); lookup=mapping.set_index("feature").physical_description.to_dict()
    ranking.merge(mapping,on="feature",how="left").to_csv(out/"feature_ranking_with_descriptions.csv",index=False); mapping.to_csv(out/"selected_features_with_descriptions.csv",index=False)
    scaler,med=fit_scaler(tr,features); (out/"scaler.json").write_text(json.dumps({"features":features,"center":scaler.center_.tolist(),"scale":scaler.scale_.tolist(),"training_medians":med.to_dict()},indent=2,default=float),encoding="utf-8")
    raww,sparse,anorm=build_adjacency(tr,features,med,cfg); save_graph(features,raww,sparse,anorm,out)
    xtr,_=make_windows(df,features,scaler,med,cfg,earliest,train_end,cfg.train_step); xva,_=make_windows(df,features,scaler,med,cfg,val_start,val_end,cfg.validation_step)
    if not len(xtr) or not len(xva): raise ValueError("No train/validation windows")
    tf=tfmod(); tf.keras.backend.clear_session(); set_seed(cfg.seed+int(re.sub(r"\D","",eid) or 0)); model=build_model(cfg,len(features),anorm)
    cb=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=cfg.patience,restore_best_weights=True,min_delta=1e-5),tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=.5,patience=max(3,cfg.patience//3),min_lr=1e-6)]
    hist=model.fit(as_graph(xtr),as_graph(xtr),validation_data=(as_graph(xva),as_graph(xva)),epochs=cfg.epochs,batch_size=cfg.batch_size,shuffle=True,callbacks=cb,verbose=2)
    model.save_weights(out/"graph_autoencoder.weights.h5"); history_df=pd.DataFrame(hist.history); history_df.to_csv(out/"training_history.csv",index=False)
    xvd,mvd=make_windows(df,features,scaler,med,cfg,val_start,val_end,cfg.test_step); vg,vs,_=reconstruction_scores(model,xvd,cfg.batch_size)
    sth=np.quantile(vs,min(max(cfg.local_threshold_quantile,.5),.9999),axis=0); cal=calibrate(vg,vs,mvd.window_end,sth,cfg); thr=float(cal["threshold"])
    vsm=rolling_median(vg,cfg.score_smoothing_windows); vlc=(vs>sth[None,:]).sum(1); vc=(vsm>thr)&(vlc>=cfg.minimum_local_sensor_count); vconf=apply_k_of_n(vc,cfg.persistence_lookback_windows,cfg.persistence_min_positive); vfar=float(vconf.mean())
    calibration_df=pd.DataFrame(cal["calibration_candidates"]); calibration_df.to_csv(out/"threshold_calibration_candidates.csv",index=False)
    pd.DataFrame({"feature":features,"local_reconstruction_threshold":sth,"physical_description":[lookup.get(f,"UNKNOWN") for f in features]}).to_csv(out/"local_sensor_thresholds.csv",index=False)
    test_start=max(df.__timestamp__.min(),start-pd.Timedelta(days=cfg.test_lookback_days)); test_end=min(df.__timestamp__.max(),end+pd.Timedelta(days=cfg.post_interval_test_days)); xt,mt=make_windows(df,features,scaler,med,cfg,test_start,test_end,cfg.test_step)
    g,s,ts=reconstruction_scores(model,xt,cfg.batch_size); top=np.argmax(s,axis=1); scored=mt.copy(); scored["event_id"]=eid; scored["label"]=label; scored["reconstruction_error"]=g; scored["smoothed_reconstruction_error"]=rolling_median(g,cfg.score_smoothing_windows); scored["threshold"]=thr; scored["score_ratio"]=scored.smoothed_reconstruction_error/max(thr,1e-12); scored["global_above_threshold"]=scored.smoothed_reconstruction_error>thr; lex=s>sth[None,:]; scored["local_sensor_exceedance_count"]=lex.sum(1); scored["candidate_warning"]=scored.global_above_threshold&(scored.local_sensor_exceedance_count>=cfg.minimum_local_sensor_count); scored["confirmed_warning"]=apply_k_of_n(scored.candidate_warning.to_numpy(),cfg.persistence_lookback_windows,cfg.persistence_min_positive); scored["top_sensor"]=[features[i] for i in top]; scored["top_sensor_description"]=[lookup.get(features[i],"UNKNOWN") for i in top]; scored["top_sensor_error"]=s[np.arange(len(s)),top]; scored["max_timepoint_error"]=ts.max(1)
    eps=build_episodes(scored,cfg)
    if not eps.empty:
        eps["duration_hours"]=(pd.to_datetime(eps.episode_end)-pd.to_datetime(eps.episode_start)).dt.total_seconds()/3600+cfg.test_step*cfg.sampling_minutes/60; eps=eps[eps.duration_hours>=cfg.minimum_episode_hours].reset_index(drop=True); scored["confirmed_warning"]=False
        for _,ep in eps.iterrows(): scored.loc[scored.window_end.between(ep.episode_start,ep.episode_end,inclusive="both"),"confirmed_warning"]=True
        eps["episode_phase"]=np.select([pd.to_datetime(eps.episode_start)<start,pd.to_datetime(eps.episode_start).between(start,end,inclusive="both"),pd.to_datetime(eps.episode_start)>end],["pre_interval","recorded_interval","post_interval"],default="unknown")
    scored.to_csv(out/"window_scores.csv",index=False); eps.to_csv(out/"warning_episodes.csv",index=False); pd.DataFrame(s,columns=features).assign(window_end=scored.window_end.to_numpy()).to_csv(out/"sensor_reconstruction_errors.csv",index=False)
    create_event_figures(
        out=out, cfg=cfg, event_id=eid, start=start, end=end,
        history_df=history_df, raw_weights=raww, sparse_adjacency=sparse,
        features=features, calibration_df=calibration_df,
        validation_scores=vg, threshold=thr, scored=scored, episodes=eps,
    )
    match=start-pd.Timedelta(days=cfg.warning_match_days); preeps=eps[(pd.to_datetime(eps.episode_start)>=match)&(pd.to_datetime(eps.episode_start)<start)] if not eps.empty else eps; first=pd.to_datetime(preeps.episode_start).min() if not preeps.empty else pd.NaT; lead=((start-first).total_seconds()/3600) if not pd.isna(first) else math.nan
    inside=scored[scored.window_end.between(start,end,inclusive="both")]
    horizons={}
    for d in (7,14,30):
        h=start-pd.Timedelta(days=d); heps=eps[(pd.to_datetime(eps.episode_start)>=h)&(pd.to_datetime(eps.episode_start)<start)] if not eps.empty else eps; horizons[f"anomaly_early_warning_{d}d"]=bool(label=="anomaly" and not heps.empty)
    summary={"event_id":eid,"label":label,"event_description":desc,"asset_id":asset,"graph_method":cfg.graph_method,"graph_edge_count":int(np.count_nonzero(np.triu(sparse,1))),"graph_density":float(np.count_nonzero(np.triu(sparse,1))/max(1,len(features)*(len(features)-1)/2)),"selected_feature_count":len(features),"training_windows":len(xtr),"validation_windows":len(xva),"test_windows":len(xt),"threshold":thr,"threshold_quantile_selected":float(cal["quantile"]),"validation_false_alarm_rate":vfar,"validation_warning_episodes_per_30d":float(cal["warning_episodes_per_30d"]),"anomaly_early_warning_detected":bool(label=="anomaly" and not preeps.empty),**horizons,"anomaly_interval_detected":bool(label=="anomaly" and inside.confirmed_warning.any()),"normal_interval_clean":bool(label=="normal" and not inside.confirmed_warning.any()),"first_pre_interval_warning":None if pd.isna(first) else str(first),"lead_time_days":None if not np.isfinite(lead) else lead/24,"total_warning_episode_count":len(eps)}
    sp.write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8"); return summary


def aggregate(results,out,method,cfg):
    df=pd.DataFrame(results); df.to_csv(out/"all_turbines_event_results.csv",index=False); a=df[df.label=="anomaly"]; n=df[df.label=="normal"]
    M={"graph_method":method,"processed_events":len(df),"anomaly_events":len(a),"normal_events":len(n),"anomaly_early_warning_recall":float(a.anomaly_early_warning_detected.mean()) if len(a) else None,"anomaly_early_warning_recall_7d":float(a.anomaly_early_warning_7d.mean()) if len(a) else None,"anomaly_early_warning_recall_14d":float(a.anomaly_early_warning_14d.mean()) if len(a) else None,"anomaly_early_warning_recall_30d":float(a.anomaly_early_warning_30d.mean()) if len(a) else None,"anomaly_interval_detection_recall":float(a.anomaly_interval_detected.mean()) if len(a) else None,"normal_interval_specificity":float(n.normal_interval_clean.mean()) if len(n) else None,"normal_interval_false_alarm_event_rate":float((~n.normal_interval_clean).mean()) if len(n) else None,"median_lead_time_days_detected_anomalies":float(a.loc[a.anomaly_early_warning_detected,"lead_time_days"].median()) if len(a) and a.anomaly_early_warning_detected.any() else None,"mean_validation_false_alarm_rate":float(df.validation_false_alarm_rate.mean()) if len(df) else None,"mean_validation_warning_episodes_per_30d":float(df.validation_warning_episodes_per_30d.mean()) if len(df) else None,"mean_graph_density":float(df.graph_density.mean()) if len(df) else None,"mean_graph_edge_count":float(df.graph_edge_count.mean()) if len(df) else None}
    (out/"aggregate_metrics.json").write_text(json.dumps(M,indent=2,ensure_ascii=False),encoding="utf-8")
    create_aggregate_figures(df, M, out, cfg)
    return M


def main():
    cfg=parse_args(); set_seed(cfg.seed); out=Path(cfg.output_dir); out.mkdir(parents=True,exist_ok=True); (out/"run_config.json").write_text(json.dumps(asdict(cfg),indent=2),encoding="utf-8")
    meta=load_metadata(cfg); fd=load_feature_description(Path(cfg.feature_description_file)); results=[]; failures=[]; print(f"Graph method: {cfg.graph_method}; events: {len(meta)}")
    for i,row in meta.iterrows():
        print(f"[{i+1}/{len(meta)}] Event {row.event_id} ({row.label})")
        try: results.append(process_event(row,cfg,fd))
        except Exception as e: failures.append({"event_id":str(row.event_id),"label":str(row.label),"error":str(e),"traceback":traceback.format_exc()}); print("FAILED:",e)
    if results:
        M=aggregate(results,out,cfg.graph_method,cfg); print(json.dumps(M,indent=2))
    pd.DataFrame(failures).to_csv(out/"failed_events.csv",index=False)
    if cfg.generate_plots: write_figure_manifest(out)
    print("COMPLETE",len(results),"successful,",len(failures),"failed")


if __name__ == "__main__": main()
