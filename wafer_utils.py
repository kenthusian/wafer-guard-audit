# wafer_utils.py
# Helper code for wafer_guard_fixed.ipynb.
# I moved everything out of the notebook into this file so that
#   1) it can be tested with pytest (see tests/)
#   2) the saved pipeline (joblib) can be loaded again without re-running the notebook

import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, precision_score, recall_score, roc_auc_score,
                             average_precision_score, confusion_matrix, precision_recall_curve)
from imblearn.over_sampling import SMOTE

SEED = 1337

# Probe station PS-L (legacy) rules from the process integration team
LEGACY_SENTINEL = 0.0                 # sensor_0 == 0.0 means the wafer went through PS-L
LEGACY_CODES = [100.0, 200.0, 300.0]  # on those wafers sensor_1 is a firmware error code

# preprocessing settings
MAX_MISSING = 0.50     # drop a sensor if more than half of the TRAIN rows are missing
MIN_MISSING_FLAG = 0.01  # add a "was missing" column if at least 1% of train rows are missing
Z_CLIP = 10.0          # clip scaled values so one crazy reading can't blow up the model

NATURAL_PREVALENCE = 0.0664   # 104 defects / 1567 wafers in the whole dataset
BATCH_SIZE = 64
LEARNING_RATE = 1e-3


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_sensor_cols(df):
    return [c for c in df.columns if c.startswith("sensor_")]


def time_split(n, fracs=(0.6, 0.2, 0.2)):
    # rows are already sorted by time, so this is just cutting the list in order
    # (oldest -> train, middle -> val, newest -> test). No shuffling!
    cut1 = int(fracs[0] * n)
    cut2 = int((fracs[0] + fracs[1]) * n)
    return np.arange(0, cut1), np.arange(cut1, cut2), np.arange(cut2, n)


# ---------------------------------------------------------------------------
# Legacy probe station handling (bug 3)
# ---------------------------------------------------------------------------

def legacy_view(frame, sensor_cols):
    """Look at each row on its own and work out the PS-L stuff.

    returns
      is_legacy : bool array [N]
      codes     : one-hot float array [N, 4] -> code 100, 200, 300, unknown
      not_real  : bool array [N, n_sensors], True where the value is NOT a real measurement
    Nothing is learned from other rows here so this can't leak anything.
    """
    s0 = frame["sensor_0"].to_numpy(np.float64)
    s1 = frame["sensor_1"].to_numpy(np.float64)

    is_legacy = s0 == LEGACY_SENTINEL
    known = np.stack([s1 == c for c in LEGACY_CODES], axis=1)
    # legacy wafer but the code isn't one we know about -> put it in "unknown"
    unknown = is_legacy & ~known.any(axis=1)
    # if a known code ever shows up on a normal wafer it's still a code, not a reading
    has_code = is_legacy | known.any(axis=1)

    codes = np.concatenate([known & has_code[:, None], unknown[:, None]], axis=1).astype(np.float32)

    not_real = np.zeros((len(frame), len(sensor_cols)), dtype=bool)
    not_real[is_legacy, sensor_cols.index("sensor_0")] = True
    not_real[has_code, sensor_cols.index("sensor_1")] = True
    return is_legacy, codes, not_real


# ---------------------------------------------------------------------------
# Preprocessing (bug 1 + bug 3)
# ---------------------------------------------------------------------------

class Preprocessor:
    """fit() only ever sees the training rows. transform() then gets used on
    val / test / new wafers with the numbers it learned from train."""

    def fit(self, frame):
        self.sensor_cols = get_sensor_cols(frame)
        X = frame[self.sensor_cols].to_numpy(np.float64).copy()
        _, _, not_real = legacy_view(frame, self.sensor_cols)
        X[not_real] = np.nan   # sentinel + error codes don't count towards any statistic

        missing_frac = np.isnan(X).mean(axis=0)
        std = np.nanstd(X, axis=0)
        # keep sensors that aren't mostly missing and actually change
        self.keep = np.where((missing_frac <= MAX_MISSING) & (np.nan_to_num(std) > 0))[0]

        Xk = X[:, self.keep]
        self.mean = np.nanmean(Xk, axis=0)
        self.std = np.nanstd(Xk, axis=0)
        self.median_z = (np.nanmedian(Xk, axis=0) - self.mean) / self.std

        # which sensors get a "was missing" flag (only count genuinely missing values)
        raw = frame[self.sensor_cols].to_numpy(np.float64)
        real_missing = (np.isnan(raw) & ~not_real)[:, self.keep].mean(axis=0)
        self.flag_cols = np.where(real_missing >= MIN_MISSING_FLAG)[0]

        self.n_cont = len(self.keep)
        self.feature_names = ([self.sensor_cols[k] for k in self.keep]
                              + ["missing_" + self.sensor_cols[self.keep[j]] for j in self.flag_cols]
                              + ["is_legacy", "s1_code_100", "s1_code_200", "s1_code_300", "s1_code_unknown"])
        return self

    def transform(self, frame):
        raw = frame[self.sensor_cols].to_numpy(np.float64)
        is_legacy, codes, not_real = legacy_view(frame, self.sensor_cols)

        Xk = raw[:, self.keep]
        not_real_k = not_real[:, self.keep]
        missing_k = np.isnan(Xk) & ~not_real_k

        Z = (Xk - self.mean) / self.std
        Z = np.clip(Z, -Z_CLIP, Z_CLIP)
        Z = np.where(missing_k, self.median_z, Z)   # fill real gaps with the TRAIN median
        Z = np.where(not_real_k, 0.0, Z)            # placeholder for sentinel/code slots,
                                                    # the loss ignores it (mask = 0)

        features = np.concatenate([
            Z,
            missing_k[:, self.flag_cols].astype(np.float64),
            is_legacy[:, None].astype(np.float64),
            codes,
        ], axis=1).astype(np.float32)

        # mask says which cells the autoencoder should try to rebuild:
        # real measured values yes, missing / placeholder values no
        mask = np.ones_like(features, dtype=np.float32)
        mask[:, :self.n_cont] = (~missing_k & ~not_real_k).astype(np.float32)

        assert np.isfinite(features).all()
        return features, mask


# ---------------------------------------------------------------------------
# Model + loss (bug 2)
# ---------------------------------------------------------------------------

def make_loader(X, M, y, shuffle, seed=SEED):
    # NOTE: no unsqueeze(1) here anymore. Features stay [N, F].
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(M), torch.from_numpy(y).float())
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      generator=g if shuffle else None, drop_last=False)


class WaferSAE(nn.Module):
    # same layers as the original notebook, I only made dropout a parameter
    def __init__(self, d_in, d_hid, d_lat, p_drop=0.10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.LayerNorm(d_hid), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(d_hid, d_hid // 2), nn.LayerNorm(d_hid // 2), nn.GELU(),
            nn.Linear(d_hid // 2, d_lat),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_lat, d_hid // 2), nn.GELU(),
            nn.Linear(d_hid // 2, d_hid), nn.GELU(),
            nn.Linear(d_hid, d_in),
        )
        self.head = nn.Linear(d_lat, 1)

    def forward(self, x):
        # the original did x.flatten(1) here which hid the shape problem. Now it just fails.
        assert x.dim() == 2, f"expected [B, F], got {tuple(x.shape)}"
        z = self.encoder(x)
        return self.decoder(z), self.head(z).squeeze(-1), z


def sae_loss(recon, x, mask, logits, y, pos_weight, lam=1.0):
    B = x.shape[0]
    # check shapes BEFORE subtracting so pytorch can't broadcast behind our back
    assert recon.shape == x.shape == mask.shape and x.dim() == 2, (recon.shape, x.shape, mask.shape)
    assert logits.shape == y.shape == (B,), (logits.shape, y.shape)

    recon_loss = ((recon - x) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)
    clf_loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
    return recon_loss + lam * clf_loss, recon_loss, clf_loss


@torch.no_grad()
def predict_proba(model, X):
    model.eval()
    return torch.sigmoid(model(torch.from_numpy(X))[1]).numpy()


def get_pos_weight(y):
    # (number of passes) / (number of fails) -> used instead of SMOTE
    return torch.tensor((1 - y).sum() / max(y.sum(), 1), dtype=torch.float32)


def train_sae(X_tr, M_tr, y_tr, X_va, y_va, seed=SEED, hidden=256, latent=32, weight_decay=1e-4,
              max_epochs=60, patience=10, lam=1.0, verbose=True):
    """Stage A training: original model size, early stopping on validation PR-AUC."""
    seed_all(seed)
    model = WaferSAE(X_tr.shape[1], hidden, latent)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    pos_weight = get_pos_weight(y_tr)
    loader = make_loader(X_tr, M_tr, y_tr, shuffle=True, seed=seed)

    best_ap, best_epoch, best_state = -1.0, 0, None
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        tot = rec = clf = 0.0
        for xb, mb, yb in loader:
            r, logit, _ = model(xb)
            loss, l_rec, l_clf = sae_loss(r, xb, mb, logit, yb, pos_weight, lam)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tot += loss.item() * len(xb)
            rec += l_rec.item() * len(xb)
            clf += l_clf.item() * len(xb)
        sched.step()

        val_ap = average_precision_score(y_va, predict_proba(model, X_va))
        history.append(dict(epoch=epoch, total=tot / len(X_tr), recon=rec / len(X_tr),
                            clf=clf / len(X_tr), val_ap=val_ap))
        if val_ap > best_ap:
            best_ap, best_epoch = val_ap, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and (epoch == 1 or epoch % 5 == 0):
            h = history[-1]
            print(f"epoch {epoch:02d} | total {h['total']:.4f} | recon {h['recon']:.4f} | "
                  f"clf {h['clf']:.4f} | val AP {val_ap:.4f}")
        if epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    if verbose:
        print(f"best epoch {best_epoch} (val AP {best_ap:.4f}), pos_weight = {pos_weight.item():.2f}")
    return model, pd.DataFrame(history)


def train_sae_fixed(X, M, y, cfg, seed):
    """Stage B training: fixed number of epochs, no early stopping (so it never looks
    at a later time window)."""
    seed_all(seed)
    model = WaferSAE(X.shape[1], cfg["hidden"], cfg["latent"], p_drop=cfg["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    pos_weight = get_pos_weight(y)
    loader = make_loader(X, M, y, shuffle=True, seed=seed)
    for _ in range(cfg["epochs"]):
        model.train()
        for xb, mb, yb in loader:
            r, logit, _ = model(xb)
            loss, _, _ = sae_loss(r, xb, mb, logit, yb, pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        sched.step()
    return model


# ---------------------------------------------------------------------------
# Stage B pipeline (preprocess -> pick features -> model), all fit on past data only
# ---------------------------------------------------------------------------

def select_features(X, y, k):
    # rank features by |correlation with the label| on the TRAIN rows only, keep top k.
    # the last 5 columns (legacy flag + codes) are always kept.
    if k is None:
        return np.arange(X.shape[1])
    body = X[:, :-5]
    yc = y - y.mean()
    xc = body - body.mean(axis=0)
    denom = np.sqrt((xc ** 2).sum(axis=0) * (yc ** 2).sum())
    corr = np.abs(np.divide(xc.T @ yc, denom, out=np.zeros(body.shape[1]), where=denom > 0))
    top = np.sort(np.argsort(-corr)[:k])
    return np.concatenate([top, np.arange(X.shape[1] - 5, X.shape[1])])


class WaferPipeline:
    def __init__(self, cfg, seeds):
        self.cfg = cfg
        self.seeds = list(seeds)

    def _features(self, frame):
        X, M = self.prep.transform(frame)
        X, M = X[:, self.cols], M[:, self.cols]
        if self.cfg.get("drop_legacy"):
            # ablation: throw away is_legacy + the 4 code columns
            X, M = X[:, :-5], M[:, :-5]
        return X, M

    def fit(self, frame, y):
        self.prep = Preprocessor().fit(frame)
        X_all, _ = self.prep.transform(frame)
        self.cols = select_features(X_all, y, self.cfg["k"])
        X, M = self._features(frame)

        if self.cfg.get("smote"):
            # SMOTE only on these training rows, after train-only preprocessing
            n_before = len(X)
            X, y = SMOTE(random_state=SEED, k_neighbors=min(5, int(y.sum()) - 1)).fit_resample(X, y)
            X = X.astype(np.float32)
            M = np.concatenate([M, np.ones((len(X) - n_before, M.shape[1]), np.float32)])

        if self.cfg["kind"] == "lr":
            weights = None if self.cfg.get("smote") else "balanced"
            self.models = [LogisticRegression(C=self.cfg["C"], class_weight=weights, max_iter=5000).fit(X, y)]
        else:
            self.models = [train_sae_fixed(X, M, y, self.cfg, s) for s in self.seeds]
        return self

    def member_proba(self, frame):
        X, _ = self._features(frame)
        if self.cfg["kind"] == "lr":
            return [self.models[0].predict_proba(X)[:, 1]]
        return [predict_proba(m, X) for m in self.models]

    def predict_proba(self, frame):
        return np.mean(self.member_proba(frame), axis=0)


def cv_predict(df, y_all, cv_edges, cfg, seeds=(0, 1, 2)):
    # expanding window: fold k trains on everything before cv_edges[k] and scores the next block
    ys, ps = [], []
    for k in range(len(cv_edges) - 1):
        past = np.arange(0, cv_edges[k])
        block = np.arange(cv_edges[k], cv_edges[k + 1])
        pipe = WaferPipeline(cfg, seeds).fit(df.iloc[past], y_all[past])
        ys.append(y_all[block])
        ps.append(pipe.predict_proba(df.iloc[block]))
    return np.concatenate(ys), np.concatenate(ps)


# ---------------------------------------------------------------------------
# Thresholds + metrics
# ---------------------------------------------------------------------------

def pick_threshold(y, p):
    # threshold with the best F1 on the data we're allowed to look at (val / CV)
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[np.argmax(f1)])


def evaluate(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    # the test window has its own defect rate, so I also work out what precision
    # would be at the dataset-wide 6.64% using Bayes rule (no resampling)
    pi = NATURAL_PREVALENCE
    prec_pi = tpr * pi / max(tpr * pi + fpr * (1 - pi), 1e-12)
    return dict(n=len(y), defects=int(y.sum()), prevalence=y.mean(),
                ap=average_precision_score(y, p), roc_auc=roc_auc_score(y, p),
                f1=f1_score(y, pred, zero_division=0),
                precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred), tp=tp, fp=fp, fn=fn, tn=tn,
                precision_at_6_64=prec_pi,
                f1_at_6_64=2 * prec_pi * tpr / max(prec_pi + tpr, 1e-12))


def show(r, title):
    print(f"--- {title} ---")
    print(f"n={r['n']}  defects={r['defects']}  defect rate={r['prevalence']:.2%}")
    print(f"PR-AUC {r['ap']:.4f} (random guess = {r['prevalence']:.4f})   ROC-AUC {r['roc_auc']:.4f}")
    print(f"F1 {r['f1']:.4f}   precision {r['precision']:.4f}   recall {r['recall']:.4f}")
    print(f"confusion: TN={r['tn']}  FP={r['fp']}  FN={r['fn']}  TP={r['tp']}")
    print(f"at the dataset-wide 6.64% defect rate: precision {r['precision_at_6_64']:.4f}, F1 {r['f1_at_6_64']:.4f}")
