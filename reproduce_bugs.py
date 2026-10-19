# reproduce_bugs.py
# Re-creates the 3 bugs from the original notebook (bnypi.ipynb) and prints the numbers
# I quote in AUDIT.md. Run it from the repo folder:  python reproduce_bugs.py  (~2 min)
import sys, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, average_precision_score
from imblearn.over_sampling import SMOTE

ROOT = Path(__file__).resolve().parent
log = print

SEED = 1337
def seed_all(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

HP = dict(batch_size=64, epochs=40, lr=1e-3, weight_decay=1e-5, hidden=256, latent=32)

df = pd.read_csv(ROOT / "secom_telemetry_modified.csv")
SENSOR_COLS = [c for c in df.columns if c.startswith("sensor_")]
X_raw = df[SENSOR_COLS].to_numpy(np.float64)
y_raw = (df["label"].to_numpy() == 1).astype(np.int64)


class WaferSAE(nn.Module):  # identical to the original notebook
    def __init__(self, d_in, d_hid, d_lat):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.LayerNorm(d_hid), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(d_hid, d_hid // 2), nn.LayerNorm(d_hid // 2), nn.GELU(),
            nn.Linear(d_hid // 2, d_lat))
        self.decoder = nn.Sequential(
            nn.Linear(d_lat, d_hid // 2), nn.GELU(),
            nn.Linear(d_hid // 2, d_hid), nn.GELU(),
            nn.Linear(d_hid, d_in))
        self.head = nn.Linear(d_lat, 1)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        z = self.encoder(x)
        return self.decoder(z), self.head(z).squeeze(-1), z


def train_eval(X_tr, y_tr, X_te, y_te, broadcast_bug):
    seed_all()
    Xt = torch.from_numpy(X_tr).float().unsqueeze(1)          # [N,1,F] as in the original
    g = torch.Generator().manual_seed(SEED)
    dl = DataLoader(TensorDataset(Xt, torch.from_numpy(y_tr).float()),
                    batch_size=HP["batch_size"], shuffle=True, generator=g, drop_last=True)
    m = WaferSAE(X_tr.shape[1], HP["hidden"], HP["latent"])
    opt = torch.optim.AdamW(m.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=HP["epochs"])
    bce = nn.BCEWithLogitsLoss()
    for _ in range(HP["epochs"]):
        m.train()
        for xb, yb in dl:
            rec, logit, _ = m(xb)
            target = xb if broadcast_bug else xb.flatten(1)
            loss = torch.mean((rec - target) ** 2) + bce(logit, yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        sched.step()
    m.eval()
    with torch.no_grad():
        p = torch.sigmoid(m(torch.from_numpy(X_te).float())[1]).numpy()
    return f1_score(y_te, p >= 0.5, zero_division=0), average_precision_score(y_te, p), y_te.mean()


# ---------------------------------------------------------------- BUG 2: broadcasting
log("=" * 78); log("BUG 2 - silent broadcasting in the reconstruction loss"); log("=" * 78)
seed_all()
m = WaferSAE(len(SENSOR_COLS), 256, 32)
xb = torch.randn(64, 1, len(SENSOR_COLS))
rec, logits, _ = m(xb)
log(f"batch_x shape        : {tuple(xb.shape)}")
log(f"reconstructed shape  : {tuple(rec.shape)}")
log(f"(reconstructed - batch_x) shape : {tuple((rec - xb).shape)}   <-- [B,B,F], no error raised")
log(f"(reconstructed - batch_x.flatten(1)) shape : {tuple((rec - xb.flatten(1)).shape)}")
bug = torch.mean((rec - xb) ** 2); ok = torch.mean((rec - xb.flatten(1)) ** 2)
log(f"buggy recon loss {bug.item():.4f} vs correct {ok.item():.4f}")
# what the buggy loss actually optimises: every output row is pulled toward the BATCH MEAN
gb = torch.autograd.grad(bug, rec, retain_graph=True)[0]
gc = torch.autograd.grad(ok, rec)[0]
cos = nn.functional.cosine_similarity(gb.flatten(), gc.flatten(), dim=0).item()
log(f"cosine(grad_buggy, grad_correct) w.r.t. reconstruction = {cos:.4f}")
analytic = 2 * (rec - xb.flatten(1).mean(0, keepdim=True)) / rec.numel()
log(f"buggy grad == grad of MSE to the batch-mean target: "
    f"{torch.allclose(gb, analytic, atol=1e-7)}  -> decoder learns to output the batch mean, "
    f"not the input")

# ---------------------------------------------------------------- BUG 3: invariant blindness
log(""); log("=" * 78); log("BUG 3 - legacy Probe Station PS-L invariants ignored"); log("=" * 78)
legacy = df["sensor_0"].to_numpy() == 0.0
log(f"wafers with sensor_0 == 0.0 (legacy sentinel): {legacy.sum()} / {len(df)}")
log(f"sensor_1 values on legacy wafers: {df.loc[legacy, 'sensor_1'].value_counts().to_dict()}")
log(f"sensor_1 codes on non-legacy wafers: {df.loc[~legacy, 'sensor_1'].isin([100, 200, 300]).sum()}")
mask_file = ROOT / "secom_legacy_mask_EVALUATOR_ONLY.csv"
if mask_file.exists():   # verification only - never used by the pipeline
    ev = pd.read_csv(mask_file)
    log(f"rule-based legacy rows == evaluator mask rows: {set(ev['row']) == set(np.where(legacy)[0])}; "
        f"codes match: {np.array_equal(ev['sensor_1_code'].to_numpy(), df.loc[ev['row'], 'sensor_1'].to_numpy())}")
s1 = df["sensor_1"]
log(f"sensor_1 mean/std genuine-only : {s1[~legacy].mean():.2f} / {s1[~legacy].std():.2f}")
log(f"sensor_1 mean/std as original  : {s1.mean():.2f} / {s1.std():.2f}   (codes mixed in)")
log(f"sensor_0 mean/std genuine-only : {df.sensor_0[~legacy].mean():.2f} / {df.sensor_0[~legacy].std():.2f}")
log(f"sensor_0 mean/std as original  : {df.sensor_0.mean():.2f} / {df.sensor_0.std():.2f}")
imp = SimpleImputer(strategy="mean").fit(X_raw[:, :2])
log(f"original imputer fills NaN sensor_1 with {imp.statistics_[1]:.2f} "
    f"(a value no real station produces; genuine mean is {s1[~legacy].mean():.2f})")
sc = StandardScaler().fit(SimpleImputer(strategy="mean").fit_transform(X_raw[:, :2]))
z = sc.transform(np.array([[s1[~legacy].quantile(.05), s1[~legacy].quantile(.95)]]).T.repeat(2, 1))[:, 1]
log(f"after original scaling, genuine sensor_1 5th..95th pct spans only {z[1]-z[0]:.3f} sigma "
    f"(would be ~3.3 sigma if scaled on genuine data): real signal is compressed ~"
    f"{s1.std()/s1[~legacy].std():.1f}x")
codes_z = sc.transform(np.array([[0, 100], [0, 200], [0, 300]]))[:, 1]
log(f"codes 100/200/300 become ordinal z-scores {np.round(codes_z, 2).tolist()} "
    f"-> categorical firmware codes treated as magnitudes")

# ---------------------------------------------------------------- BUG 1: leakage ablation
log(""); log("=" * 78); log("BUG 1 - leakage + balanced test set: ablation (F1 @0.5, AP)"); log("=" * 78)
rows = []

# A: original pipeline exactly
Xs = StandardScaler().fit_transform(SimpleImputer(strategy="mean", keep_empty_features=True).fit_transform(X_raw))
Xb, yb = SMOTE(random_state=SEED).fit_resample(Xs, y_raw)
a, b, c, d = train_test_split(Xb, yb, test_size=0.2, stratify=yb, random_state=SEED)
rows.append(("A original (fit-all, SMOTE->split, broadcast bug)", *train_eval(a, c, b, d, True)))

# B: same leakage, broadcast bug fixed -> isolates leakage
rows.append(("B original leakage, broadcast fixed", *train_eval(a, c, b, d, False)))

# C: random split FIRST, prep + SMOTE on train only, natural-prevalence test
i_tr, i_te = train_test_split(np.arange(len(y_raw)), test_size=0.2, stratify=y_raw, random_state=SEED)
imp = SimpleImputer(strategy="mean", keep_empty_features=True).fit(X_raw[i_tr])
sc = StandardScaler().fit(imp.transform(X_raw[i_tr]))
tr = sc.transform(imp.transform(X_raw[i_tr])); te = sc.transform(imp.transform(X_raw[i_te]))
trb, ytrb = SMOTE(random_state=SEED).fit_resample(tr, y_raw[i_tr])
rows.append(("C random split first, train-only fit, natural test", *train_eval(trb, ytrb, te, y_raw[i_te], False)))

# D: chronological split (oldest 80% -> train, newest 20% -> test)
cut = int(0.8 * len(y_raw)); i_tr, i_te = np.arange(cut), np.arange(cut, len(y_raw))
imp = SimpleImputer(strategy="mean", keep_empty_features=True).fit(X_raw[i_tr])
sc = StandardScaler().fit(imp.transform(X_raw[i_tr]))
tr = sc.transform(imp.transform(X_raw[i_tr])); te = sc.transform(imp.transform(X_raw[i_te]))
trb, ytrb = SMOTE(random_state=SEED).fit_resample(tr, y_raw[i_tr])
rows.append(("D chronological split, train-only fit, natural test", *train_eval(trb, ytrb, te, y_raw[i_te], False)))

# how many test points in A are SMOTE synthetics / near-duplicates of training points
n_syn = len(yb) - len(y_raw)
log(f"SMOTE created {n_syn} synthetic defects; test set A defect rate = {d.mean():.2%}")
from sklearn.neighbors import NearestNeighbors
nn_ = NearestNeighbors(n_neighbors=1).fit(a)
dist_te = nn_.kneighbors(b)[0].ravel()
log(f"test defects in A: median distance to nearest TRAIN point = {np.median(dist_te[d == 1]):.2f}, "
    f"test passes: {np.median(dist_te[d == 0]):.2f}  (synthetic defects sit between training defects)")
log("")
log(f"{'scenario':<55} {'F1@0.5':>7} {'AP':>7} {'test prev':>9}")
for r in rows:
    log(f"{r[0]:<55} {r[1]:7.3f} {r[2]:7.3f} {r[3]:9.2%}")
log("\nAP of a random classifier equals the test prevalence.")
