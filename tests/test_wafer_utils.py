# Tests for wafer_utils.py
# run from the repo root with:   pytest -q

import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import wafer_utils as wu

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "secom_telemetry_modified.csv")
MASK = os.path.join(ROOT, "secom_legacy_mask_EVALUATOR_ONLY.csv")


def tiny_frame():
    # 6 fake wafers, 4 sensors. rows 2 and 3 are legacy (sensor_0 == 0)
    return pd.DataFrame({
        "sensor_0": [3000.0, 3010.0, 0.0, 0.0, 2990.0, np.nan],
        "sensor_1": [2500.0, 2480.0, 100.0, 300.0, 2510.0, 2490.0],
        "sensor_2": [1.0, 2.0, 3.0, np.nan, 5.0, 6.0],
        "sensor_3": [7.0, 7.0, 7.0, 7.0, 7.0, 7.0],   # constant -> should be dropped
    })


@pytest.fixture(scope="module")
def real_df():
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    return df.sort_values("timestamp", kind="stable").reset_index(drop=True)


# ---------------- legacy probe station ----------------

def test_legacy_rows_and_codes():
    f = tiny_frame()
    is_legacy, codes, not_real = wu.legacy_view(f, wu.get_sensor_cols(f))
    assert is_legacy.tolist() == [False, False, True, True, False, False]
    assert codes[2].tolist() == [1, 0, 0, 0]   # code 100
    assert codes[3].tolist() == [0, 0, 1, 0]   # code 300
    assert codes[[0, 1, 4, 5]].sum() == 0
    # sensor_0 and sensor_1 are marked "not a real value" only on the legacy rows
    assert not_real[:, 0].tolist() == is_legacy.tolist()
    assert not_real[:, 1].tolist() == is_legacy.tolist()


def test_legacy_rule_on_real_data(real_df):
    is_legacy, codes, _ = wu.legacy_view(real_df, wu.get_sensor_cols(real_df))
    assert is_legacy.sum() == 146
    assert codes[:, :3].sum() == 146 and codes[:, 3].sum() == 0
    # error codes never show up on normal wafers
    assert not real_df.loc[~is_legacy, "sensor_1"].isin(wu.LEGACY_CODES).any()


@pytest.mark.skipif(not os.path.exists(MASK), reason="evaluator mask not in repo")
def test_legacy_rule_matches_evaluator_mask(real_df):
    # only used as a check, the pipeline never reads this file
    # (the mask uses row numbers of the original csv, which is already sorted by time)
    mask = pd.read_csv(MASK)
    is_legacy, _, _ = wu.legacy_view(real_df, wu.get_sensor_cols(real_df))
    assert set(mask["row"]) == set(np.where(is_legacy)[0])


# ---------------- preprocessing / causality ----------------

def test_codes_and_sentinel_not_used_for_scaling():
    f = tiny_frame()
    p = wu.Preprocessor().fit(f)
    names = p.feature_names
    i1 = names.index("sensor_1")
    i0 = names.index("sensor_0")
    # mean should only use the real readings (2500, 2480, 2510, 2490), not 100 / 300
    assert p.mean[i1] == pytest.approx(2495.0)
    assert p.mean[i0] == pytest.approx(3000.0)
    assert "sensor_3" not in names   # constant sensor dropped


def test_placeholder_is_masked_out():
    f = tiny_frame()
    p = wu.Preprocessor().fit(f)
    X, M = p.transform(f)
    i1 = p.feature_names.index("sensor_1")
    assert M[2, i1] == 0 and M[3, i1] == 0      # legacy rows: don't rebuild the code slot
    assert M[0, i1] == 1                        # normal row: do
    assert np.isfinite(X).all()


def test_preprocessor_only_learns_from_train(real_df):
    train_ix, _, test_ix = wu.time_split(len(real_df))
    p1 = wu.Preprocessor().fit(real_df.iloc[train_ix])

    # mess up the test rows completely, fit again on train -> must be identical
    messed = real_df.copy()
    cols = wu.get_sensor_cols(real_df)
    messed.loc[test_ix, cols] = messed.loc[test_ix, cols] * 1000 + 5
    p2 = wu.Preprocessor().fit(messed.iloc[train_ix])

    assert np.array_equal(p1.keep, p2.keep)
    assert np.allclose(p1.mean, p2.mean) and np.allclose(p1.std, p2.std)

    # and transforming one wafer doesn't depend on which other wafers are in the batch
    one = p1.transform(real_df.iloc[test_ix[:1]])[0]
    many = p1.transform(real_df.iloc[test_ix])[0][:1]
    assert np.array_equal(one, many)


def test_time_split_is_in_order(real_df):
    tr, va, te = wu.time_split(len(real_df))
    assert len(tr) + len(va) + len(te) == len(real_df)
    ts = real_df["timestamp"]
    assert ts[tr].max() < ts[va].min()
    assert ts[va].max() < ts[te].min()


# ---------------- tensor shapes / loss ----------------

def test_model_rejects_3d_input():
    m = wu.WaferSAE(10, 16, 4)
    with pytest.raises(AssertionError):
        m(torch.zeros(4, 1, 10))   # this is the shape the original notebook used


def test_loss_rejects_mismatched_shapes():
    x = torch.zeros(4, 1, 10)
    with pytest.raises(AssertionError):
        wu.sae_loss(torch.zeros(4, 10), x, torch.ones(4, 1, 10), torch.zeros(4), torch.zeros(4),
                    torch.tensor(1.0))


def test_masked_cells_dont_change_loss():
    torch.manual_seed(0)
    x = torch.randn(8, 5)
    recon = torch.randn(8, 5)
    mask = torch.ones(8, 5)
    mask[:, 0] = 0
    y = torch.zeros(8)
    logits = torch.zeros(8)
    l1 = wu.sae_loss(recon, x, mask, logits, y, torch.tensor(1.0))[1]
    recon2 = recon.clone()
    recon2[:, 0] += 100.0   # huge error but only in masked cells
    l2 = wu.sae_loss(recon2, x, mask, logits, y, torch.tensor(1.0))[1]
    assert torch.isclose(l1, l2)
