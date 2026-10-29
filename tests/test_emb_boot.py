import json

import numpy as np
import pandas as pd
import pytest

from cricmind import config
from cricmind.archetypes import emb_boot
from cricmind.archetypes import vectors as vec


def _p3_report(final):
    path = config.REPORTS / "p3" / "cric2vec.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"final": final}))


@pytest.fixture(autouse=True)
def _tmp_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPORTS", tmp_path / "rep")
    monkeypatch.setattr(config, "ARTIFACTS", tmp_path / "art")


def test_the_retrain_recipe_is_the_phase_3_final_and_refuses_the_ablation_inputs():
    _p3_report({"dim": 8, "epochs": 15, "ablation": True})
    with pytest.raises(ValueError, match="ablation"):
        emb_boot.recipe()


def test_run_retrains_each_missing_resample_with_the_final_recipe_and_resumes(monkeypatch):
    _p3_report({"dim": 8, "epochs": 15, "ablation": False})
    frame = pd.DataFrame({"match_id": np.repeat(np.arange(10), 3), "y": 0})
    calls = []

    def fit(res, competitions, dim, seed, epochs):
        calls.append((len(res) % 3, competitions, dim, seed, epochs))
        return None, []

    def save(net, res, dim, epochs, out):
        out.mkdir(parents=True)
        np.save(out / "embeddings.npy", np.zeros((1, dim)))

    monkeypatch.setattr(emb_boot, "load_train", lambda processed_dir=None: (frame, ["IPL"]))
    monkeypatch.setattr(emb_boot, "fit_net", fit)
    monkeypatch.setattr(emb_boot, "save_artifacts", save)
    save(None, None, 8, 15, vec.boot_dir(0))  # resample 0 already done by an earlier, interrupted run

    report = emb_boot.run(n_boot=3)

    assert calls == [(0, ["IPL"], 8, config.SEED, 15)] * 2  # resamples 1 and 2 only, whole matches
    assert set(report["seconds"]) == {"1", "2"} and report["resample_seeds"] == [42, 43, 44]
    assert (vec.boot_dir(2) / "embeddings.npy").exists()
