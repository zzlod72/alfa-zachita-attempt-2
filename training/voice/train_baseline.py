"""
Предварительный эксперимент. Проверка на «шорткат» и бейзлайн на признаках.

1. Шорткат: бустинг только по метаданным исходного файла — частота
   дискретизации и длительность. Если такой «модели» хватает для высокой
   точности, значит, без предобработки нейросеть училась бы на них.
2. Бейзлайн: ~190 признаков на фрагмент (LFCC, MFCC, спектральные,
   просодия — см. features.py) → HistGradientBoostingClassifier.
3. Оценка на трёх тестах (знакомые / незнакомые / телефонный канал), EER по
   каждому отложенному генератору, важность групп признаков.

Запуск:
    python voice\\train_baseline.py              # полоса 16 кГц
    python voice\\train_baseline.py --sr 8000    # телефонная полоса 8 кГц

Результат:
    models/voice/baseline_{sr}.joblib
    reports/voice/baseline_{sr}.json и .md
"""

import argparse
import json
import os
import sys
import time

# Подавляет предупреждение joblib/loky о подсчёте физических ядер на Windows.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
from concurrent.futures import ProcessPoolExecutor

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import audioprep  # noqa: E402
import evalutil  # noqa: E402
import features  # noqa: E402
import index_ruasd  # noqa: E402

ROOT = os.path.join(HERE, "..", "..")
TESTS = ("test_seen", "test_unseen", "test_phone")


def load(prep_dir: str, split: str):
    meta = pd.read_parquet(os.path.join(prep_dir, f"{split}_meta.parquet"))
    wave = np.load(os.path.join(prep_dir, f"{split}_wave.npy"), mmap_mode="r")
    meta = meta[meta.ok].reset_index(drop=True)
    return meta, wave


def _feat_chunk(job):
    path, rows, sr = job
    wave = np.load(path, mmap_mode="r")
    out = []
    for r in rows:
        x = wave[r].astype(np.float32) / 32767
        if sr != audioprep.TARGET_SR:
            x, _ = audioprep.to_narrowband(x)
        out.append(features.extract(x, sr))
    return np.stack(out)


def feats(prep_dir, split, meta, sr, workers):
    cache = os.path.join(prep_dir, f"feats_{sr}_{split}.npy")
    if os.path.exists(cache):
        X = np.load(cache)
        if len(X) == len(meta):
            return X
    path = os.path.join(prep_dir, f"{split}_wave.npy")
    rows = meta.row.to_numpy()
    jobs = [(path, rows[i:i + 500], sr) for i in range(0, len(rows), 500)]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        X = np.concatenate(list(ex.map(_feat_chunk, jobs)))
    np.save(cache, X)
    print(f"  признаки {split}: {X.shape} за {time.time() - t0:.0f} с")
    return X


def hgb():
    return HistGradientBoostingClassifier(max_iter=600, learning_rate=0.06, max_leaf_nodes=63,
                                          l2_regularization=1.0, early_stopping=True,
                                          validation_fraction=0.1, random_state=42)


def shortcut_check(metas) -> dict:
    """Насколько угадывается класс только по исходной частоте и длительности."""
    cols = ["sr", "dur"]
    tr = metas["train"]
    m = hgb().fit(tr[cols].astype(float), tr.y)
    va = metas["val"]
    _, thr = evalutil.eer(va.y, m.predict_proba(va[cols].astype(float))[:, 1])
    out = {}
    for t in ("test_seen", "test_unseen"):
        if t in metas:
            mt = metas[t]
            out[t] = evalutil.report(mt.y, m.predict_proba(mt[cols].astype(float))[:, 1], thr)
    return out




def group_of(name: str) -> str:
    if name.startswith("lfcc"):
        return "LFCC (+Δ, ΔΔ)"
    if name.startswith("mfcc"):
        return "MFCC"
    if name in features.PROSODY_NAMES:
        return "просодия"
    return "спектральные"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=index_ruasd.DEFAULT_DIR)
    ap.add_argument("--sr", type=int, default=16000, choices=[16000, 8000])
    ap.add_argument("--workers", type=int, default=min(10, max(1, (os.cpu_count() or 4) - 2)))
    ap.add_argument("--exclude", nargs="*", default=[],
                    choices=["lfcc", "mfcc", "spectral", "prosody"],
                    help="абляция: исключить группы признаков (например, --exclude prosody)")
    a = ap.parse_args()
    prep = os.path.join(a.data_dir, "prepared")
    model_dir = os.path.join(ROOT, "models", "voice")
    rep_dir = os.path.join(ROOT, "training", "reports", "voice")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(rep_dir, exist_ok=True)

    metas, X = {}, {}
    for s in ("train", "val") + TESTS:
        if os.path.exists(os.path.join(prep, f"{s}_meta.parquet")):
            metas[s], _ = load(prep, s)

    print("1. Проверка на шорткат (только частота и длительность исходного файла)")
    shortcut = shortcut_check(metas)
    for t, r in shortcut.items():
        print(f"   {t:<12} EER {r.get('eer')}  ROC-AUC {r.get('roc_auc')}")

    print(f"2. Признаки, полоса {a.sr // 2000} кГц")
    for s in metas:
        X[s] = feats(prep, s, metas[s], a.sr, a.workers)

    names = features.feature_names()
    gmap = {"lfcc": "LFCC (+Δ, ΔΔ)", "mfcc": "MFCC", "spectral": "спектральные", "prosody": "просодия"}
    drop = {gmap[g] for g in a.exclude}
    keep = np.array([group_of(n) not in drop for n in names])
    names = [n for n, k in zip(names, keep) if k]
    X = {s: v[:, keep] for s, v in X.items()}
    if a.exclude:
        print(f"   абляция: исключены {', '.join(a.exclude)}; осталось признаков {keep.sum()}")

    print("3. Обучение HistGradientBoosting")
    t0 = time.time()
    model = hgb().fit(X["train"], metas["train"].y)
    fit_sec = time.time() - t0
    p_val = model.predict_proba(X["val"])[:, 1]
    val_eer, thr = evalutil.eer(metas["val"].y, p_val)
    print(f"   обучено за {fit_sec:.0f} с, итераций {model.n_iter_}; EER на val {val_eer:.4f}")

    results = {"val": evalutil.report(metas["val"].y, p_val, thr)}
    scores = {}
    for t in TESTS:
        if t in metas:
            scores[t] = model.predict_proba(X[t])[:, 1]
            results[t] = evalutil.report(metas[t].y, scores[t], thr)
            r = results[t]
            print(f"   {t:<12} EER {r.get('eer')}  ROC-AUC {r.get('roc_auc')}  "
                  f"ложные тревоги {r['false_alarm']}  пропуски {r['miss']}")

    per_source = {t: evalutil.by_source(metas[t], scores[t], thr)
                  for t in ("test_unseen", "test_phone") if t in scores}

    print("4. Важность признаков (перестановочная, на val)")
    rng = np.random.default_rng(0)
    sub = rng.choice(len(X["val"]), size=min(3000, len(X["val"])), replace=False)
    pi = permutation_importance(model, X["val"][sub], metas["val"].y.values[sub],
                                scoring="roc_auc", n_repeats=3, random_state=0, n_jobs=1)
    imp = pd.DataFrame({"feature": names, "importance": pi.importances_mean})
    imp["group"] = imp.feature.map(group_of)
    by_group = imp.groupby("group").importance.sum().sort_values(ascending=False)
    top = imp.sort_values("importance", ascending=False).head(20)
    print("   по группам:", {k: round(v, 4) for k, v in by_group.items()})

    tag = f"baseline_{a.sr}" + ("_no_" + "_".join(a.exclude) if a.exclude else "")
    split_rep = json.load(open(os.path.join(prep, "split_report.json"), encoding="utf-8"))
    joblib.dump({"model": model, "threshold": thr, "sr": a.sr,
                 "seconds": split_rep["seconds"], "feature_names": names},
                os.path.join(model_dir, f"{tag}.joblib"), compress=3)
    out = {"model": "HistGradientBoosting на признаках", "band_khz": a.sr // 2000,
           "excluded_groups": a.exclude, "n_features": int(len(names)),
           "threshold_from": "EER на val", "fit_sec": round(fit_sec, 1),
           "n_iter": int(model.n_iter_), "shortcut_check": shortcut, "results": results,
           "per_source": per_source,
           "importance_by_group": {k: round(float(v), 4) for k, v in by_group.items()},
           "top_features": [{"feature": r.feature, "importance": round(float(r.importance), 4)}
                            for r in top.itertuples()]}
    with open(os.path.join(rep_dir, f"{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    lines = [f"# Бейзлайн: признаки + бустинг, полоса {a.sr // 2000} кГц", "",
             "| Выборка | EER | ROC-AUC | Ложные тревоги | Пропуски |", "|---|---|---|---|---|"]
    for k, r in results.items():
        lines.append(f"| {k} | {r.get('eer', '—')} | {r.get('roc_auc', '—')} "
                     f"| {r['false_alarm']} | {r['miss']} |")
    lines += ["", "## Проверка на шорткат (только исходная частота и длительность)", "",
              "| Выборка | EER | ROC-AUC |", "|---|---|---|"]
    for k, r in shortcut.items():
        lines.append(f"| {k} | {r.get('eer')} | {r.get('roc_auc')} |")
    for t, rows in per_source.items():
        lines += ["", f"## {t}: по источникам", "",
                  "| Источник | Класс | n | EER | Ложные тревоги | Пропуски |",
                  "|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['source']} | {r['class']} | {r['n']} | {r.get('eer', '—')} "
                         f"| {r.get('false_alarm_at_thr', '—')} | {r.get('miss_at_thr', '—')} |")
    lines += ["", "## Важность групп признаков", ""]
    lines += [f"- {k}: {round(float(v), 4)}" for k, v in by_group.items()]
    with open(os.path.join(rep_dir, f"{tag}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nОтчёт: {os.path.join(rep_dir, tag + '.md')}")


if __name__ == "__main__":
    main()
