"""
Детектор синтезированной речи на ВСЕХ записях из скачанных архивов RuASD.

Разбиение 70 / 15 / 15:
    train — обучение модели;
    val   — выбор порога (точка EER) — модель его не видит при обучении;
    test  — итоговые цифры, считаются один раз.

Разбиение по исходной записи: запись и все её искажённые копии (кодеки, шум,
реверберация) всегда попадают в одну и ту же выборку. Иначе модель видела бы
в обучении ту же фразу, что потом проверяется в тесте, и цифры были бы
завышены. Выборка определяется хешем идентификатора записи, поэтому
результат не зависит от порядка файлов и воспроизводится на любом компьютере.

Запуск (из папки training):
    python voice\\train_full.py                # признаки + обучение + отчёт
    python voice\\train_full.py --holdout      # доп. эксперимент: 10 генераторов и
                                              # 2 корпуса целиком в отдельный тест

Признаки считаются один раз и кешируются (data/ruasd/full/), повторный
запуск с другими параметрами обучения занимает секунды.

Результат:
    models/voice/full_8000.joblib            — модель (для predict_voice.py)
    training/reports/voice/full_8000.md      — таблицы с результатами
    training/reports/voice/full_8000.json    — все цифры
"""

import argparse
import glob
import hashlib
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import audioprep  # noqa: E402
import evalutil  # noqa: E402
import features  # noqa: E402
import fullworker  # noqa: E402
import index_ruasd  # noqa: E402
from prepare_ruasd import HOLDOUT_GENS, HOLDOUT_REAL  # noqa: E402

ROOT = os.path.join(HERE, "..", "..")
SEED = 42
THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


# ------------------------------------------------------------------ признаки

def _stem(p) -> str:
    b = os.path.basename(str(p))
    return b.rsplit(".", 1)[0] if "." in b else b


def build_features(a) -> pd.DataFrame:
    out_dir = os.path.join(a.data_dir, "full")
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(out_dir, f"feats_{a.sr}_{a.seconds:g}s.parquet")

    tars = sorted(glob.glob(os.path.join(a.data_dir, "*.tar")))
    idx_path = os.path.join(a.data_dir, "ruasd_index.parquet")
    idx = pd.read_parquet(idx_path) if os.path.exists(idx_path) else pd.DataFrame()
    if idx.empty or set(idx.shard) != {os.path.basename(t) for t in tars}:
        print("Индекс устарел — перестраиваю…")
        sys.argv = ["index_ruasd.py", "--data-dir", a.data_dir]
        index_ruasd.main()
        idx = pd.read_parquet(idx_path)

    if os.path.exists(cache):
        df = pd.read_parquet(cache)
        if set(df.shard.unique()) == set(idx.shard.unique()):
            print(f"Признаки из кеша: {cache} ({len(df)} записей)")
            return df
        print("Кеш признаков не совпадает с архивами — пересчитываю.")

    df = idx[idx.member.notna() & idx.label.isin(["fake", "real"])].copy()
    df["group"] = df["group"].fillna("raw")
    df["dur"] = pd.to_numeric(df["wav_duration"], errors="coerce").fillna(
        pd.to_numeric(df["duration"], errors="coerce"))
    n_all = len(df)
    df = df[df.dur >= a.seconds + 0.4]
    print(f"Записей в архивах: {n_all}; длиннее {a.seconds + 0.4:g} с: {len(df)} "
          f"(короче — {n_all - len(df)}, отброшены)")

    parts_dir = os.path.join(out_dir, f"parts_{a.sr}_{a.seconds:g}s")
    os.makedirs(parts_dir, exist_ok=True)
    part_of = lambda sh: os.path.join(parts_dir, sh.replace(".tar", ".npz"))  # noqa: E731
    jobs = [(os.path.join(a.data_dir, sh), g.member.tolist(), a.seconds, a.sr, part_of(sh))
            for sh, g in df.groupby("shard") if not os.path.exists(part_of(sh))]
    done_before = df.shard.nunique() - len(jobs)
    if done_before:
        print(f"Уже посчитано архивов: {done_before} — продолжаю с остальных.")
    t0, skipped = time.time(), {}
    print(f"Считаю признаки: {len(jobs)} архивов, {a.workers} процессов…")
    # Процессам-обработчикам — по одному потоку BLAS/OpenMP: иначе каждый из
    # них резервирует память под десятки потоков и кончается файл подкачки.
    saved = {v: os.environ.get(v) for v in THREAD_VARS}
    os.environ.update({v: "1" for v in THREAD_VARS})
    try:
        with ProcessPoolExecutor(a.workers) as ex:
            futs = [ex.submit(fullworker.shard_job, j) for j in jobs]
            for k, f in enumerate(as_completed(futs), 1):
                sh, n_rows, sk = f.result()
                for key, v in sk.items():
                    skipped[key] = skipped.get(key, 0) + v
                el = time.time() - t0
                print(f"  [{k}/{len(jobs)}] {sh}: {n_rows} записей, прошло {el / 60:.1f} мин, "
                      f"осталось ~{el / k * (len(jobs) - k) / 60:.1f} мин", flush=True)
    finally:
        for v, old_val in saved.items():
            if old_val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = old_val
    if skipped:
        print("Пропущено:", skipped)

    fn = features.feature_names()
    parts = []
    for sh in sorted(df.shard.unique()):
        z = np.load(part_of(sh), allow_pickle=True)
        if len(z["members"]):
            part = pd.DataFrame(z["X"], columns=fn)
            part.insert(0, "member", z["members"].astype(str))
            part.insert(0, "shard", sh)
            parts.append(part)
    feat = pd.concat(parts, ignore_index=True)
    keep = ["shard", "member", "label", "group", "subset", "augmentation", "source_audio",
            "filename", "audio_relpath", "sample_id", "sr", "dur"]
    meta = df[[c for c in keep if c in df.columns]]
    out = meta.merge(feat, on=["shard", "member"], how="inner")
    out.to_parquet(cache, index=False)
    print(f"Признаки сохранены: {cache} ({len(out)} записей, {(time.time() - t0) / 60:.1f} мин)")
    print("Промежуточные файлы можно удалить:", parts_dir)
    return out


# ------------------------------------------------------------------ разбиение

def assign_groups(df: pd.DataFrame) -> pd.Series:
    """Идентификатор исходной записи: у искажённой копии — её исходник."""
    raw = df[df.group == "raw"]
    raw_id = raw["shard"] + "/" + raw["member"]
    canon = {}
    for col in ("member", "sample_id", "audio_relpath", "filename"):   # последний приоритетнее
        if col in raw.columns:
            st = raw[col].map(lambda v: _stem(v) if pd.notna(v) else None)
            canon.update({k: v for k, v in zip(st, raw_id) if k})
    own = df["shard"] + "/" + df["member"]
    grp = own.copy()
    if "source_audio" in df.columns:
        aug = df.group != "raw"
        linked = df.loc[aug, "source_audio"].map(lambda v: canon.get(_stem(v)) if pd.notna(v) else None)
        grp[aug] = linked.fillna(own[aug])
    return grp


def split_of(groups: pd.Series, seed: int) -> pd.Series:
    """70/15/15 по хешу группы: детерминированно и независимо от порядка."""
    u = groups.map(lambda g: int(hashlib.md5(f"{seed}:{g}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF)
    return pd.Series(np.select([u < 0.70, u < 0.85], ["train", "val"], "test"), index=groups.index)


# ------------------------------------------------------------------ метрики

def metrics(y, s, thr) -> dict:
    y, s = np.asarray(y), np.asarray(s)
    r = evalutil.report(y, s, thr)
    tn, fp = r["confusion_matrix"][0]
    fn, tp = r["confusion_matrix"][1]
    acc = (tp + tn) / max(1, len(y))
    r["accuracy"] = round(acc, 4)
    r["error_rate"] = round(1 - acc, 4)
    r["accuracy_95ci"] = [round(acc - 1.96 * np.sqrt(acc * (1 - acc) / len(y)), 4),
                          round(acc + 1.96 * np.sqrt(acc * (1 - acc) / len(y)), 4)]
    if 0 < y.sum() < len(y):
        r["balanced_accuracy"] = round((1 - r["false_alarm"] + 1 - r["miss"]) / 2, 4)
    return r


def bootstrap(y, s, thr, n=200, seed=0) -> dict:
    """95% интервалы по бутстрепу: как сильно метрики зависят от состава теста."""
    rng, y, s = np.random.default_rng(seed), np.asarray(y), np.asarray(s)
    acc, bal, eer = [], [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        yi, pi = y[i], s[i] >= thr
        acc.append((yi == pi).mean())
        if 0 < yi.sum() < len(yi):
            bal.append(((pi[yi == 1]).mean() + (~pi[yi == 0]).mean()) / 2)
            eer.append(evalutil.eer(yi, s[i])[0])
    q = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]  # noqa: E731
    sd = lambda v: round(float(np.std(v)), 4)  # noqa: E731
    return {"accuracy_ci": q(acc), "accuracy_sd": sd(acc), "balanced_accuracy_ci": q(bal),
            "balanced_accuracy_sd": sd(bal), "eer_ci": q(eer), "eer_sd": sd(eer), "resamples": n}


def per_source(meta, s, thr) -> list:
    rows = []
    for (lbl, src), g in meta.assign(s=s).groupby(["label", "src"]):
        flagged = float((g.s >= thr).mean())
        rows.append({"source": src, "class": "синтез" if lbl == "fake" else "живая", "n": int(len(g)),
                     "detect_rate" if lbl == "fake" else "false_alarm": round(flagged, 4)})
    return rows


# ------------------------------------------------------------------ отчёт

def pct(v):
    return "—" if v is None else f"{v * 100:.1f}%"


def write_md(r: dict, path: str):
    t = r["test"]
    L = [f"# Детектор синтезированной речи: все записи, разбиение 70/15/15", "",
         f"Модель: градиентный бустинг (HistGradientBoostingClassifier, scikit-learn "
         f"{r['environment']['sklearn']}), {r['model']['n_trees']} деревьев, "
         f"{r['model']['n_features']} признаков фрагмента {r['seconds']:g} с при {r['sr'] // 1000} кГц. "
         f"Веса классов сбалансированы. Порог {r['threshold']:.3f} — точка EER на val.", "",
         "## Данные", "",
         f"Архивов RuASD: {r['data']['archives']}, записей с признаками: {r['data']['records']} "
         f"(исходных {r['data']['raw']}, искажённых копий {r['data']['augmented']}). "
         f"Записи короче {r['seconds'] + 0.4:g} с отброшены: {r['data']['too_short']}.", "",
         "| Выборка | Живая речь | Синтез | Всего | Доля |", "|---|---|---|---|---|"]
    for s, v in r["sizes"].items():
        L.append(f"| {s} | {v['real']} | {v['fake']} | {v['real'] + v['fake']} | {v['share']:.1%} |")
    if r.get("holdout"):
        L += ["", f"Отдельный тест на незнакомых источниках: генераторы "
                  f"{', '.join(r['holdout']['gens'])}; корпуса {', '.join(r['holdout']['real'])}."]
    L += ["", "## Итоговые метрики (test)", "",
          "| Метрика | Значение | 95% интервал | Ст. отклонение (бутстреп) |", "|---|---|---|---|",
          f"| Точность (accuracy) | {pct(t['accuracy'])} | {pct(t['boot']['accuracy_ci'][0])}–{pct(t['boot']['accuracy_ci'][1])} | {t['boot']['accuracy_sd'] * 100:.2f} п.п. |",
          f"| Ошибка (1 − accuracy) | {pct(t['error_rate'])} | | |",
          f"| Сбалансированная точность | {pct(t['balanced_accuracy'])} | {pct(t['boot']['balanced_accuracy_ci'][0])}–{pct(t['boot']['balanced_accuracy_ci'][1])} | {t['boot']['balanced_accuracy_sd'] * 100:.2f} п.п. |",
          f"| EER | {t['eer']:.4f} | {t['boot']['eer_ci'][0]:.4f}–{t['boot']['eer_ci'][1]:.4f} | {t['boot']['eer_sd']:.4f} |",
          f"| ROC-AUC | {t['roc_auc']:.4f} | | |",
          f"| Precision (синтез) | {t['precision']:.4f} | | |",
          f"| Recall (синтез) | {t['recall']:.4f} | | |",
          f"| F1 | {t['f1']:.4f} | | |",
          f"| Ложные тревоги на живой речи | {pct(t['false_alarm'])} | | |",
          f"| Пропуски синтеза | {pct(t['miss'])} | | |", "",
          f"Матрица ошибок (строки — истина, столбцы — прогноз; живая / синтез): "
          f"{t['confusion_matrix']}.", "",
          "## Все выборки", "",
          "| Выборка | Accuracy | Сбаланс. точность | EER | ROC-AUC | Ложные тревоги | Пропуски |",
          "|---|---|---|---|---|---|---|"]
    for name, v in r["by_split"].items():
        L.append(f"| {name} | {pct(v['accuracy'])} | {pct(v.get('balanced_accuracy'))} | "
                 f"{v.get('eer', '—')} | {v.get('roc_auc', '—')} | {pct(v['false_alarm'])} | {pct(v['miss'])} |")
    L += ["", "## Test: чистые записи и искажённые копии", "",
          "| Часть | Записей | Accuracy | EER | Ложные тревоги | Пропуски |", "|---|---|---|---|---|---|"]
    for name, v in r["test_by_condition"].items():
        L.append(f"| {name} | {v['n']} | {pct(v['accuracy'])} | {v.get('eer', '—')} "
                 f"| {pct(v['false_alarm'])} | {pct(v['miss'])} |")
    gen = [x for x in r["test_per_source"] if x["class"] == "синтез" and x["n"] >= 30]
    if gen:
        d = np.array([x["detect_rate"] for x in gen])
        L += ["", "## Test: по источникам", "",
              f"Доля найденного синтеза по {len(d)} генераторам: {d.mean():.1%} ± {d.std(ddof=1):.1%} "
              f"(минимум {d.min():.1%}, максимум {d.max():.1%}).", "",
              "| Источник | Класс | Записей | Найдено синтеза / ложные тревоги |", "|---|---|---|---|"]
        for x in sorted(r["test_per_source"], key=lambda x: (x["class"], x.get("detect_rate", 1))):
            v = x.get("detect_rate", x.get("false_alarm"))
            L.append(f"| {x['source']} | {x['class']} | {x['n']} | {pct(v)} |")
    if r.get("unseen"):
        u = r["unseen"]
        L += ["", "## Незнакомые генераторы и корпуса (вне 70/15/15)", "",
              f"Записей: {u['n']}, accuracy {pct(u['accuracy'])}, сбалансированная точность "
              f"{pct(u.get('balanced_accuracy'))}, EER {u.get('eer')}, ROC-AUC {u.get('roc_auc')}, "
              f"ложные тревоги {pct(u['false_alarm'])}, пропуски {pct(u['miss'])}.", "",
              "| Источник | Класс | Записей | Найдено синтеза / ложные тревоги |", "|---|---|---|---|"]
        for x in sorted(r.get("unseen_per_source") or [],
                        key=lambda x: (x["class"], x.get("detect_rate", 1))):
            v = x.get("detect_rate", x.get("false_alarm"))
            L.append(f"| {x['source']} | {x['class']} | {x['n']} | {pct(v)} |")
    L += ["", "## Рабочие точки (test)", "",
          "| Порог | Найдено синтеза | Ложные тревоги | Accuracy |", "|---|---|---|---|"]
    for p in r["operating_points"]:
        L.append(f"| {p['threshold']:.2f} | {pct(p['detect_rate'])} | {pct(p['false_alarm'])} | {pct(p['accuracy'])} |")
    L += ["", "## Вклад групп признаков", "",
          "Падение ROC-AUC на test при перемешивании группы:", ""]
    L += [f"- {k}: {v}" for k, v in r["importance_by_group"].items()]
    L += ["", "## Воспроизводимость", "",
          f"seed {SEED}; разбиение по хешу идентификатора исходной записи; "
          f"архивов {r['data']['archives']}, отпечаток списка `{r['data']['archives_sha256_16']}`; "
          f"Python {r['environment']['python']}, scikit-learn {r['environment']['sklearn']}, "
          f"numpy {r['environment']['numpy']}; признаки {r['timing']['features_min']} мин, "
          f"обучение {r['timing']['fit_sec']} с.", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


# ------------------------------------------------------------------ main

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
    ap.add_argument("--sr", type=int, default=8000, choices=[8000, 16000])
    ap.add_argument("--seconds", type=float, default=3.0, help="длина фрагмента, с")
    ap.add_argument("--holdout", action="store_true",
                    help="10 генераторов и 2 корпуса целиком в отдельный тест (вне 70/15/15)")
    ap.add_argument("--workers", type=int, default=min(6, max(1, (os.cpu_count() or 4) - 2)),
                    help="процессов для расчёта признаков (при нехватке памяти уменьшите до 2–4)")
    a = ap.parse_args()

    t_feat = time.time()
    df = build_features(a)
    features_min = round((time.time() - t_feat) / 60, 1)

    df["y"] = (df.label == "fake").astype(int)
    df["src"] = np.where(df.group == "raw", df.subset.astype(str),
                         df.augmentation.astype(str))
    df["grp"] = assign_groups(df)
    df["split"] = split_of(df["grp"], SEED)

    unseen_mask = pd.Series(False, index=df.index)
    if a.holdout:
        hold_src = df.loc[df.group == "raw"].assign(s=df.subset.astype(str))
        held_groups = set(hold_src[hold_src.s.isin(HOLDOUT_GENS + HOLDOUT_REAL)]["grp"])
        unseen_mask = df.grp.isin(held_groups)
        df.loc[unseen_mask, "split"] = "unseen"

    fn = features.feature_names()
    X = {s: df.loc[df.split == s, fn].to_numpy(np.float32) for s in ("train", "val", "test", "unseen")}
    Y = {s: df.loc[df.split == s, "y"].to_numpy() for s in X}
    sizes = {}
    total = int((df.split != "unseen").sum())
    for s in ("train", "val", "test"):
        part = df[df.split == s]
        sizes[s] = {"real": int((part.y == 0).sum()), "fake": int((part.y == 1).sum()),
                    "share": len(part) / max(1, total)}
    print("\nРазбиение:", {s: v["real"] + v["fake"] for s, v in sizes.items()},
          "| незнакомые:", int(unseen_mask.sum()))

    model = HistGradientBoostingClassifier(
        max_iter=600, learning_rate=0.06, max_leaf_nodes=63, l2_regularization=1.0,
        class_weight="balanced", early_stopping=True, validation_fraction=0.1,
        random_state=SEED)
    t0 = time.time()
    model.fit(X["train"], Y["train"])
    fit_sec = round(time.time() - t0, 1)
    scores = {s: model.predict_proba(X[s])[:, 1] for s in X if len(X[s])}
    _, thr = evalutil.eer(Y["val"], scores["val"])
    print(f"Обучено за {fit_sec} с, деревьев {model.n_iter_}, порог {thr:.3f}")

    by_split = {s: metrics(Y[s], scores[s], thr) for s in ("train", "val", "test")}
    test = dict(by_split["test"])
    test["boot"] = bootstrap(Y["test"], scores["test"], thr)
    tmeta = df[df.split == "test"]
    cond = {}
    for name, m in (("чистые записи", tmeta.group.values == "raw"),
                    ("искажённые копии (кодеки, шум, реверберация)", tmeta.group.values != "raw")):
        if m.sum() and 0 < Y["test"][m].sum() < m.sum():
            cond[name] = metrics(Y["test"][m], scores["test"][m], thr)
    ops = []
    for p in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, round(thr, 3)):
        pr = scores["test"] >= p
        ops.append({"threshold": p, "detect_rate": float(pr[Y["test"] == 1].mean()),
                    "false_alarm": float(pr[Y["test"] == 0].mean()),
                    "accuracy": float((pr == (Y["test"] == 1)).mean())})
    ops.sort(key=lambda o: o["threshold"])

    # важность групп признаков на подвыборке test
    rng = np.random.default_rng(0)
    sub = rng.choice(len(X["test"]), size=min(8000, len(X["test"])), replace=False)
    Xs, ys = X["test"][sub], Y["test"][sub]
    base = roc_auc_score(ys, model.predict_proba(Xs)[:, 1])
    imp = {}
    for g in ("LFCC (+Δ, ΔΔ)", "MFCC", "спектральные", "просодия"):
        cols = [i for i, n in enumerate(fn) if group_of(n) == g]
        drops = []
        for _ in range(3):
            Xp = Xs.copy()
            Xp[:, cols] = Xp[rng.permutation(len(Xp))][:, cols]
            drops.append(base - roc_auc_score(ys, model.predict_proba(Xp)[:, 1]))
        imp[g] = round(float(np.mean(drops)), 4)

    shards = sorted(os.path.basename(p) for p in glob.glob(os.path.join(a.data_dir, "*.tar")))
    idx = pd.read_parquet(os.path.join(a.data_dir, "ruasd_index.parquet"))
    n_idx = int((idx.member.notna() & idx.label.isin(["fake", "real"])).sum())
    tag = f"full_{a.sr}" + ("_holdout" if a.holdout else "")
    r = {
        "model": {"type": "HistGradientBoostingClassifier", "n_trees": int(model.n_iter_),
                  "n_features": len(fn), "params": {k: v for k, v in model.get_params().items()
                                                    if k in ("max_iter", "learning_rate", "max_leaf_nodes",
                                                             "l2_regularization", "class_weight")}},
        "sr": a.sr, "seconds": a.seconds, "threshold": round(float(thr), 4),
        "data": {"archives": len(shards),
                 "archives_sha256_16": hashlib.sha256("\n".join(shards).encode()).hexdigest()[:16],
                 "records": int(len(df)), "raw": int((df.group == "raw").sum()),
                 "augmented": int((df.group != "raw").sum()), "too_short": n_idx - int(len(df))},
        "sizes": sizes, "by_split": by_split, "test": test, "test_by_condition": cond,
        "test_per_source": per_source(tmeta, scores["test"], thr),
        "operating_points": ops, "importance_by_group": imp,
        "holdout": {"gens": HOLDOUT_GENS, "real": HOLDOUT_REAL} if a.holdout else None,
        "unseen": metrics(Y["unseen"], scores["unseen"], thr) if a.holdout and len(X["unseen"]) else None,
        "unseen_per_source": (per_source(df[df.split == "unseen"], scores["unseen"], thr)
                              if a.holdout and len(X["unseen"]) else None),
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "numpy": np.__version__, "platform": platform.platform()},
        "timing": {"features_min": features_min, "fit_sec": fit_sec},
    }
    model_dir = os.path.join(ROOT, "models", "voice")
    rep_dir = os.path.join(ROOT, "training", "reports", "voice")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(rep_dir, exist_ok=True)
    joblib.dump({"model": model, "threshold": float(thr), "sr": a.sr, "seconds": a.seconds,
                 "feature_names": fn}, os.path.join(model_dir, f"{tag}.joblib"), compress=3)
    with open(os.path.join(rep_dir, f"{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2, default=float)
    write_md(r, os.path.join(rep_dir, f"{tag}.md"))

    t = r["test"]
    print(f"\nTEST: accuracy {t['accuracy']:.4f} (95% ДИ {t['boot']['accuracy_ci']}), "
          f"сбаланс. точность {t['balanced_accuracy']:.4f}, EER {t['eer']}, ROC-AUC {t['roc_auc']}, "
          f"F1 {t['f1']}")
    if r["unseen"]:
        u = r["unseen"]
        print(f"НЕЗНАКОМЫЕ: accuracy {u['accuracy']:.4f}, EER {u.get('eer')}, пропуски {u['miss']}")
    print(f"Отчёт: {os.path.join(rep_dir, tag + '.md')}")


if __name__ == "__main__":
    main()
