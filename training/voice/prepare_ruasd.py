"""
Предварительный эксперимент. Разбиение на выборки и подготовка фрагментов.

Разбиение — главное в этом шаге. Модель проверяется на трёх тестах:

  test_seen    — генераторы и корпуса, которые были в обучении (другие записи);
  test_unseen  — генераторы и корпуса, которых модель НЕ видела: облачные
                 сервисы, клонирование голоса, «токенные» модели и два корпуса
                 живой речи. Это главный тест — так выглядит реальная атака;
  test_phone   — записи, прогнанные через телефонные кодеки, шум и
                 реверберацию (авторы RuASD), оба класса.

Отложенные генераторы берутся целыми семействами: например, всё семейство
XTTS (coquiTTS, ruxTTS, xtts_ru_ipa), иначе «незнакомый» генератор был бы
близким родственником знакомого.

Обработка каждой записи (audioprep.py): ресемплинг в 16 кГц → обрезка
тишины → нормализация громкости → фрагмент фиксированной длины из середины
(по умолчанию 3 с). Записи короче фрагмента отбрасываются, а не дополняются.

Запуск:
    python voice\\prepare_ruasd.py                 # по умолчанию
    python voice\\prepare_ruasd.py --per-class 10000 --workers 4

Результат (data/ruasd/prepared/):
    {split}_wave.npy    — int16 [записи, отсчёты], 16 кГц
    {split}_meta.parquet — метка (1 = синтез), источник, исходная частота…
    split_report.json   — состав выборок
"""

import argparse
import glob
import json
import os
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import audioprep  # noqa: E402
import index_ruasd  # noqa: E402

# Отложенные генераторы: облачные API, семейство XTTS (клонирование голоса),
# GPT-SoVITS (клонирование), Fish-Speech и Bark (генерация аудиотокенов).
HOLDOUT_GENS = ["11labs", "edge_tts", "sberTTS", "vkTTS",
                "coquiTTS", "ruxTTS", "xtts_ru_ipa",
                "vosk-gpt-sovits", "fishsTTS", "bark_small_tts"]
# Отложенные корпуса живой речи: OpenSTT (радио, YouTube, телефония) и SOVA-YT.
HOLDOUT_REAL = ["OpenSTT", "SOVA-YT"]
SPLITS = ("train", "val", "test_seen", "test_unseen", "test_phone")


def waterfill(avail: dict, total: int) -> dict:
    """Делит total между источниками максимально поровну, не превышая
    доступного: маленькие источники берутся целиком, остаток — большим."""
    alloc, left, items = {}, total, sorted(avail.items(), key=lambda kv: kv[1])
    for i, (k, v) in enumerate(items):
        share = left // (len(items) - i)
        alloc[k] = min(v, share)
        left -= alloc[k]
    return alloc


def stem(p) -> str:
    b = os.path.basename(str(p))
    return b.rsplit(".", 1)[0] if "." in b else b


def build_splits(df: pd.DataFrame, a) -> pd.DataFrame:
    df = df.sample(frac=1, random_state=a.seed).reset_index(drop=True)
    df["split"] = None
    raw, aug = df[df.group == "raw"], df[df.group != "raw"]

    # --- незнакомые генераторы и корпуса
    for lbl, names, cap in (("fake", a.holdout_gens, a.unseen_per_gen),
                            ("real", a.holdout_real, a.unseen_per_corpus)):
        part = raw[(raw.label == lbl) & raw.source.isin(names)]
        for _, g in part.groupby("source"):
            df.loc[g.index[:cap], "split"] = "test_unseen"

    # --- знакомые: поровну по источникам, затем 75/10/15
    seen = raw[~raw.source.isin(a.holdout_gens + a.holdout_real)]
    avail = {lbl: seen[seen.label == lbl].source.value_counts().to_dict() for lbl in ("fake", "real")}
    n_class = min(a.per_class, sum(avail["fake"].values()), sum(avail["real"].values()))
    for lbl in ("fake", "real"):
        for src, k in waterfill(avail[lbl], n_class).items():
            idx = seen[(seen.label == lbl) & (seen.source == src)].index[:k]
            n_tr, n_va = int(len(idx) * 0.75), int(len(idx) * 0.10)
            df.loc[idx[:n_tr], "split"] = "train"
            df.loc[idx[n_tr:n_tr + n_va], "split"] = "val"
            df.loc[idx[n_tr + n_va:], "split"] = "test_seen"

    # --- телефонный канал: исключаем искажённые копии записей из обучения
    train_stems = set()
    for col in ("filename", "audio_relpath", "sample_id", "member"):
        train_stems |= set(df.loc[df.split == "train", col].dropna().map(stem))
    in_train = aug["source_audio"].dropna().map(stem).isin(train_stems)
    df["src_in_train"] = False
    df.loc[in_train[in_train].index, "src_in_train"] = True
    ok_aug = aug[~df.loc[aug.index, "src_in_train"]]
    for _, g in ok_aug.groupby(["source", "label"]):
        df.loc[g.index[:a.phone_per_cond], "split"] = "test_phone"
    print(f"Искажённых записей, чей исходник попал в обучение (исключены): {int(in_train.sum())}")
    all_stems = set()
    for col in ("filename", "audio_relpath", "sample_id", "member"):
        all_stems |= set(df.loc[df.group == "raw", col].dropna().map(stem))
    src = aug["source_audio"].dropna()
    found = src.map(stem).isin(all_stems).mean() if len(src) else 0
    print(f"  исходник находится среди скачанных исходных записей у {found:.0%} искажённых")
    if len(src) and found == 0:
        print("  пример source_audio:", src.iloc[0])
        print("  пример filename исходной записи:", df.loc[df.group == "raw", "filename"].dropna().iloc[0])
    return df[df.split.notna()].copy()


def process_shard(job):
    """Обрабатывает записи одного архива и пишет фрагменты прямо в файлы
    выборок (memmap). Возвращает только флаги успеха: гонять сотни мегабайт
    аудио между процессами через pickle на Windows — лишняя нагрузка."""
    path, items, n, out_dir = job
    want = {member: (split, row) for member, split, row in items}
    arrays, done = {}, []
    with tarfile.open(path, "r") as tar:
        for m in tar:
            if m.name not in want:
                continue
            split, row = want[m.name]
            try:
                x, sr = audioprep.read_wav(tar.extractfile(m).read())
                c = audioprep.center_crop(audioprep.preprocess(x, sr), n)
            except Exception:  # noqa: BLE001
                c = None
            if c is not None:
                if split not in arrays:
                    arrays[split] = np.load(os.path.join(out_dir, f"{split}_wave.npy"),
                                            mmap_mode="r+")
                arrays[split][row] = (np.clip(c, -1, 1) * 32767).astype(np.int16)
            done.append((m.name, c is not None))
    for arr in arrays.values():
        arr.flush()
    del arrays
    return os.path.basename(path), done


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=index_ruasd.DEFAULT_DIR)
    ap.add_argument("--seconds", type=float, default=3.0, help="длина фрагмента, с")
    ap.add_argument("--per-class", type=int, default=22000,
                    help="записей каждого класса в train+val+test_seen")
    ap.add_argument("--unseen-per-gen", type=int, default=400)
    ap.add_argument("--unseen-per-corpus", type=int, default=2000)
    ap.add_argument("--phone-per-cond", type=int, default=1000)
    ap.add_argument("--holdout-gens", nargs="*", default=HOLDOUT_GENS)
    ap.add_argument("--holdout-real", nargs="*", default=HOLDOUT_REAL)
    ap.add_argument("--workers", type=int, default=min(10, max(1, (os.cpu_count() or 4) - 2)),
                    help="число процессов (по умолчанию до 10; при ошибке WinError 1450 уменьшите)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    tars = sorted(glob.glob(os.path.join(a.data_dir, "*.tar")))
    idx_path = os.path.join(a.data_dir, "ruasd_index.parquet")
    df = pd.read_parquet(idx_path) if os.path.exists(idx_path) else pd.DataFrame()
    if df.empty or set(df.shard) != {os.path.basename(t) for t in tars}:
        print("Индекс устарел — перестраиваю…")
        sys.argv = ["index_ruasd.py", "--data-dir", a.data_dir]
        index_ruasd.main()
        df = pd.read_parquet(idx_path)

    n_samples = int(a.seconds * audioprep.TARGET_SR)
    df = df[df.member.notna()].copy()
    df["group"] = df["group"].fillna("raw")
    df["source"] = np.where(df.group == "raw", df.subset.astype(str), df.augmentation.astype(str))
    df["dur"] = pd.to_numeric(df["wav_duration"], errors="coerce").fillna(
        pd.to_numeric(df["duration"], errors="coerce"))
    df = df[df.dur >= a.seconds + 0.4]       # запас на обрезку тишины

    missing = [g for g in a.holdout_gens + a.holdout_real if g not in set(df.source)]
    if missing:
        print("ВНИМАНИЕ: в данных нет отложенных источников:", ", ".join(missing))

    sel = build_splits(df, a)
    out_dir = os.path.join(a.data_dir, "prepared")
    os.makedirs(out_dir, exist_ok=True)

    # Место под фрагменты: файлы .npy создаются заранее и сразу закрываются,
    # процессы-обработчики пишут в них сами.
    sel["row"] = sel.groupby("split").cumcount()
    for s in SPLITS:
        n_rows = int((sel.split == s).sum())
        if n_rows:
            arr = np.lib.format.open_memmap(os.path.join(out_dir, f"{s}_wave.npy"), mode="w+",
                                            dtype=np.int16, shape=(n_rows, n_samples))
            arr.flush()
            del arr
    ok = pd.Series(False, index=sel.index)
    where = {(r.shard, r.member): i for i, r in zip(sel.index, sel.itertuples())}

    jobs = [(os.path.join(a.data_dir, sh), list(zip(g.member, g.split, g.row)), n_samples, out_dir)
            for sh, g in sel.groupby("shard")]
    print(f"Обрабатываю {len(sel)} записей из {len(jobs)} архивов в {a.workers} процессах…")
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(process_shard, j) for j in jobs]
        for k, f in enumerate(as_completed(futs), 1):
            sh, done = f.result()
            for member, good in done:
                if good:
                    ok[where[(sh, member)]] = True
            print(f"  [{k}/{len(jobs)}] {sh}")

    sel["ok"] = ok
    sel["y"] = (sel.label == "fake").astype(int)
    cols = ["row", "ok", "y", "label", "group", "source", "sr", "dur", "shard", "member",
            "sample_id", "src_in_train"]
    report = {"seconds": a.seconds, "sample_rate": audioprep.TARGET_SR,
              "holdout_gens": a.holdout_gens, "holdout_real": a.holdout_real, "splits": {}}
    print("\nСостав выборок (только успешно обработанные):")
    for s in SPLITS:
        part = sel[(sel.split == s)]
        if part.empty:
            continue
        part[cols].to_parquet(os.path.join(out_dir, f"{s}_meta.parquet"), index=False)
        good = part[part.ok]
        by = good.groupby("label").size().to_dict()
        report["splits"][s] = {
            "real": int(by.get("real", 0)), "fake": int(by.get("fake", 0)),
            "failed": int((~part.ok).sum()),
            "sources": {f"{lbl}:{src}": int(n) for (lbl, src), n in
                        good.groupby(["label", "source"]).size().items()},
        }
        print(f"  {s:<12} живая {by.get('real', 0):>6}  синтез {by.get('fake', 0):>6}  "
              f"источников {good.source.nunique():>3}  отброшено {(~part.ok).sum()}")
    with open(os.path.join(out_dir, "split_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nГотово: {out_dir}")


if __name__ == "__main__":
    main()
