"""
Карта датасета RuASD без скачивания архивов целиком.

Первый архив показал, что записи в RuASD лежат не вперемешку, а по порядку
источников: в архиве 000000 только два генератора и ни одной живой записи.
Значит, «каждый шестой архив» может пропустить целые генераторы и корпуса.

Этот скрипт для каждого из 250 архивов читает по HTTP только заголовок tar
и метаданные первой записи (несколько килобайт вместо гигабайта) и строит
карту: какой источник с какого архива начинается. По карте подбирается
список архивов, где представлен каждый генератор, каждый корпус живой речи
и искажённые (телефонные) записи.

Запуск:
    python voice\\map_ruasd.py                 # карта + выбор архивов
    python voice\\map_ruasd.py --budget 60     # больше архивов (остаток — живой речи
                                               # и крупным генераторам)

Результат (в data/ruasd/):
    shard_map.csv        — архив → метка, группа, источник первой записи
    chosen_shards.txt    — список архивов для download_ruasd.py --list
"""

import argparse
import json
import math
import os
import sys
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import index_ruasd  # noqa: E402

BASE_URL = "https://huggingface.co/datasets/{repo}/resolve/main/{name}"
TOKEN = os.environ.get("HF_TOKEN")


def http_range(url: str, start: int, length: int, tries: int = 4) -> bytes:
    """Чтение диапазона байтов. HF перенаправляет на CDN, заголовок Range
    при редиректе сохраняется."""
    headers = {"Range": f"bytes={start}-{start + length - 1}",
               "User-Agent": "alfa-zashchita-coursework"}
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                        timeout=30) as r:
                return r.read()
        except Exception:  # noqa: BLE001
            if attempt == tries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    return b""


def first_meta(read, max_members: int = 12) -> dict | None:
    """Идёт по заголовкам tar (по 512 байт), перепрыгивая через содержимое
    файлов, пока не встретит первый JSON с полем label."""
    off = 0
    for _ in range(max_members):
        hdr = read(off, 512)
        if len(hdr) < 512 or hdr == b"\0" * 512:
            return None
        info = tarfile.TarInfo.frombuf(hdr, "utf-8", "surrogateescape")
        data_off = off + 512
        if info.name.lower().endswith(".json") and info.isfile():
            try:
                d = json.loads(read(data_off, info.size))
            except Exception:  # noqa: BLE001
                d = None
            if isinstance(d, list) and d:
                d = d[0]
            if isinstance(d, dict) and "label" in d:
                return d
        off = data_off + math.ceil(info.size / 512) * 512
    return None


def probe(repo: str, name: str) -> dict:
    url = BASE_URL.format(repo=repo, name=name)
    try:
        d = first_meta(lambda o, n: http_range(url, o, n)) or {}
        err = "" if d else "нет метаданных в начале архива"
    except Exception as exc:  # noqa: BLE001
        d, err = {}, f"{type(exc).__name__}: {exc}"[:120]
    return {"shard": name, "label": d.get("label"), "group": d.get("group") or "raw",
            "subset": d.get("subset"), "augmentation": d.get("augmentation"),
            "model": d.get("model"), "error": err}


def segments(m: pd.DataFrame) -> pd.DataFrame:
    """Подряд идущие архивы с одинаковым источником первой записи
    (для искажённых записей источник — тип искажения)."""
    src = np.where(m["group"] != "raw", m["augmentation"].astype(str), m["subset"].astype(str))
    key = m["label"].astype(str) + "|" + m["group"].astype(str) + "|" + pd.Series(src, index=m.index)
    m = m.assign(seg=(key != key.shift()).cumsum())
    return m


# Узкополосные «телефонные» условия в порядке предпочтения.
PHONE_CONDITIONS = ["codec_alaw", "codec_speex_8khz", "noise_codec_mulaw",
                    "reverb_codec_opus_8khz", "codec_mulaw", "codec_opus_8khz",
                    "noise_codec_alaw", "noise_codec_speex_8khz", "reverb_codec_mulaw"]


def choose(m: pd.DataFrame, budget: int | None = None, aug_pairs: int = 3,
           have: set | None = None) -> list[str]:
    """Выбор архивов.

    Исходные записи: каждому источнику (генератору или корпусу) — один архив,
    из середины его диапазона. Если бюджет больше — остаток живой речи,
    потом крупным генераторам.

    Искажённые записи: «пары» — для одного и того же телефонного кодека
    архив с живой речью и архив с синтезом. Без пары метрику на телефонном
    канале не посчитать: в тесте должны быть оба класса.

    Уже скачанные архивы (have) используются в первую очередь, чтобы при
    повторном выборе не качать лишнее.
    """
    have = have or set()
    pick = lambda sh: next((x for x in sh if x in have), sh[len(sh) // 2])  # noqa: E731
    m = m.dropna(subset=["label"])
    aug = m[m["group"] != "raw"]
    raw = segments(m[m["group"] == "raw"])
    chosen = []

    by_cond = aug.groupby("augmentation")["label"].apply(set)
    paired = [c for c in PHONE_CONDITIONS if c in by_cond and by_cond[c] >= {"real", "fake"}]
    for c in paired[:aug_pairs]:
        for lbl in ("real", "fake"):
            chosen.append(pick(list(aug[(aug["augmentation"] == c) & (aug["label"] == lbl)]["shard"])))

    segs = raw.groupby("seg").agg(label=("label", "first"), subset=("subset", "first"),
                                  shards=("shard", list))
    if budget is None:
        budget = len(segs) + len(chosen)
    left = budget - len(chosen)
    segs["prio"] = (segs["label"] != "real").astype(int)     # живая речь первой
    segs = segs.sort_values("prio", kind="stable")
    alloc = {s: 0 for s in segs.index}
    for s in segs.index:                                     # по одному на источник
        if left == 0:
            break
        alloc[s], left = 1, left - 1
    order = list(segs[segs.prio == 0].index) + list(
        segs[segs.prio == 1].sort_values("shards", key=lambda c: -c.str.len()).index)
    while left > 0:                                          # добавка
        progressed = False
        for s in order:
            if left == 0:
                break
            if alloc[s] < len(segs.loc[s, "shards"]):
                alloc[s] += 1
                left -= 1
                progressed = True
        if not progressed:
            break
    for s, k in alloc.items():
        sh = segs.loc[s, "shards"]
        if k == 1:
            chosen.append(pick(sh))                          # скачанный или середина
        elif k > 1:
            idx = np.unique(np.linspace(0, len(sh) - 1, k).round().astype(int))
            chosen += [sh[i] for i in idx]
    return sorted(set(chosen))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=int, default=None,
                    help="сколько архивов выбрать (по умолчанию: по одному на каждый "
                         "источник + телефонные пары)")
    ap.add_argument("--data-dir", default=index_ruasd.DEFAULT_DIR)
    ap.add_argument("--aug-pairs", type=int, default=3,
                    help="сколько телефонных кодеков взять (по архиву живой речи и синтеза)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.data_dir, exist_ok=True)

    import download_ruasd
    repo, tars = download_ruasd.list_shards()
    print(f"Репозиторий {repo}: {len(tars)} архивов. Читаю первую запись каждого…")

    map_path = os.path.join(args.data_dir, "shard_map.csv")
    done = {}
    if os.path.exists(map_path):              # повторный запуск не перечитывает готовое
        old = pd.read_csv(map_path)
        done = {r.shard: r._asdict() for r in old.itertuples(index=False)
                if not (isinstance(r.error, str) and r.error)}
    todo = [t for t in tars if t not in done]
    with ThreadPoolExecutor(args.workers) as ex:
        for i, row in enumerate(ex.map(lambda t: probe(repo, t), todo), 1):
            done[row["shard"]] = row
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}")
    m = pd.DataFrame([done[t] for t in tars])
    m.to_csv(map_path, index=False, encoding="utf-8-sig")

    bad = m[m["label"].isna()]
    if len(bad):
        print(f"\nНе удалось прочитать {len(bad)} архивов. Пример ошибки: {bad.iloc[0]['error']}")

    seg = segments(m.dropna(subset=["label"]))
    print("\nКарта (диапазон архивов → источник первой записи):")
    for _, g in seg.groupby("seg"):
        a, b = g["shard"].iloc[0][6:12], g["shard"].iloc[-1][6:12]
        r = g.iloc[0]
        extra = f" [{r['augmentation']}]" if r["group"] != "raw" and r["augmentation"] else ""
        src = r["subset"] if r["group"] == "raw" else ""
        print(f"  {a}–{b} ({len(g):>3})  {r['label']:<4} {r['group']:<9} {src}{extra}")

    have = {os.path.basename(p) for p in
            __import__("glob").glob(os.path.join(args.data_dir, "*.tar"))}
    chosen = choose(m, args.budget, args.aug_pairs, have)
    if have:
        print(f"\nУже скачано архивов: {len(have)}; новых в выборе: {len(set(chosen) - have)}")
    with open(os.path.join(args.data_dir, "chosen_shards.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(chosen) + "\n")
    sel = seg[seg["shard"].isin(chosen)]
    print(f"\nВыбрано архивов: {len(chosen)} → chosen_shards.txt")
    print("  по классам первой записи:", sel.groupby(["label", "group"]).size().to_dict())
    aug_sel = sel[sel["group"] != "raw"]
    if len(aug_sel):
        print("  телефонные условия:", ", ".join(sorted(set(aug_sel["augmentation"].astype(str)))))
    raw_seg = seg[seg["group"] == "raw"]
    missed = sorted(set(raw_seg["subset"].astype(str)) - set(sel[sel["group"] == "raw"]["subset"].astype(str)))
    if missed:
        print(f"  Не вошли источники ({len(missed)}): {', '.join(missed)}. Увеличьте --budget.")
    print("Дальше:  python voice\\download_ruasd.py --list "
          + os.path.join("data", "ruasd", "chosen_shards.txt"))


if __name__ == "__main__":
    main()
