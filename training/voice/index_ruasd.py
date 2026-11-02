"""
Индексация скачанных архивов RuASD.

Проходит по всем ruasd-*.tar в каталоге, НЕ распаковывая аудио на диск:
читает JSON-метаданные каждой записи и заголовок её WAV-файла (частота
дискретизации, число каналов, длительность). Результат — одна таблица,
с которой дальше работают все шаги: выборка, разбиение, обучение.

Запуск (обычно вызывается автоматически из download_ruasd.py):
    python index_ruasd.py
    python index_ruasd.py --data-dir D:\\ruasd

Результат:
    data/ruasd/ruasd_index.parquet  — по строке на запись
    data/ruasd/ruasd_report.json    — сводка: генераторы, корпуса, частоты
"""

import argparse
import glob
import json
import os
import struct
import sys
import tarfile
from collections import defaultdict

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(HERE, "..", "data", "ruasd")

# Поля метаданных, которые сохраняем. Тексты (transcription, true_lines и т.п.)
# сознательно не берём: модель должна слушать звук, а не читать текст.
KEEP_FIELDS = ["sample_id", "label", "group", "subset", "augmentation", "source_type",
               "model", "duration", "speakers", "source_audio", "mos_pred", "ops",
               "filename", "audio_relpath"]

AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg", ".opus")


def wav_header(fileobj, file_size: int | None = None) -> dict:
    """Читает заголовок RIFF/WAV по чанкам, не загружая аудиоданные.

    Модуль wave из стандартной библиотеки не открывает float-WAV, поэтому
    разбираем чанки fmt и data вручную. Некоторые синтезаторы пишут WAV
    «потоком» и оставляют в поле размера данных заглушку (0 или 0xFFFFFFFF),
    поэтому размер данных ограничивается фактическим размером файла.
    """
    pos = 12
    head = fileobj.read(12)
    if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        return {}
    info = {}
    while True:
        ch = fileobj.read(8)
        if len(ch) < 8:
            break
        cid, size = ch[:4], struct.unpack("<I", ch[4:])[0]
        pos += 8
        if cid == b"fmt ":
            fmt = fileobj.read(size)
            pos += size
            audio_fmt, channels, sr, _, block_align, bits = struct.unpack("<HHIIHH", fmt[:16])
            info.update(codec=audio_fmt, channels=channels, sr=sr, bits=bits,
                        block_align=block_align)
        elif cid == b"data":
            if file_size is not None:
                real = max(0, file_size - pos)
                if size == 0 or size > real:
                    size = real
            if info.get("block_align"):
                info["frames"] = size // info["block_align"]
                info["wav_duration"] = round(info["frames"] / info["sr"], 3)
            break
        else:
            fileobj.read(size + (size & 1))  # пропускаем чанк (с выравниванием)
            pos += size + (size & 1)
    info.pop("block_align", None)
    return info


def _stem(name: str) -> str:
    base = os.path.basename(name)
    for ext in (".json",) + AUDIO_EXT:
        if base.lower().endswith(ext):
            return name[: -len(ext)]
    return name


def index_shard(path: str) -> list[dict]:
    shard = os.path.basename(path)
    metas, audio = {}, {}
    with tarfile.open(path, "r") as tar:
        for m in tar:
            if not m.isfile():
                continue
            low = m.name.lower()
            if low.endswith(".json"):
                try:
                    data = json.load(tar.extractfile(m))
                except Exception:  # noqa: BLE001
                    continue
                items = data if isinstance(data, list) else [data]
                for i, d in enumerate(items):
                    if isinstance(d, dict) and "label" in d:
                        key = _stem(m.name) if len(items) == 1 else f"{_stem(m.name)}#{i}"
                        metas[key] = d
            elif low.endswith(AUDIO_EXT):
                info = wav_header(tar.extractfile(m), m.size) if low.endswith(".wav") else {}
                audio[_stem(m.name)] = {"member": m.name, "bytes": m.size, **info}

    # Сопоставление метаданных и аудио: сначала по общему имени (формат
    # WebDataset), затем по полю filename / audio_relpath.
    by_base = {os.path.basename(k): k for k in audio}
    rows = []
    for key, d in metas.items():
        a = audio.get(key)
        if a is None:
            for f in (d.get("filename"), d.get("audio_relpath")):
                if f and _stem(os.path.basename(str(f))) in by_base:
                    a = audio[by_base[_stem(os.path.basename(str(f)))]]
                    break
        row = {"shard": shard, "key": key}
        row.update({k: d.get(k) for k in KEEP_FIELDS})
        for k in ("speakers", "ops"):
            if row[k] is not None and not isinstance(row[k], str):
                row[k] = json.dumps(row[k], ensure_ascii=False)
        row.update(a or {"member": None})
        rows.append(row)
    return rows


def summarize(df: pd.DataFrame) -> dict:
    hours = lambda s: round(float(s.fillna(0).sum()) / 3600, 2)  # noqa: E731
    dur = df["wav_duration"].fillna(df["duration"])
    df = df.assign(dur=pd.to_numeric(dur, errors="coerce"))
    rep = {
        "records": int(len(df)),
        "without_audio": int(df["member"].isna().sum()),
        "by_label": {k: {"n": int(len(g)), "hours": hours(g.dur)} for k, g in df.groupby("label")},
        "by_group": df["group"].value_counts(dropna=False).to_dict(),
    }
    raw = df[df["group"].fillna("raw") == "raw"]
    fake, real = raw[raw.label == "fake"], raw[raw.label == "real"]
    rep["fake_by_subset"] = {k: int(v) for k, v in fake["subset"].value_counts().items()}
    rep["real_by_subset"] = {k: int(v) for k, v in real["subset"].value_counts().items()}
    aug = df[df["group"] == "augmented"]
    rep["augmentations"] = {f"{a} / {lbl}": int(n) for (a, lbl), n in
                            aug.groupby(["augmentation", "label"]).size().items()}
    # Частоты дискретизации по классам — главная проверка на «шорткат».
    sr = defaultdict(dict)
    for (lbl, s), n in raw.groupby(["label", "sr"]).size().items():
        sr[lbl][str(int(s))] = int(n)
    rep["sample_rate_by_label"] = dict(sr)
    return rep


def print_report(rep: dict):
    print(f"\nЗаписей: {rep['records']} (без найденного аудио: {rep['without_audio']})")
    for k, v in rep["by_label"].items():
        print(f"  {k}: {v['n']} записей, {v['hours']} ч")
    print("Группы:", rep["by_group"])
    print(f"\nГенераторов (subset у fake): {len(rep['fake_by_subset'])}")
    for k, v in rep["fake_by_subset"].items():
        print(f"  {v:>7}  {k}")
    print(f"\nКорпусов живой речи (subset у real): {len(rep['real_by_subset'])}")
    for k, v in rep["real_by_subset"].items():
        print(f"  {v:>7}  {k}")
    if rep["augmentations"]:
        print("\nИскажённые записи (условие / класс):")
        for k, v in list(rep["augmentations"].items())[:30]:
            print(f"  {v:>7}  {k}")
    print("\nЧастоты дискретизации (только исходные записи):")
    for lbl, d in rep["sample_rate_by_label"].items():
        print(f"  {lbl}: {d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=DEFAULT_DIR)
    args = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(args.data_dir, "**", "*.tar"), recursive=True))
    if not shards:
        sys.exit(f"Не найдено .tar в {args.data_dir}")
    rows = []
    for i, p in enumerate(shards, 1):
        r = index_shard(p)
        rows += r
        print(f"[{i}/{len(shards)}] {os.path.basename(p)}: {len(r)} записей")

    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("В архивах не найдено метаданных с полем label — пришлите вывод, разберёмся.")
    out = os.path.join(args.data_dir, "ruasd_index.parquet")
    df.to_parquet(out, index=False)
    rep = summarize(df)
    with open(os.path.join(args.data_dir, "ruasd_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print_report(rep)
    print(f"\nИндекс: {out}")


if __name__ == "__main__":
    main()
