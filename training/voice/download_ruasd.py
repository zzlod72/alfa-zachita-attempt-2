"""
Шаг 1 голосового модуля. Загрузка части датасета RuASD.

Датасет весит около 250 ГБ и разбит на архивы ruasd-NNNNNN.tar по ~1 ГБ.
Скачиваем только N архивов, причём не первые подряд, а равномерно разнесённые
по всему датасету: если записи в архивах упорядочены (например, сначала один
генератор, потом другой), первые 40 архивов могут покрыть лишь несколько
систем синтеза. Равномерный выбор даёт максимальное разнообразие при любом
порядке.

Рекомендуемый порядок (записи в архивах упорядочены по источникам):
    python voice/map_ruasd.py                                     # карта, выбор архивов
    python voice/download_ruasd.py --list data/ruasd/chosen_shards.txt

Другие варианты:
    python download_ruasd.py                    # 40 архивов, равномерно
    python download_ruasd.py --shards 5         # пробный запуск, ~5 ГБ
    python download_ruasd.py --data-dir D:\\ruasd

Загрузка докачивается: если прервать и запустить снова, уже скачанные
архивы пропускаются. После загрузки автоматически строится индекс
(index_ruasd.py) и печатается сводка по генераторам и корпусам.

Лицензия датасета: CC-BY-NC-SA-4.0 (только некоммерческое использование).
"""

import argparse
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import index_ruasd  # noqa: E402

REPOS = ["lab260/RuASD", "MTUCI/RuASD"]   # датасет переезжал; пробуем оба имени
GB = 1024 ** 3


def list_shards():
    from huggingface_hub import HfApi
    api = HfApi()
    last_err = None
    for repo in REPOS:
        try:
            files = api.list_repo_files(repo, repo_type="dataset")
            tars = sorted(f for f in files if f.endswith(".tar"))
            if tars:
                return repo, tars
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    sys.exit(f"Не удалось получить список файлов RuASD: {last_err}")


def pick_spread(tars: list[str], n: int) -> list[str]:
    if n >= len(tars):
        return tars
    idx = np.unique(np.linspace(0, len(tars) - 1, n).round().astype(int))
    return [tars[i] for i in idx]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", type=int, default=40, help="сколько архивов скачать")
    ap.add_argument("--list", help="файл со списком архивов (из map_ruasd.py)")
    ap.add_argument("--first", action="store_true",
                    help="взять первые N архивов вместо равномерного выбора")
    ap.add_argument("--data-dir", default=index_ruasd.DEFAULT_DIR)
    ap.add_argument("--workers", type=int, default=4, help="параллельных загрузок")
    ap.add_argument("--no-index", action="store_true", help="не строить индекс после загрузки")
    args = ap.parse_args()

    repo, tars = list_shards()
    if args.list:
        with open(args.list, encoding="utf-8") as f:
            wanted = [ln.strip() for ln in f if ln.strip()]
        chosen = [t for t in tars if t in set(wanted)]
    elif args.first:
        chosen = tars[: args.shards]
    else:
        chosen = pick_spread(tars, args.shards)
    print(f"Репозиторий {repo}: всего архивов {len(tars)}, выбрано {len(chosen)}")
    print("Номера:", ", ".join(c.split("-")[-1].split(".")[0] for c in chosen))

    os.makedirs(args.data_dir, exist_ok=True)
    have = {f for f in chosen if os.path.exists(os.path.join(args.data_dir, f))}
    need_gb = (len(chosen) - len(have)) * 1.0 + 2   # ~1 ГБ на архив + запас
    free_gb = shutil.disk_usage(args.data_dir).free / GB
    print(f"Уже скачано: {len(have)}. Нужно ещё ~{need_gb:.0f} ГБ, свободно {free_gb:.0f} ГБ")
    if free_gb < need_gb:
        sys.exit("Не хватает места на диске. Укажите другой диск через --data-dir "
                 "или уменьшите --shards.")

    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns=chosen,
                      local_dir=args.data_dir, max_workers=args.workers,
                      token=os.environ.get("HF_TOKEN"))
    print("Загрузка завершена.")

    if not args.no_index:
        sys.argv = ["index_ruasd.py", "--data-dir", args.data_dir]
        index_ruasd.main()


if __name__ == "__main__":
    main()
