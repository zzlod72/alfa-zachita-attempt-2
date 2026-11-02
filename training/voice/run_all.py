"""
Предварительный эксперимент одной командой.

    python voice\\run_all.py              # все прогоны (~10–20 мин)
    python voice\\run_all.py --skip-16k   # без сравнения полос, ещё быстрее
    python voice\\run_all.py --prepare    # заново сделать разбиение и фрагменты

Шаги:
  1. prepare_ruasd.py — только если data/ruasd/prepared ещё нет (или --prepare);
  2. train_baseline.py — основная модель, полоса 8 кГц (телефон);
  3. train_baseline.py — две абляции: без просодии и только просодия;
  4. train_baseline.py — полоса 16 кГц для сравнения (если не --skip-16k);
  5. make_summary.py — сводный отчёт reports/voice/summary.md.

Признаки кешируются по полосе, поэтому абляции и повторные запуски быстрые.
Все случайные выборы фиксированы (seed 42), повторный запуск на тех же
архивах даёт те же цифры.
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import index_ruasd  # noqa: E402


def run(script, *args):
    cmd = [sys.executable, os.path.join(HERE, script), *args]
    print("\n" + "=" * 70 + "\n>>> " + " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=index_ruasd.DEFAULT_DIR)
    ap.add_argument("--prepare", action="store_true", help="пересоздать разбиение и фрагменты")
    ap.add_argument("--skip-16k", action="store_true", help="не обучать модель для полосы 16 кГц")
    a = ap.parse_args()
    t0 = time.time()
    dd = ["--data-dir", a.data_dir]

    if a.prepare or not os.path.exists(os.path.join(a.data_dir, "prepared", "split_report.json")):
        run("prepare_ruasd.py", *dd)
    else:
        print("Разбиение и фрагменты уже есть (data/ruasd/prepared) — шаг подготовки пропущен.")
    run("train_baseline.py", "--sr", "8000", *dd)
    run("train_baseline.py", "--sr", "8000", "--exclude", "prosody", *dd)
    run("train_baseline.py", "--sr", "8000", "--exclude", "lfcc", "mfcc", "spectral", *dd)
    if not a.skip_16k:
        run("train_baseline.py", "--sr", "16000", *dd)
    run("make_summary.py", *dd)
    print(f"\nВсё готово за {(time.time() - t0) / 60:.1f} мин. "
          f"Сводка: training\\reports\\voice\\summary.md")


if __name__ == "__main__":
    main()
