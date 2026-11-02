"""
Сводный отчёт предварительного эксперимента голосового модуля.

Собирает результаты всех прогонов train_baseline.py в один файл
reports/voice/summary.md: состав данных, проверка на шорткат, три теста,
незнакомые генераторы по отдельности, телефонные искажения, вклад групп
признаков и абляции, а также блок воспроизводимости (версии, архивы, seed).

    python voice\\make_summary.py
"""

import argparse
import glob
import hashlib
import json
import os
import platform
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import index_ruasd  # noqa: E402

REP_DIR = os.path.join(HERE, "..", "reports", "voice")
TESTS = {"test_seen": "Знакомые источники", "test_unseen": "Незнакомые источники",
         "test_phone": "Телефонный канал"}
RUNS = {"baseline_8000": "Все признаки, 8 кГц (основная)",
        "baseline_8000_no_prosody": "Без просодии, 8 кГц",
        "baseline_8000_no_lfcc_mfcc_spectral": "Только просодия, 8 кГц",
        "baseline_16000": "Все признаки, 16 кГц"}


def load(name):
    p = os.path.join(REP_DIR, f"{name}.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def pct(v):
    return "—" if v is None else f"{float(v) * 100:.1f}%"


def num(v):
    return "—" if v is None else f"{float(v):.3f}"


def versions():
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("numpy", "scipy", "sklearn", "pandas"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            pass
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=index_ruasd.DEFAULT_DIR)
    a = ap.parse_args()

    runs = {k: load(k) for k in RUNS}
    main_run = runs.get("baseline_8000")
    if main_run is None:
        sys.exit("Нет reports/voice/baseline_8000.json — сначала train_baseline.py --sr 8000")
    split = json.load(open(os.path.join(a.data_dir, "prepared", "split_report.json"), encoding="utf-8"))
    shards = sorted(os.path.basename(p) for p in glob.glob(os.path.join(a.data_dir, "*.tar")))
    shard_hash = hashlib.sha256("\n".join(shards).encode()).hexdigest()[:16]

    L = ["# Голосовой модуль: распознавание синтезированной речи", "",
         "Сводка всех прогонов предварительного эксперимента.", ""]

    # 1. данные
    L += ["## 1. Данные", "",
          f"RuASD (lab260/RuASD, CC-BY-NC-SA-4.0), скачано архивов: {len(shards)} из 250. "
          f"Фрагменты по {split['seconds']} с из середины записи, ресемплинг в "
          f"{split['sample_rate'] // 1000} кГц, обрезка тишины, нормализация громкости.", "",
          "| Выборка | Живая речь | Синтез | Источников |", "|---|---|---|---|"]
    for s, r in split["splits"].items():
        L.append(f"| {s} | {r['real']} | {r['fake']} | {len(r['sources'])} |")
    L += ["", "Отложены целиком в `test_unseen` (в обучение не попадали):", "",
          f"- генераторы: {', '.join(split['holdout_gens'])};",
          f"- корпуса живой речи: {', '.join(split['holdout_real'])}.", ""]

    # 2. шорткат
    sc = main_run.get("shortcut_check", {})
    L += ["## 2. Проверка на шорткат", "",
          "Бустинг только по исходной частоте дискретизации и длительности файла — без звука.", "",
          "| Выборка | EER | ROC-AUC |", "|---|---|---|"]
    for t, r in sc.items():
        L.append(f"| {TESTS.get(t, t)} | {num(r.get('eer'))} | {num(r.get('roc_auc'))} |")
    sc_auc = sc.get("test_seen", {}).get("roc_auc")
    if sc_auc is not None:
        L += ["", ("Вывод: формат файла почти полностью выдаёт класс. Поэтому все записи "
                   "приводятся к одной частоте, а модель получает только звук."
                   if sc_auc >= 0.9 else
                   "Вывод: по формату файла класс угадывается лишь частично."), ""]

    # 3. основные результаты
    L += ["## 3. Основные результаты", "",
          "Порог выбран на валидации по точке EER и на тестах не меняется.", "",
          "| Модель | Тест | EER | ROC-AUC | Ложные тревоги | Пропуски синтеза |",
          "|---|---|---|---|---|---|"]
    for k, title in RUNS.items():
        r = runs.get(k)
        if not r:
            continue
        for t, tname in TESTS.items():
            x = r["results"].get(t)
            if x:
                L.append(f"| {title} | {tname} | {num(x.get('eer'))} | {num(x.get('roc_auc'))} "
                         f"| {pct(x.get('false_alarm'))} | {pct(x.get('miss'))} |")
    res = main_run["results"]
    seen, unseen, phone = (res.get(t, {}).get("eer") for t in TESTS)
    L += ["", "Автоматические выводы по основной модели:", ""]
    if seen is not None and unseen is not None:
        if unseen > seen * 1.2 + 0.005:
            L.append(f"- На незнакомых источниках EER {num(unseen)} против {num(seen)} на знакомых "
                     f"(в {unseen / max(seen, 1e-3):.1f} раза выше): модель хорошо узнаёт «почерк» "
                     f"знакомых генераторов, перенос на новые генераторы слабее.")
        else:
            L.append(f"- На незнакомых источниках EER {num(unseen)}, на знакомых {num(seen)}: "
                     f"качество на новых генераторах сопоставимо со знакомыми, модель не "
                     f"привязалась к конкретным генераторам.")
    if phone is not None and unseen is not None:
        L.append(f"- Телефонный канал: EER {num(phone)}"
                 + (" — искажения канала заметно ухудшают распознавание."
                    if phone > unseen else " — кодеки ухудшают качество не сильнее, чем новый генератор."))
    r16 = runs.get("baseline_16000")
    if r16 and unseen is not None and "test_unseen" in r16["results"]:
        u16 = r16["results"]["test_unseen"].get("eer")
        if u16 is not None:
            L.append(f"- Полоса 16 кГц против 8 кГц на незнакомых источниках: EER {num(u16)} против "
                     f"{num(unseen)}. " + ("Часть признаков синтеза лежит выше 4 кГц и теряется в телефонной полосе."
                                           if u16 < unseen else "Верхние частоты почти не добавляют информации."))
    L.append("")

    # 4. по генераторам
    rows = main_run.get("per_source", {}).get("test_unseen", [])
    if rows:
        L += ["## 4. Незнакомые источники по отдельности (основная модель)", "",
              "Для генератора EER считается против всей отложенной живой речи.", "",
              "| Источник | Класс | Записей | EER | Пропуски | Ложные тревоги |", "|---|---|---|---|---|---|"]
        for r in sorted(rows, key=lambda r: -(r.get("eer") or -1)):
            L.append(f"| {r['source']} | {r['class']} | {r['n']} | {num(r.get('eer'))} "
                     f"| {pct(r.get('miss_at_thr'))} | {pct(r.get('false_alarm_at_thr'))} |")
        L.append("")
    rows = main_run.get("per_source", {}).get("test_phone", [])
    if rows:
        L += ["## 5. Телефонный канал по типам искажений (основная модель)", "",
              "| Искажение | Записей | EER | Пропуски | Ложные тревоги |", "|---|---|---|---|---|"]
        for r in sorted(rows, key=lambda r: -(r.get("eer") or -1)):
            L.append(f"| {r['source']} | {r['n']} | {num(r.get('eer'))} "
                     f"| {pct(r.get('miss_at_thr'))} | {pct(r.get('false_alarm_at_thr'))} |")
        L.append("")

    # 6. признаки
    L += ["## 6. Какие признаки работают", "",
          "Перестановочная важность групп (падение ROC-AUC на val при перемешивании, основная модель):", ""]
    for g, v in main_run.get("importance_by_group", {}).items():
        L.append(f"- {g}: {v}")
    L += ["", "Абляция — EER на тестах:", "",
          "| Набор признаков | Знакомые | Незнакомые | Телефон |", "|---|---|---|---|"]
    for k in ("baseline_8000", "baseline_8000_no_prosody", "baseline_8000_no_lfcc_mfcc_spectral"):
        r = runs.get(k)
        if r:
            e = [num(r["results"].get(t, {}).get("eer")) for t in TESTS]
            L.append(f"| {RUNS[k]} ({r['n_features']} призн.) | {e[0]} | {e[1]} | {e[2]} |")
    L.append("")

    # 7. воспроизводимость
    env = versions()
    repro = {"archives": len(shards), "archives_sha256_16": shard_hash, "seed": 42,
             "fragment_seconds": split["seconds"], "holdout_gens": split["holdout_gens"],
             "holdout_real": split["holdout_real"], "environment": env,
             "models": {k: {"fit_sec": r["fit_sec"], "n_iter": r["n_iter"], "n_features": r["n_features"],
                            "threshold": r["results"]["val"]["threshold"]}
                        for k, r in runs.items() if r}}
    with open(os.path.join(REP_DIR, "reproducibility.json"), "w", encoding="utf-8") as f:
        json.dump(repro, f, ensure_ascii=False, indent=2)
    L += ["## 7. Воспроизводимость", "",
          f"- Архивы: {len(shards)} шт., отпечаток списка `{shard_hash}` (список — `data/ruasd/chosen_shards.txt`).",
          "- Разбиение: `data/ruasd/prepared/*_meta.parquet`, `split_report.json`; seed 42.",
          f"- Окружение: Python {env['python']}, scikit-learn {env.get('sklearn')}, "
          f"numpy {env.get('numpy')}, scipy {env.get('scipy')}; {env['platform']}.",
          "- Время обучения: " + "; ".join(f"{RUNS[k]} — {r['fit_sec']} с"
                                           for k, r in runs.items() if r) + ".",
          "- Команда: `python voice\\run_all.py` (из папки training).", ""]

    out = os.path.join(REP_DIR, "summary.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"Сводка: {out}")


if __name__ == "__main__":
    main()
