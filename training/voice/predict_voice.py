"""
Проверка обученного детектора на своём аудиофайле.

    python voice\\predict_voice.py запись.wav [ещё.wav ...]
    python voice\\predict_voice.py --model baseline_8000 запись.wav   # модель предварительного эксперимента

Обработка та же, что при обучении (audioprep.py): ресемплинг в 16 кГц,
обрезка тишины, нормализация громкости. Длинная запись режется на окна
длины фрагмента с перекрытием 50%, оценки окон усредняются.

Форматы — wav/flac/ogg. Запись с телефона (m4a, mp3) сначала переведите
в wav, например: ffmpeg -i voice.m4a voice.wav
"""

import argparse
import os
import sys

import joblib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import audioprep  # noqa: E402
import features  # noqa: E402

MODEL_DIR = os.path.join(HERE, "..", "..", "models", "voice")


def score_file(bundle, path: str):
    with open(path, "rb") as f:
        x, sr = audioprep.read_wav(f.read())
    x = audioprep.preprocess(x, sr)
    n = int(bundle["seconds"] * audioprep.TARGET_SR)
    keep = np.isin(features.feature_names(), bundle["feature_names"])
    rows = []
    for w in audioprep.windows(x, n):
        if bundle["sr"] != audioprep.TARGET_SR:
            w, _ = audioprep.to_narrowband(w)
        rows.append(features.extract(w, bundle["sr"])[keep])
    p = bundle["model"].predict_proba(np.stack(rows))[:, 1]
    return float(p.mean()), len(rows), len(x) / audioprep.TARGET_SR


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--model", default="full_8000", help="имя модели в models/voice")
    a = ap.parse_args()
    bundle = joblib.load(os.path.join(MODEL_DIR, f"{a.model}.joblib"))
    for path in a.files:
        try:
            p, k, dur = score_file(bundle, path)
        except Exception as exc:  # noqa: BLE001
            print(f"{path}: не удалось обработать ({type(exc).__name__}: {exc})")
            continue
        verdict = "ПОХОЖЕ НА СИНТЕЗ" if p >= bundle["threshold"] else "похоже на живую речь"
        note = "  (запись короче фрагмента — оценка ненадёжна)" if dur < bundle["seconds"] else ""
        print(f"{path}: вероятность синтеза {p:.3f} по {k} окнам, порог "
              f"{bundle['threshold']:.3f} — {verdict}{note}")


if __name__ == "__main__":
    main()
