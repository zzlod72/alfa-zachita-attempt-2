"""
Обработчик одного архива для train_full.py.

Вынесен в отдельный лёгкий модуль: на Windows каждый процесс-обработчик
заново импортирует модуль своей функции, и если это train_full.py, то в
каждый процесс грузятся pandas и scikit-learn. При 10 процессах это
съедает файл подкачки (ошибка «Файл подкачки слишком мал», WinError 1455).
Здесь — только numpy/scipy и код признаков.

Результат каждого архива сразу пишется на диск (parts/<архив>.npz), поэтому
прерванный расчёт продолжается с места остановки.
"""

import os
import tarfile

import numpy as np

import audioprep
import features


def shard_job(job):
    path, members, seconds, sr, out_path = job
    n = int(seconds * audioprep.TARGET_SR)
    want = set(members)
    rows, feats, skipped = [], [], {}
    with tarfile.open(path, "r") as tar:
        for m in tar:
            if m.name not in want:
                continue
            try:
                x, src_sr = audioprep.read_wav(tar.extractfile(m).read())
                c = audioprep.center_crop(audioprep.preprocess(x, src_sr), n)
                del x
                if c is None:
                    k = "короче фрагмента после обрезки тишины"
                    skipped[k] = skipped.get(k, 0) + 1
                    continue
                if sr != audioprep.TARGET_SR:
                    c, _ = audioprep.to_narrowband(c)
                f = np.nan_to_num(features.extract(c, sr), nan=0.0, posinf=0.0, neginf=0.0)
                rows.append(m.name)
                feats.append(f.astype(np.float32))
            except Exception as exc:  # noqa: BLE001
                k = type(exc).__name__
                skipped[k] = skipped.get(k, 0) + 1
    X = np.stack(feats) if feats else np.zeros((0, len(features.feature_names())), np.float32)
    tmp = out_path + ".tmp.npz"
    np.savez(tmp, members=np.array(rows, dtype=object), X=X)
    os.replace(tmp, out_path)
    return os.path.basename(path), len(rows), skipped
