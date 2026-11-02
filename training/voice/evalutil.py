"""Метрики для задачи antispoofing (EER, отчёт по выборке, разбивка по источникам)."""

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve


def eer(y, score) -> tuple[float, float]:
    """Equal Error Rate и порог, на котором доля пропусков синтеза (FNR)
    равна доле ложных тревог на живой речи (FPR). Чем меньше, тем лучше;
    0.5 — уровень случайного угадывания."""
    fpr, tpr, thr = roc_curve(y, score)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def report(y, score, thr) -> dict:
    y, score = np.asarray(y), np.asarray(score)
    out = {"n": int(len(y)), "real": int((y == 0).sum()), "fake": int((y == 1).sum())}
    pred = (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out.update(threshold=round(float(thr), 4),
               false_alarm=round(fp / max(1, fp + tn), 4),     # живую назвали синтезом
               miss=round(fn / max(1, fn + tp), 4),            # синтез пропустили
               precision=round(float(precision_score(y, pred, zero_division=0)), 4),
               recall=round(float(recall_score(y, pred, zero_division=0)), 4),
               f1=round(float(f1_score(y, pred, zero_division=0)), 4),
               confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]])
    if 0 < y.sum() < len(y):
        e, _ = eer(y, score)
        out.update(eer=round(e, 4), roc_auc=round(float(roc_auc_score(y, score)), 4))
    return out


def by_source(meta, score, thr, split_label="") -> list:
    """Для теста на незнакомых данных: EER каждого отложенного генератора
    против всей отложенной живой речи; для корпусов — доля ложных тревог."""
    rows = []
    real = meta.y.values == 0
    for src, g in meta.groupby("source"):
        m = meta.source.values == src
        if g.y.nunique() == 2:                      # условие с обоими классами
            e, _ = eer(g.y.values, score[m])
            ym, sm = g.y.values, score[m]
            rows.append({"source": src, "class": "оба", "n": int(m.sum()), "eer": round(e, 4),
                         "false_alarm_at_thr": round(float((sm[ym == 0] >= thr).mean()), 4),
                         "miss_at_thr": round(float((sm[ym == 1] < thr).mean()), 4)})
        elif g.y.iloc[0] == 1:
            mask = m | real
            e, _ = eer(meta.y.values[mask], score[mask])
            rows.append({"source": src, "class": "синтез", "n": int(m.sum()), "eer": round(e, 4),
                         "miss_at_thr": round(float((score[m] < thr).mean()), 4)})
        else:
            rows.append({"source": src, "class": "живая", "n": int(m.sum()),
                         "false_alarm_at_thr": round(float((score[m] >= thr).mean()), 4)})
    return rows
