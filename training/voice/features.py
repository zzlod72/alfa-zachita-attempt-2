"""
Признаки фрагмента речи для градиентного бустинга (182 числа).

Только numpy/scipy, без librosa: тот же код используется и при обучении,
и при применении модели. Параметры кадров масштабируются по частоте: окно 25 мс, шаг 10 мс.

Группы признаков (статистики по кадрам: среднее и стандартное отклонение):
  LFCC  — кепстр на ЛИНЕЙНОЙ шкале частот. Стандарт в задаче antispoofing:
          артефакты вокодеров сидят в высоких частотах, а мел-шкала их сжимает;
  MFCC  — кепстр на мел-шкале, для сравнения;
  спектр — центроид, ширина, спад (rolloff), плоскостность, поток, доля
          энергии в верхней половине полосы;
  просодия — частота основного тона (F0), доля озвученных кадров, приближённые
          джиттер и шиммер (нестабильность периода и амплитуды между соседними
          озвученными кадрами), доля пауз, разброс громкости. Это численное
          описание ритма и интонации речи.
"""

import numpy as np
from scipy.fft import dct, irfft, rfft

N_CEPS = 20


def _frames(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def _params(sr: int):
    win, hop = int(0.025 * sr), int(0.010 * sr)
    n_fft = 1 << (win - 1).bit_length()
    return win, hop, n_fft


def power_spectrum(x: np.ndarray, sr: int) -> np.ndarray:
    win, hop, n_fft = _params(sr)
    fr = _frames(x, win, hop) * np.hamming(win)
    return np.abs(rfft(fr, n_fft, axis=1)) ** 2          # [кадры, n_fft/2+1]


def _tri_bank(edges_hz: np.ndarray, sr: int, n_fft: int) -> np.ndarray:
    bins = np.floor((n_fft + 1) * edges_hz / sr).astype(int)
    fb = np.zeros((len(edges_hz) - 2, n_fft // 2 + 1))
    for i in range(len(edges_hz) - 2):
        a, b, c = bins[i], bins[i + 1], bins[i + 2]
        if b > a:
            fb[i, a:b] = (np.arange(a, b) - a) / (b - a)
        if c > b:
            fb[i, b:c] = (c - np.arange(b, c)) / (c - b)
    return fb


_BANKS = {}


def filterbank(kind: str, sr: int, n_filters: int) -> np.ndarray:
    key = (kind, sr, n_filters)
    if key not in _BANKS:
        _, _, n_fft = _params(sr)
        if kind == "linear":
            edges = np.linspace(0, sr / 2, n_filters + 2)
        else:
            mel = np.linspace(0, 2595 * np.log10(1 + (sr / 2) / 700), n_filters + 2)
            edges = 700 * (10 ** (mel / 2595) - 1)
        _BANKS[key] = _tri_bank(edges, sr, n_fft)
    return _BANKS[key]


def cepstrum(pspec: np.ndarray, sr: int, kind: str, n_filters: int = 40) -> np.ndarray:
    e = pspec @ filterbank(kind, sr, n_filters).T
    return dct(np.log(e + 1e-10), type=2, norm="ortho", axis=1)[:, :N_CEPS]


def deltas(c: np.ndarray, n: int = 2) -> np.ndarray:
    p = np.pad(c, ((n, n), (0, 0)), mode="edge")
    num = sum(k * (p[n + k: len(c) + n + k] - p[n - k: len(c) + n - k]) for k in range(1, n + 1))
    return num / (2 * sum(k * k for k in range(1, n + 1)))


def _pitch(x: np.ndarray, sr: int):
    """F0 по автокорреляции в окнах 40 мс; диапазон 70–400 Гц."""
    win, hop = int(0.040 * sr), int(0.010 * sr)
    fr = _frames(x, win, hop)
    fr = fr - fr.mean(axis=1, keepdims=True)
    energy = np.sqrt(np.mean(fr ** 2, axis=1) + 1e-12)
    spec = rfft(fr, 2 * win, axis=1)
    ac = irfft(np.abs(spec) ** 2, axis=1)[:, :win]
    ac = ac / (ac[:, :1] + 1e-12)
    lo, hi = int(sr / 400), min(int(sr / 70), win - 1)
    lag = lo + np.argmax(ac[:, lo:hi], axis=1)
    strength = ac[np.arange(len(ac)), lag]
    loud = energy > (energy.max() * 0.05)
    voiced = (strength > 0.5) & loud
    peaks = np.abs(fr).max(axis=1)
    return sr / lag, voiced, peaks, energy, loud


def prosody(x: np.ndarray, sr: int) -> np.ndarray:
    f0, voiced, peaks, energy, loud = _pitch(x, sr)
    vr = voiced.mean()
    if voiced.sum() >= 3:
        f = f0[voiced]
        st = 12 * np.log2(f / np.median(f))                 # полутоны от медианы
        # Соседние озвученные кадры: приближённые джиттер и шиммер.
        pair = voiced[1:] & voiced[:-1]
        T = 1.0 / f0
        jit = np.mean(np.abs(np.diff(T))[pair]) / np.mean(T[voiced]) if pair.any() else 0.0
        shim = (np.mean(np.abs(np.diff(peaks))[pair]) / np.mean(peaks[voiced])
                if pair.any() else 0.0)
        f0_stats = [np.mean(f), np.std(st), np.percentile(st, 95) - np.percentile(st, 5)]
    else:
        jit = shim = 0.0
        f0_stats = [0.0, 0.0, 0.0]
    edb = 20 * np.log10(energy + 1e-9)
    return np.array(f0_stats + [vr, jit, shim, 1 - loud.mean(), np.std(edb[loud]) if loud.any() else 0.0])


PROSODY_NAMES = ["f0_mean_hz", "f0_std_st", "f0_range_st", "voiced_ratio",
                 "jitter", "shimmer", "pause_ratio", "energy_std_db"]


def spectral(pspec: np.ndarray, sr: int) -> np.ndarray:
    freqs = np.linspace(0, sr / 2, pspec.shape[1])
    s = pspec + 1e-12
    tot = s.sum(axis=1)
    cent = (s * freqs).sum(axis=1) / tot
    bw = np.sqrt((s * (freqs - cent[:, None]) ** 2).sum(axis=1) / tot)
    cum = np.cumsum(s, axis=1)
    roll = freqs[np.argmax(cum >= 0.85 * cum[:, -1:], axis=1)]
    flat = np.exp(np.mean(np.log(s), axis=1)) / np.mean(s, axis=1)
    mag = np.sqrt(s)
    flux = np.r_[0, np.sqrt(np.sum(np.diff(mag, axis=0) ** 2, axis=1))]
    hi = s[:, freqs >= sr / 4].sum(axis=1) / tot
    out = []
    for v in (cent, bw, roll, flat, flux, hi):
        out += [np.mean(v), np.std(v)]
    return np.array(out)


SPECTRAL_NAMES = [f"{n}_{s}" for n in ("centroid", "bandwidth", "rolloff85", "flatness",
                                      "flux", "hiband_ratio") for s in ("mean", "std")]


def feature_names() -> list[str]:
    names = []
    for kind in ("lfcc", "lfcc_d", "lfcc_dd"):
        names += [f"{kind}{i}_{s}" for s in ("mean", "std") for i in range(N_CEPS)]
    names += [f"mfcc{i}_{s}" for s in ("mean", "std") for i in range(N_CEPS)]
    names += SPECTRAL_NAMES + ["zcr_mean", "zcr_std"] + PROSODY_NAMES
    return names


def extract(x: np.ndarray, sr: int) -> np.ndarray:
    """Вектор признаков одного фрагмента (≈190 чисел)."""
    ps = power_spectrum(x, sr)
    lf = cepstrum(ps, sr, "linear", 20)
    parts = []
    for c in (lf, deltas(lf), deltas(deltas(lf))):
        parts += [c.mean(0), c.std(0)]
    mf = cepstrum(ps, sr, "mel", 40)
    parts += [mf.mean(0), mf.std(0), spectral(ps, sr)]
    win, hop, _ = _params(sr)
    fr = _frames(x, win, hop)
    zcr = np.mean(np.abs(np.diff(np.sign(fr), axis=1)) > 0, axis=1)
    parts += [np.array([zcr.mean(), zcr.std()]), prosody(x, sr)]
    return np.concatenate(parts).astype(np.float32)
