"""
Предобработка аудио — общая для подготовки данных, обучения и применения модели.

Каждый шаг закрывает конкретную «утечку», найденную в RuASD:
  1. всё приводится к одной частоте (16 кГц): по одной частоте исходного
     файла живую речь и синтез можно было различить с точностью ~90%;
  2. обрезается тишина в начале и в конце: у синтеза она «стерильная»,
     у записей — с шумом, и модель могла бы учиться на паузах;
  3. громкость нормализуется по RMS: уровни записи у корпусов разные;
  4. фрагмент фиксированной длины вырезается без дополнения: повтор
     (зацикливание) короткой записи создаёт периодичность, а дополнение
     нулями — искусственную тишину, и то и другое выдаёт длину исходника.
"""

import io
import struct
from math import gcd

import numpy as np
from scipy.signal import resample_poly

TARGET_SR = 16000


def read_wav(data: bytes) -> tuple[np.ndarray, int]:
    """WAV → (моно float32 в [-1, 1], частота). Понимает PCM 8/16/24/32 бит,
    float 32/64 и WAVE_FORMAT_EXTENSIBLE; корректно обрабатывает «потоковые»
    файлы с заглушкой вместо размера данных. Прочие форматы — через soundfile.
    """
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return _read_soundfile(data)
    pos, fmt = 12, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = pos + 8
        if cid == b"fmt ":
            fmt = data[body:body + size]
        elif cid == b"data":
            if size == 0 or body + size > len(data):
                size = len(data) - body
            if fmt is None:
                break
            code, ch, sr, _, align, bits = struct.unpack("<HHIIHH", fmt[:16])
            if code == 0xFFFE and len(fmt) >= 26:          # EXTENSIBLE
                code = struct.unpack("<H", fmt[24:26])[0]
            raw = data[body:body + size - size % align]
            x = _decode(raw, code, bits)
            if x is None:
                return _read_soundfile(data)
            x = x.reshape(-1, ch).mean(axis=1) if ch > 1 else x
            return x.astype(np.float32), sr
        pos = body + size + (size & 1)
    return _read_soundfile(data)


def _decode(raw: bytes, code: int, bits: int):
    if code == 1:
        if bits == 16:
            return np.frombuffer(raw, "<i2") / 32768.0
        if bits == 32:
            return np.frombuffer(raw, "<i4") / 2147483648.0
        if bits == 8:
            return (np.frombuffer(raw, "u1").astype(np.float32) - 128) / 128.0
        if bits == 24:
            b = np.frombuffer(raw, "u1").reshape(-1, 3).astype(np.int32)
            v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
            v = np.where(v >= 1 << 23, v - (1 << 24), v)
            return v / 8388608.0
    if code == 3:
        return np.frombuffer(raw, "<f4" if bits == 32 else "<f8").astype(np.float32)
    return None


def _read_soundfile(data: bytes):
    import soundfile as sf
    x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return x.mean(axis=1), sr


def resample(x: np.ndarray, sr: int, target: int = TARGET_SR) -> np.ndarray:
    if sr == target:
        return x
    g = gcd(sr, target)
    return resample_poly(x, target // g, sr // g).astype(np.float32)


def trim_silence(x: np.ndarray, sr: int, top_db: float = 40.0, margin_s: float = 0.1):
    """Обрезает тишину по краям: кадры 25 мс, порог — пик минус top_db."""
    frame = int(0.025 * sr)
    n = len(x) // frame
    if n < 3:
        return x
    rms = np.sqrt(np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    thr = max(db.max() - top_db, -60.0)
    idx = np.where(db > thr)[0]
    if len(idx) == 0:
        return x
    m = int(margin_s * sr)
    return x[max(0, idx[0] * frame - m): min(len(x), (idx[-1] + 1) * frame + m)]


def normalize_rms(x: np.ndarray, target_dbfs: float = -23.0) -> np.ndarray:
    x = x - x.mean()
    rms = np.sqrt(np.mean(x ** 2) + 1e-12)
    y = x * (10 ** (target_dbfs / 20) / rms)
    peak = np.abs(y).max()
    return (y / peak * 0.99 if peak > 0.99 else y).astype(np.float32)


def preprocess(x: np.ndarray, sr: int) -> np.ndarray:
    """Полная цепочка до фиксированной длины, кроме нарезки."""
    x = resample(x, sr, TARGET_SR)
    x = trim_silence(x, TARGET_SR)
    return normalize_rms(x)


def center_crop(x: np.ndarray, n: int) -> np.ndarray | None:
    """Фрагмент длиной n из середины. Короче n — None (без дополнения)."""
    if len(x) < n:
        return None
    s = (len(x) - n) // 2
    return x[s: s + n]


def windows(x: np.ndarray, n: int, hop: int | None = None) -> np.ndarray:
    """Окна длиной n для длинной записи (при оценке файла оценки окон усредняются).
    Запись короче окна дополняется повтором — для инференса это допустимо."""
    hop = hop or n // 2
    if len(x) < n:
        x = np.tile(x, int(np.ceil(n / max(len(x), 1))))[:n]
    starts = range(0, len(x) - n + 1, hop)
    return np.stack([x[s: s + n] for s in starts])


def to_narrowband(x: np.ndarray, sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Полоса телефонного канала: 16 кГц → 8 кГц (всё выше 4 кГц отрезается)."""
    return resample(x, sr, 8000), 8000
