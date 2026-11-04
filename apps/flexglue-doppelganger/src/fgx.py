"""CPU helpers for the FlexGlue covisibility experiment.

ONNX call sequence follows cvg/FlexGlue onnx/match_pair.py (MIT).
Images and weights stay under work/ and are not committed.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
import time
import urllib.parse
import urllib.request
import zlib

import cv2
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(APP, "work")
USER_AGENT = "tech-demos-flexglue/0.1 (CPU research demo; local)"


def ensure_dirs() -> None:
    for name in ("cache", "dataset", "images", "feat", "sift", "pert", "viz"):
        os.makedirs(os.path.join(WORK, name), exist_ok=True)


def set_thread_env(n: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    cv2.setNumThreads(n)


def rss_mb() -> tuple[float, float]:
    rss = hwm = 0.0
    with open("/proc/self/status", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) / 1024.0
            elif line.startswith("VmHWM:"):
                hwm = int(line.split()[1]) / 1024.0
    return rss, hwm


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def http_bytes(url: str, timeout: int = 180) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def download(url: str, dest: str, sha256: str | None = None) -> str:
    if os.path.isfile(dest) and (sha256 is None or sha256_file(dest) == sha256):
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".tmp"
    print(f"download {url}", flush=True)
    data = http_bytes(url, timeout=600)
    with open(tmp, "wb") as handle:
        handle.write(data)
    if sha256 is not None and sha256_file(tmp) != sha256:
        os.remove(tmp)
        raise RuntimeError(f"checksum mismatch for {url}")
    os.replace(tmp, dest)
    return dest


def license_bucket(license_name: str) -> str:
    text = (license_name or "").lower().replace("_", " ")
    if any(token in text for token in ("noncommercial", "-nc", " nc")):
        return "nc"
    if any(token in text for token in ("noderiv", "no deriv", "-nd", " nd")):
        return "nd"
    if "cc0" in text or "public domain" in text or text.strip() in {"pdm", "pd"}:
        return "open"
    if "by-sa" in text or "by sa" in text:
        return "sa"
    if "cc by" in text or text.strip().startswith("by "):
        return "by"
    return "other"


def _load_pairs(cfg: dict) -> np.ndarray:
    archive = download(
        cfg["dataset"]["pairs_metadata_url"],
        os.path.join(WORK, "dataset", "pairs_metadata.tar.gz"),
    )
    npy_name = "./doppelgangers/pairs_metadata/test_pairs.npy"
    with tarfile.open(archive, "r:gz") as tar:
        member = tar.extractfile(npy_name)
        if member is None:
            raise RuntimeError("test_pairs.npy missing from pairs_metadata archive")
        raw = member.read()
    return np.load(io.BytesIO(raw), allow_pickle=True)


def _load_attr(cfg: dict) -> dict:
    path = download(
        cfg["dataset"]["attributions_url"],
        os.path.join(WORK, "dataset", "attributions.json"),
    )
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _decode_image(data: bytes) -> np.ndarray | None:
    if not data or data[:1] in (b"<", b"{"):
        return None
    buf = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None or bgr.size == 0:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _cap_long_side(rgb: np.ndarray, limit: int) -> np.ndarray:
    height, width = rgb.shape[:2]
    long_side = max(height, width)
    if long_side <= limit:
        return rgb
    scale = limit / float(long_side)
    size = (max(32, int(round(width * scale))), max(32, int(round(height * scale))))
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)


def _save_png(rgb: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"failed to encode {path}")
    with open(path, "wb") as handle:
        handle.write(buf.tobytes())


def commons_url(filename: str) -> str:
    title = filename.replace(" ", "_")
    return "https://commons.wikimedia.org/wiki/Special:FilePath/" + urllib.parse.quote(title, safe="()'!$+,;@")


def fetch_commons_image(filename: str, dest: str, long_side: int) -> tuple[str, tuple[int, int]] | None:
    if os.path.isfile(dest):
        rgb = cv2.cvtColor(cv2.imread(dest, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return dest, (int(rgb.shape[1]), int(rgb.shape[0]))
    try:
        data = http_bytes(commons_url(filename), timeout=90)
    except Exception as exc:  # noqa: BLE001 — network failures skip the pair
        print(f"  skip download {filename}: {exc}", flush=True)
        return None
    rgb = _decode_image(data)
    if rgb is None:
        print(f"  skip decode {filename}", flush=True)
        return None
    original = (int(rgb.shape[1]), int(rgb.shape[0]))
    rgb = _cap_long_side(rgb, long_side)
    _save_png(rgb, dest)
    return dest, original


def _safe(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
    return cleaned[:120] or "img"


def materialize_pool(cfg: dict) -> tuple[list[dict], list[dict]]:
    """First-fit scene sample. Download failure is the only skip. No matcher calls."""
    ensure_dirs()
    pairs = _load_pairs(cfg)
    attr = _load_attr(cfg)
    allow = set(cfg["dataset"]["image_license_allow"])
    n_scenes = int(cfg["dataset"]["n_scenes"])
    max_attempts = int(cfg["dataset"]["max_attempts_per_label"])
    long_side = int(cfg["dataset"]["ingest_long_side_max"])
    scenes = sorted({str(row[0]).split("/")[0] for row in pairs})
    pool: list[dict] = []
    log: list[dict] = []
    scene_ord = 0
    for scene in scenes:
        if len(pool) >= n_scenes:
            break
        found: dict[int, dict] = {}
        attempts = {0: 0, 1: 0}
        for row in pairs:
            if str(row[0]).split("/")[0] != scene:
                continue
            label = int(row[2])
            if label in found or attempts[label] >= max_attempts:
                continue
            names = [str(row[0]).split("/")[-1], str(row[1]).split("/")[-1]]
            metas = []
            eligible = True
            for name in names:
                meta = attr.get(name)
                if meta is None or license_bucket(str(meta.get("license", ""))) not in allow:
                    eligible = False
                    break
                metas.append(meta)
            if not eligible:
                continue
            attempts[label] += 1
            slot = f"sc{scene_ord:02d}_{'pos' if label == 1 else 'neg'}"
            paths = []
            originals = []
            ok = True
            for side, name in enumerate(names):
                dest = os.path.join(WORK, "images", f"{slot}_{side}_{_safe(name)}.png")
                got = fetch_commons_image(name, dest, long_side)
                if got is None:
                    ok = False
                    break
                paths.append(got[0])
                originals.append({"wh_original": list(got[1]), "file": name})
            if not ok:
                log.append({"scene": scene, "label": label, "files": names, "reason": "download_or_decode"})
                continue
            found[label] = {
                "id": slot,
                "scene": scene,
                "label": label,
                "files": names,
                "paths": paths,
                "originals": originals,
                "n_sift_dataset": int(row[3]),
                "licenses": [str(m.get("license", "")) for m in metas],
                "authors": [str(m.get("user", "")) for m in metas],
                "urls": [str(m.get("url", "")) for m in metas],
                "license_buckets": [license_bucket(str(m.get("license", ""))) for m in metas],
            }
            if 0 in found and 1 in found:
                break
        if 0 in found and 1 in found:
            pool.append(found[1])
            pool.append(found[0])
            print(f"pool scene {scene_ord}: {scene}", flush=True)
            scene_ord += 1
        else:
            log.append({"scene": scene, "reason": "scene_incomplete", "attempts": attempts})
    if len(pool) < 2 * n_scenes:
        raise RuntimeError(f"only {len(pool) // 2} complete scenes; need {n_scenes}")
    return pool, log


def fetch_anchor(cfg: dict) -> dict | None:
    spec = cfg["dataset"].get("anchor_berlin")
    if not spec:
        return None
    long_side = int(cfg["dataset"]["ingest_long_side_max"])
    paths = []
    for item in spec:
        dest = os.path.join(WORK, "images", f"anchor_{item['id']}.png")
        if not os.path.isfile(dest):
            decoded = _decode_image(http_bytes(item["url"]))
            if decoded is None:
                return None
            _save_png(_cap_long_side(decoded, long_side), dest)
        paths.append(dest)
    return {
        "id": "anchor_berlin",
        "scene": "flexglue_readme",
        "label": int(spec["label"]),
        "files": [item["id"] + ".jpg" for item in spec],
        "paths": paths,
        "licenses": ["MIT-repo-asset", "MIT-repo-asset"],
        "authors": ["cvg/FlexGlue", "cvg/FlexGlue"],
        "urls": [item["url"] for item in spec],
        "note": spec.get("note", ""),
        "in_metric": False,
    }


def sample_scene_ids(pool: list[dict], k: int, seed: int) -> list[str]:
    scenes = []
    for item in pool:
        if item["scene"] not in scenes:
            scenes.append(item["scene"])
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(scenes), size=k, replace=False)
    return [scenes[int(i)] for i in sorted(int(x) for x in picked)]


def load_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def image_digest(rgb: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(rgb.shape).encode())
    digest.update(np.ascontiguousarray(rgb).tobytes())
    return digest.hexdigest()[:20]


def rotate_image(rgb: np.ndarray, degrees: float) -> np.ndarray:
    if abs(degrees) < 1e-6:
        return rgb
    height, width = rgb.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), degrees, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int(round(height * sin + width * cos))
    new_h = int(round(height * cos + width * sin))
    matrix[0, 2] += (new_w / 2.0) - width / 2.0
    matrix[1, 2] += (new_h / 2.0) - height / 2.0
    return cv2.warpAffine(rgb, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))


def downscale_image(rgb: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 0.999:
        return rgb
    height, width = rgb.shape[:2]
    size = (max(32, int(round(width * scale))), max(32, int(round(height * scale))))
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)


def blur_image(rgb: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return rgb
    kernel = max(3, int(round(sigma * 6)) | 1)
    return cv2.GaussianBlur(rgb, (kernel, kernel), sigmaX=float(sigma))


def texture_mix(rgb: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return rgb
    mean = rgb.astype(np.float32).mean(axis=(0, 1), keepdims=True)
    mixed = rgb.astype(np.float32) * (1.0 - amount) + mean * amount
    return np.clip(mixed, 0, 255).astype(np.uint8)


def resize_long_side(rgb: np.ndarray, side: int) -> np.ndarray:
    height, width = rgb.shape[:2]
    long_side = max(height, width)
    if long_side == side:
        return rgb
    scale = side / float(long_side)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    size = (max(32, int(round(width * scale))), max(32, int(round(height * scale))))
    return cv2.resize(rgb, size, interp)


class OnnxFlexGlue:
    """Descriptor + sparse matcher. CPU execution provider only."""

    def __init__(self, cfg: dict):
        import onnxruntime as ort

        ort.set_default_logger_severity(3)
        ensure_dirs()
        up = cfg["upstream"]
        desc_path = download(up["onnx_descriptor"], os.path.join(WORK, "cache", "flexglue_descriptor.onnx"), up["descriptor_sha256"])
        match_path = download(up["onnx_matcher"], os.path.join(WORK, "cache", "flexglue_matcher.onnx"), up["matcher_sha256"])
        threads = int(cfg["runtime"]["intra_op_threads"])
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        providers = ["CPUExecutionProvider"]
        self.descriptor = ort.InferenceSession(desc_path, sess_options=options, providers=providers)
        self.matcher = ort.InferenceSession(match_path, sess_options=options, providers=providers)
        for session, name in ((self.descriptor, "descriptor"), (self.matcher, "matcher")):
            if session.get_providers() != ["CPUExecutionProvider"]:
                raise RuntimeError(f"{name} providers {session.get_providers()} are not CPU-only")
        in_shape = self.descriptor.get_inputs()[0].shape
        self.height = int(in_shape[-2])
        self.width = int(in_shape[-1])
        self._feat: dict[str, np.ndarray] = {}
        self._feat_s: dict[str, float] = {}
        self._sift: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._sift_s: dict[str, float] = {}

    def describe(self, rgb: np.ndarray) -> tuple[np.ndarray, float, float, bool]:
        """Return features, this-call seconds, first-seen seconds, and whether it was cached."""
        digest = image_digest(rgb)
        cached = digest in self._feat
        if cached:
            return self._feat[digest], 0.0, self._feat_s[digest], True
        resized = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        tensor = resized.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        t0 = time.perf_counter()
        (features,) = self.descriptor.run(None, {"image": tensor})
        elapsed = time.perf_counter() - t0
        self._feat[digest] = features
        self._feat_s[digest] = elapsed
        return features, elapsed, elapsed, False

    def detect(self, rgb: np.ndarray, max_detect: int) -> tuple[np.ndarray, np.ndarray, float, float, bool]:
        """Return keypoints, descriptors, this-call seconds, first-seen seconds, cached."""
        digest = image_digest(rgb)
        cached = digest in self._sift
        if cached:
            kpts, desc = self._sift[digest]
            return kpts, desc, 0.0, self._sift_s[digest], True
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        t0 = time.perf_counter()
        sift = cv2.SIFT_create(nfeatures=max_detect)
        keypoints, desc = sift.detectAndCompute(gray, None)
        elapsed = time.perf_counter() - t0
        if not keypoints or desc is None:
            kpts = np.zeros((0, 2), np.float32)
            desc = np.zeros((0, 128), np.float32)
        else:
            order = sorted(range(len(keypoints)), key=lambda i: -keypoints[i].response)
            pts = np.array([keypoints[i].pt for i in order], dtype=np.float32)
            desc = np.asarray(desc, dtype=np.float32)[order]
            _, first = np.unique(pts, axis=0, return_index=True)
            keep = np.sort(first)
            kpts = pts[keep][:max_detect]
            desc = desc[keep][:max_detect]
        self._sift[digest] = (kpts, desc)
        self._sift_s[digest] = elapsed
        return kpts, desc, elapsed, elapsed, False

    def match(
        self,
        features0: np.ndarray,
        features1: np.ndarray,
        kpts0: np.ndarray,
        kpts1: np.ndarray,
        size0: tuple[int, int],
        size1: tuple[int, int],
        filter_threshold: float,
        max_distance_px: float,
        min_covisibility: float,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        if len(kpts0) == 0 or len(kpts1) == 0:
            empty = np.zeros((0, 2), np.int64)
            return empty, np.zeros(0, np.float32), float("nan"), 0.0
        # OpenCV puts the top-left pixel centre at (0, 0); FlexGlue uses (0.5, 0.5).
        feeds = {
            "features0": features0,
            "features1": features1,
            "kpts0": (kpts0 + 0.5)[None].astype(np.float32),
            "kpts1": (kpts1 + 0.5)[None].astype(np.float32),
            "image_size0": np.array([size0], dtype=np.float32),
            "image_size1": np.array([size1], dtype=np.float32),
            "filter_threshold": np.array(filter_threshold, dtype=np.float32),
            "max_distance_px": np.array(max_distance_px, dtype=np.float32),
            "min_covisibility": np.array(min_covisibility, dtype=np.float32),
        }
        t0 = time.perf_counter()
        matches0, mscores0, covis = self.matcher.run(None, feeds)
        elapsed = time.perf_counter() - t0
        matches0 = matches0[0]
        mscores0 = mscores0[0]
        valid = matches0 > -1
        if not np.any(valid):
            empty = np.zeros((0, 2), np.int64)
            return empty, np.zeros(0, np.float32), float(covis[0]), elapsed
        pairs = np.stack([np.flatnonzero(valid), matches0[valid]], axis=-1).astype(np.int64)
        return pairs, mscores0[valid].astype(np.float32), float(covis[0]), elapsed


def subsample(kpts: np.ndarray, desc: np.ndarray, n_keep: int, n_cand: int, seed: int, key: str) -> tuple[np.ndarray, np.ndarray]:
    if len(kpts) == 0:
        return kpts, desc
    capped = min(len(kpts), n_cand)
    kpts = kpts[:capped]
    desc = desc[:capped]
    if len(kpts) <= n_keep:
        return kpts, desc
    seq = np.random.SeedSequence([int(seed), zlib.adler32(key.encode("utf-8")) & 0xFFFFFFFF])
    rng = np.random.default_rng(seq)
    chosen = np.sort(rng.choice(len(kpts), size=n_keep, replace=False))
    return kpts[chosen], desc[chosen]


def sift_ratio_matches(desc0: np.ndarray, desc1: np.ndarray, ratio: float) -> np.ndarray:
    if len(desc0) < 2 or len(desc1) < 2:
        return np.zeros((0, 2), np.int64)
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn = matcher.knnMatch(desc0, desc1, k=2)
    keep = []
    for pair in knn:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < ratio * second.distance:
            keep.append((best.queryIdx, best.trainIdx))
    if not keep:
        return np.zeros((0, 2), np.int64)
    return np.asarray(keep, dtype=np.int64)


def fundamental_inliers(pts0: np.ndarray, pts1: np.ndarray, thresh_px: float) -> tuple[int, float]:
    if len(pts0) < 8:
        return 0, 0.0
    method = getattr(cv2, "USAC_MAGSAC", cv2.FM_RANSAC)
    _model, mask = cv2.findFundamentalMat(
        np.ascontiguousarray(pts0, dtype=np.float32),
        np.ascontiguousarray(pts1, dtype=np.float32),
        method,
        thresh_px,
        0.999,
    )
    if mask is None:
        return 0, 0.0
    count = int(mask.ravel().astype(bool).sum())
    return count, count / float(len(pts0))


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    ok = np.isfinite(scores)
    scores = scores[ok]
    labels = labels[ok]
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / (len(pos) * len(neg)))


def self_check() -> None:
    assert license_bucket("CC BY 4.0") == "by"
    assert license_bucket("CC BY-SA 4.0") == "sa"
    assert license_bucket("CC0 1.0") == "open"
    assert license_bucket("CC BY-NC 4.0") == "nc"
    scores = np.array([0.9, 0.2, 0.8, 0.1])
    labels = np.array([1, 0, 1, 0])
    assert abs(auroc(scores, labels) - 1.0) < 1e-9
    tied = auroc(np.array([0.5, 0.5]), np.array([1, 0]))
    assert abs(tied - 0.5) < 1e-9
    print("self-check ok", flush=True)


def draw_matches(
    rgb0: np.ndarray,
    rgb1: np.ndarray,
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    matches: np.ndarray,
    scores: np.ndarray,
    title: str,
    max_lines: int = 120,
) -> np.ndarray:
    if len(matches) > max_lines:
        order = np.argsort(-scores)[:max_lines]
        matches = matches[order]
        scores = scores[order]
    h0, w0 = rgb0.shape[:2]
    h1, w1 = rgb1.shape[:2]
    canvas = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0] = cv2.cvtColor(rgb0, cv2.COLOR_RGB2BGR)
    canvas[:h1, w0:] = cv2.cvtColor(rgb1, cv2.COLOR_RGB2BGR)
    for (i, j), score in zip(matches, scores):
        score_f = float(score)
        color = (0, int(255 * score_f), int(255 * (1.0 - score_f)))
        p0 = tuple(np.round(kpts0[int(i)]).astype(int))
        p1 = (int(round(kpts1[int(j)][0])) + w0, int(round(kpts1[int(j)][1])))
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 2, color, -1, cv2.LINE_AA)
    bar = np.full((36, canvas.shape[1], 3), 16, np.uint8)
    cv2.putText(bar, title[:180], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([bar, canvas])
