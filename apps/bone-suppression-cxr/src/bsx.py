"""Shared pipeline for the bone-suppression-cxr demo.

Pre/post-processing is adapted from the authors' reference script `suppress.py` in
https://huggingface.co/qureaiorg/bone-suppression (Apache-2.0, Qure.ai) and matches the HF Space
hugging-apps/bone-suppression-chest-xray. Weights are CC BY-NC-SA 4.0 and are downloaded at run
time (never committed). Research use only; not a medical device; not for diagnosis.
"""
import os, time
import numpy as np
import cv2
import torch

MODEL_REPO = "qureaiorg/bone-suppression"
SPACE_REPO = "hugging-apps/bone-suppression-chest-xray"
BONE_FILE = "weights/bone_suppression.ts"
LUNG_FILE = "weights/lung_component_suppression.ts"
SIZE = 1024
HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
WORK = os.path.join(APP, "work")          # git-ignored (apps/*/work/)


def fetch(src):
    """'hf:<file in model repo>' or 'hf-space:<file in space repo>' -> local path (HF cache)."""
    from huggingface_hub import hf_hub_download
    kind, rest = src.split(":", 1)
    if kind == "hf":
        repo, f = rest.split("/", 2)[:2], rest.split("/", 2)[2]
        return hf_hub_download("/".join(repo), f)
    if kind == "hf-space":
        repo, f = rest.split("/", 2)[:2], rest.split("/", 2)[2]
        return hf_hub_download("/".join(repo), f, repo_type="space")
    return src


def load_model(which):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(MODEL_REPO, BONE_FILE if which == "bone" else LUNG_FILE)
    return torch.jit.load(p, map_location="cpu").eval()


def read_gray(path):
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"cannot read {path}")
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr[..., :3], cv2.COLOR_BGR2GRAY)
    return arr.astype(np.float32)


def minmax(x):
    lo, hi = float(x.min()), float(x.max())
    return ((x - lo) / (hi - lo + 1e-10)).astype(np.float32)


def preprocess(arr):
    """Resize to 1024x1024 (INTER_AREA) and per-image min-max to [0,1]."""
    return minmax(cv2.resize(arr, (SIZE, SIZE), interpolation=cv2.INTER_AREA))


def first(out):
    if isinstance(out, (tuple, list)):
        out = out[0]
    if isinstance(out, dict):
        out = next(iter(out.values()))
    return out


@torch.inference_mode()
def run(model, x):
    """x: (1024,1024) float32 -> (output float32, seconds)."""
    t = time.perf_counter()
    y = first(model(torch.from_numpy(np.ascontiguousarray(x))[None, None]))[0, 0].float().numpy()
    return y, time.perf_counter() - t


def to_u8(x, stretch=False):
    y = np.asarray(x, np.float32)
    if stretch:  # robust 0.5-99.5 percentile stretch for viewing (as in the HF Space)
        lo, hi = (float(v) for v in np.percentile(y, [0.5, 99.5]))
        if hi - lo > 1e-6:
            y = (y - lo) / (hi - lo)
    return (np.clip(y, 0, 1) * 255).round().astype(np.uint8)


def looks_inverted(arr):
    """Space heuristic: mediastinum darker than lung fields => bone-dark (inverted) input."""
    h, w = arr.shape
    y0, y1 = int(0.40 * h), int(0.62 * h)
    c = arr[y0:y1, int(0.46 * w):int(0.54 * w)]
    l = arr[y0:y1, int(0.20 * w):int(0.32 * w)]
    r = arr[y0:y1, int(0.68 * w):int(0.80 * w)]
    rng = float(arr.max() - arr.min()) + 1e-10
    return bool(((np.median(l) + np.median(r)) / 2 - np.median(c)) / rng > 0.04)


# ---------- perturbations (all act on a normalised 1024px image, then re-apply min-max) ----------
def add_noise(x, sigma, rng):
    return minmax(x + rng.normal(0.0, sigma, x.shape).astype(np.float32))


def lowres(x, px):
    s = cv2.resize(x, (px, px), interpolation=cv2.INTER_AREA)
    return minmax(cv2.resize(s, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC))


def gamma(x, g):
    return minmax(np.power(np.clip(x, 0, 1), g))


def contrast(x, k):
    return minmax(np.clip((x - 0.5) * k + 0.5, 0, 1))


def rotate(x, deg):
    M = cv2.getRotationMatrix2D((SIZE / 2, SIZE / 2), deg, 1.0)
    return cv2.warpAffine(x, M, (SIZE, SIZE), flags=cv2.INTER_LINEAR, borderValue=0.0)


def jpeg(x, q):
    ok, buf = cv2.imencode(".jpg", to_u8(x), [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return minmax(cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).astype(np.float32))


def invert(x):
    return minmax(1.0 - x)


def ssim(a, b):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a, b, data_range=float(a.max() - a.min())))


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else float(10 * np.log10((a.max() - a.min()) ** 2 / mse))
