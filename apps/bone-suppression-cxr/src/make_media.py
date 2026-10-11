"""Screenshot grid (PNG) and a short sweep video (MP4) from the synthetic CT-projection examples only
(CC BY-NC-SA 4.0, Qure.ai). JSRT images are NOT redistributed here (JSRT terms)."""
import os, sys, glob, subprocess, shutil
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__)); import bsx
W = bsx.WORK; ART = os.path.join(bsx.APP, "artifacts")
def L(name): return np.load(os.path.join(W, name))
def tile(img, text, size=384):
    t = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA); t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(t, (0, 0), (size, 26), (20, 20, 20), -1); cv2.putText(t, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    return t
rows = []
for k in ["cxr_train_1_a_1", "cxr_train_8_a_2"]:
    full, bone, lung = L(f"{k}_full.npy"), L(f"{k}_bone.npy"), L(f"{k}_lung.npy"); soft = full - bone
    rows.append(np.hstack([tile(bsx.to_u8(full), f"{k}: input"), tile(bsx.to_u8(bone, True), "bone (stretched)"),
                           tile(bsx.to_u8(soft), "soft tissue = full - bone"), tile(bsx.to_u8(lung, True), "lung component (stretched)"),
                           tile(bsx.to_u8(soft - lung), "non-lung soft = soft - lung")]))
grid = np.vstack(rows)
foot = np.full((30, grid.shape[1], 3), 20, np.uint8)
cv2.putText(foot, "qureaiorg/bone-suppression (CC BY-NC-SA 4.0, Qure.ai) on CPU | synthetic CT-derived examples | research use only, not for diagnosis", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
os.makedirs(ART, exist_ok=True); cv2.imwrite(os.path.join(ART, "bone-suppression-grid.png"), np.vstack([grid, foot]))

# sweep video: input vs bone output under increasing perturbation (breakpoint image cxr_train_1_a_1)
k = "cxr_train_1_a_1"; fr = os.path.join(W, "frames"); shutil.rmtree(fr, ignore_errors=True); os.makedirs(fr)
seq = [("clean", L(f"{k}_full.npy"), L(f"{k}_bone.npy"))]
order = [("noise_sigma", ["0.01", "0.03", "0.05", "0.1", "0.2", "0.4"]), ("lowres_px", ["512", "256", "128", "64"]),
         ("gamma", ["0.33", "3.0"]), ("rotation_deg", ["5", "15", "30", "90"]), ("inverted", ["True"])]
for f, levels in order:
    for v in levels:
        p = os.path.join(W, f"bp_{k}_{f}_{v}_in.npy")
        if os.path.exists(p): seq.append((f"{f}={v}", np.load(p), np.load(p.replace("_in.npy", "_bone.npy"))))
clean_bone = L(f"{k}_bone.npy")
import json; raw = json.load(open(os.path.join(bsx.APP, "experiments/exp1_cpu_stability/raw.json")))
ss = {f"{r['family']}={r['level']}": r["ssim_bone"] for r in raw["breakpoints"] if r["image"] == k}
lo, hi = np.percentile(clean_bone, [0.5, 99.5])
for i, (name, x, b) in enumerate(seq):
    bv = (np.clip((b - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)   # fixed stretch from the clean output
    s = ss.get(name, 1.0)
    im = np.hstack([tile(bsx.to_u8(x), f"input: {name}", 512), tile(bv, f"bone output  SSIM vs clean = {s:.3f}", 512)])
    for j in range(3): cv2.imwrite(os.path.join(fr, f"{i*3+j:04d}.png"), im)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "3", "-i", os.path.join(fr, "%04d.png"), "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", os.path.join(ART, "bone-suppression-sweep.mp4")], check=True)
print("wrote artifacts")
