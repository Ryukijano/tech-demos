"""Small Gradio UI: paste text → p(AI) + top named SAE feature attributions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr
import yaml

APP = Path(__file__).resolve().parents[1]
EXP = APP / "experiments" / "exp1_cpu_auditable"
sys.path.insert(0, str(APP / "src"))

from detector_util import explain_one, load_detector  # noqa: E402

DETECTOR = None
CFG = None


def _get_detector():
    global DETECTOR, CFG
    if DETECTOR is None:
        CFG = yaml.safe_load((EXP / "config.yaml").read_text())
        DETECTOR = load_detector(
            CFG["model"]["hub_id"],
            torch_threads=int(CFG["runtime"]["torch_threads"]),
        )
    return DETECTOR, CFG


def analyze(text: str, tau: float):
    text = (text or "").strip()
    if not text:
        return "Paste some text first.", [], ""
    det, cfg = _get_detector()
    out = explain_one(det, text, head=cfg["model"]["head"], tau=float(tau), highlights=True)
    summary = (
        f"**label = `{out.label}`** · **p(AI) = {out.p_ai:.3f}** · "
        f"logit={out.logit:.3f} · bias={out.bias:.3f} · "
        f"active features={len(out.features)} · τ={tau:.2f}"
    )
    rows = [
        [
            f"{f.importance:+.4f}",
            (f.description or f"feature#{f.id}")[:160],
            str(f.id),
        ]
        for f in out.features[:20]
    ]
    # Simple token highlight for the top positive feature, if any.
    html = "<p style='font-family:ui-monospace,monospace;line-height:1.6'>"
    if out.tokens and out.features:
        top = out.features[0]
        hot = {i for i, _ in (top.highlights or [])}
        for i, tok in enumerate(out.tokens):
            esc = (
                tok.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            if i in hot:
                html += (
                    f"<mark style='background:#f5b7b1;padding:0 2px' "
                    f"title='top feature'>{esc}</mark>"
                )
            else:
                html += esc
    else:
        html += (text[:2000].replace("<", "&lt;"))
    html += "</p>"
    return summary, rows, html


def build() -> gr.Blocks:
    demo_text = (
        "In this paper, we propose a novel framework that leverages synergistic "
        "paradigms to optimize holistic outcomes across diverse domains."
    )
    with gr.Blocks(title="SAE text detector (DetectorScope CPU)") as demo:
        gr.Markdown(
            "# Auditable SAE text detector\n"
            "Official [DetectorScope](https://github.com/google-deepmind/detectorscope) "
            "(`liamdugan/detector-scope`) on CPU. Each decision is a sum of named "
            "sparse-autoencoder features (arXiv:2610.30006)."
        )
        with gr.Row():
            text = gr.Textbox(label="Text", value=demo_text, lines=8)
            with gr.Column():
                tau = gr.Slider(0.0, 0.95, value=0.85, step=0.05, label="τ (interpretability)")
                btn = gr.Button("Detect", variant="primary")
                summary = gr.Markdown()
        table = gr.Dataframe(
            headers=["importance", "description", "feature_id"],
            label="Top feature attributions (+ pushes toward AI)",
            interactive=False,
        )
        html = gr.HTML(label="Top-feature token highlights")
        btn.click(analyze, inputs=[text, tau], outputs=[summary, table, html])
        demo.load(analyze, inputs=[text, tau], outputs=[summary, table, html])
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    _get_detector()
    build().launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
