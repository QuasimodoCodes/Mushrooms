"""
app_chonthichar.py — Chonthichar's Custom Mushroom Safety UI
=============================================================
Custom branded version of the Mushroom Safety System using YOLOv11n-cls.

Changes from original app.py:
- Custom branding with Chonthichar's name
- YOLOv11 model references
- Purple/green color theme
- Confidence meter visual
- Model info panel showing YOLOv11 stats

Usage:
    python app_chonthichar.py
"""

import gradio as gr
import sys
import os
import logging
import shutil
from datetime import datetime

# ── Project root ──────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)
from config import DRIFT_CONFIDENCE_THRESHOLD, PROMETHEUS_PORT, ACTIVE_LLM_PROVIDER

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

pipeline_dir = os.path.join(os.path.dirname(__file__), "pipeline")
sys.path.insert(0, pipeline_dir)

import prometheus_client
from prometheus_client import Counter

DRIFT_EVENTS = Counter(
    'mushroom_drift_events_total',
    'Total drift events (low confidence)',
    ['species']
)
prometheus_client.start_http_server(PROMETHEUS_PORT)

from predict    import predict_image
from integration import get_mushroom_context
from audit_layer import audit_prediction
from risk_engine import assess_risk


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_context_db(base_dir):
    for fname in ("mushroom_context.json", "mushroom_context.csv"):
        for base in (base_dir, os.path.dirname(os.path.dirname(base_dir))):
            p = os.path.join(base, "data", fname)
            if os.path.exists(p):
                return p
    return None


def log_drift_image(image_path, confidence, predicted_species):
    if confidence >= DRIFT_CONFIDENCE_THRESHOLD:
        return
    logger.warning(f"[DRIFT] Low confidence ({confidence:.2f}) for '{predicted_species}'")
    base_dir  = os.path.dirname(__file__)
    drift_dir = os.path.join(os.path.dirname(os.path.dirname(base_dir)), "data", "drift_images")
    os.makedirs(drift_dir, exist_ok=True)
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe       = predicted_species.replace(" ", "_").replace("/", "-")
    dest       = os.path.join(drift_dir, f"drift_{ts}_{safe}_conf{int(confidence*100)}.jpg")
    try:
        shutil.copy2(image_path, dest)
        DRIFT_EVENTS.labels(species=safe).inc()
    except Exception as e:
        logger.error(f"Drift save failed: {e}")


def _confidence_bar(conf_pct: float) -> str:
    """Return a simple text-based confidence meter."""
    filled = int(conf_pct / 5)   # 20 blocks total
    bar    = "█" * filled + "░" * (20 - filled)
    color  = "🟢" if conf_pct >= 70 else "🟡" if conf_pct >= 50 else "🔴"
    return f"{color} `{bar}` {conf_pct:.1f}%"


# ── CSS — purple/green Chonthichar theme ──────────────────────────────────────
CSS = """
/* ── Base ── */
body, .gradio-container { background: #0f0f1a !important; }

/* ── Header ── */
.chonthichar-header {
    background: linear-gradient(135deg, #3b0764 0%, #1a1a2e 60%, #0d3320 100%);
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 16px;
    border: 1px solid #7c3aed44;
}
.chonthichar-header h1 { color: #a78bfa; font-size: 2rem; margin: 0; }
.chonthichar-header p  { color: #d1d5db; margin: 6px 0 0; font-size: 0.95rem; }
.chonthichar-header .badge {
    display: inline-block;
    background: #7c3aed33;
    color: #c4b5fd;
    border: 1px solid #7c3aed66;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.8rem;
    margin-top: 8px;
}

/* ── Model info panel ── */
.model-info {
    background: #1e1b4b;
    border: 1px solid #4c1d95;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.85rem;
    color: #c4b5fd;
}
.model-info strong { color: #a78bfa; }

/* ── Panels ── */
.panel { border-radius: 8px; padding: 12px; background: #111827; }
.scroll-panel .prose { max-height: 72vh; overflow-y: auto; padding-right: 6px; }

/* ── Status bar ── */
.status-bar {
    font-size: 0.95rem; font-weight: 600;
    padding: 6px 10px; border-radius: 6px;
    background: #1e1b4b; color: #a78bfa;
}

/* ── Submit button ── */
.submit-btn { background: #7c3aed !important; color: white !important; }
.submit-btn:hover { background: #6d28d9 !important; }

/* ── Footer ── */
.footer {
    text-align: center;
    color: #6b7280;
    font-size: 0.8rem;
    margin-top: 16px;
    padding: 8px;
    border-top: 1px solid #1f2937;
}
"""


# ── Main pipeline function ────────────────────────────────────────────────────

def _run(image, season, location):
    H = lambda v, txt="": gr.update(visible=v, value=txt)

    if image is None:
        yield H(False), H(False), H(False), H(False), H(False)
        return

    base_dir  = os.path.dirname(__file__)
    csv_path  = _find_context_db(base_dir)

    # ── Step 1: YOLOv11 Vision ────────────────────────────────────────────────
    predicted_species, confidence = predict_image(image)

    if predicted_species is None:
        yield H(False), H(True, "❌ Vision API unavailable. Make sure it is running."), H(False), H(False), H(False)
        return

    formatted = predicted_species.replace("_", " ").title()
    log_drift_image(image, confidence, predicted_species)

    context = get_mushroom_context(formatted, csv_path) if csv_path else {}
    if "error" in context:
        context = {
            "toxicity_type": "Unknown", "habitat": "Unknown",
            "season":        "Unknown", "region": "Unknown",
            "key_warnings":  "Species not found in database. Treat as potentially dangerous.",
        }

    conf_pct = confidence * 100
    conf_bar = _confidence_bar(conf_pct)

    yolo_md = f"""## 🔍 YOLOv11 Identification Result

{conf_bar}

| Field | Value |
|---|---|
| **Species** | {formatted} |
| **Confidence** | {conf_pct:.1f}% |
| **Toxicity** | {context.get('toxicity_type', 'Unknown')} |
| **Habitat** | {context.get('habitat', 'Unknown')} |
| **Season** | {context.get('season', 'Unknown')} |
| **Region** | {context.get('region', 'Unknown')} |

> *Identified by YOLOv11n-cls — 87.6% Top-1 accuracy on 169 species*
"""

    yield H(False), H(True, yolo_md), H(True, "🧠 Requesting LLM visual audit — this takes 10–30s..."), H(False), H(False)

    # ── Step 2: LLM audit + risk engine ──────────────────────────────────────
    llm_verdict = audit_prediction(formatted, confidence, context, season, location, image_path=image)
    decision    = assess_risk(formatted, confidence, context, llm_verdict)

    risk_emoji = {"CRITICAL": "🚨", "HIGH": "⚠️", "MODERATE": "⚠️", "LOW": "✅"}
    rl = decision['risk_level']

    report  = f"## {risk_emoji.get(rl,'❓')} Risk Level: {rl}\n\n"
    report += f"**Recommendation:** {decision['recommendation']}\n\n"

    if decision['risk_factors']:
        report += "### Risk Factors\n"
        for rf in decision['risk_factors']:
            report += f"- {rf}\n"

    report += f"\n### LLM Audit\n{llm_verdict}\n"

    if context.get('key_warnings'):
        report += f"\n---\n⚠️ **Key Warning:** {context['key_warnings']}\n"

    report += "\n---\n*Safety assessment by LLM audit layer + deterministic risk engine*"

    yield H(False), H(True, yolo_md), H(False), H(True, report), H(True)


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="🍄 Mushroom Guardian — Chonthichar", css=CSS) as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="chonthichar-header">
        <h1>🍄 Mushroom Guardian</h1>
        <p>Upload a mushroom photo to identify the species and receive a safety assessment.</p>
        <span class="badge">⚡ Powered by YOLOv11n-cls · 87.6% Top-1 · 169 Species</span>
        <span class="badge" style="margin-left:8px">👤 Chonthichar Edition</span>
    </div>
    """)

    # ── Model info panel ──────────────────────────────────────────────────────
    gr.HTML("""
    <div class="model-info">
        🤖 <strong>Vision Model:</strong> YOLOv11n-cls &nbsp;|&nbsp;
        📊 <strong>Top-1:</strong> 87.6% &nbsp;|&nbsp;
        📊 <strong>Top-5:</strong> 97.8% &nbsp;|&nbsp;
        ⚡ <strong>Speed:</strong> 0.3ms/image &nbsp;|&nbsp;
        💾 <strong>Size:</strong> 3.6MB TFLite &nbsp;|&nbsp;
        🌿 <strong>Species:</strong> 169
    </div>
    """)

    gr.Markdown("---")

    # ── Inputs ────────────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="filepath",
                label="🍄 Drop a mushroom photo here",
                height=240
            )
        with gr.Column(scale=1):
            season_input = gr.Dropdown(
                choices=["Spring", "Summer", "Autumn", "Winter"],
                value="Autumn",
                label="🗓️ Season"
            )
            location_input = gr.Textbox(
                value="Norway",
                label="📍 Location"
            )
            gr.Markdown(
                "> 💡 Providing accurate season and location improves the LLM safety audit."
            )
            with gr.Row():
                clear_btn  = gr.ClearButton(value="🗑️ Clear")
                submit_btn = gr.Button("🔍 Identify Mushroom", variant="primary",
                                       elem_classes=["submit-btn"])

    gr.Markdown("---")

    # ── Results ───────────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=1):
            yolo_status = gr.Textbox(
                value="", label="⏳ Vision Model",
                interactive=False, elem_classes=["status-bar"], visible=False
            )
            yolo_box = gr.Markdown(
                value="", label="🔍 YOLOv11 Identification",
                elem_classes=["scroll-panel"], visible=False
            )

        with gr.Column(scale=1):
            llm_status = gr.Textbox(
                value="", label="⏳ LLM Audit",
                interactive=False, elem_classes=["status-bar"], visible=False
            )
            llm_box = gr.Markdown(
                value="", label="🧠 Safety Report",
                elem_classes=["scroll-panel"], visible=False
            )
            flag_btn = gr.Button("🚩 Flag Result", visible=False)

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="footer">
        ⚠️ This tool is for educational purposes only. Never consume wild mushrooms based solely on AI identification.
        Always consult an expert mycologist. &nbsp;|&nbsp; 
        Developed by <strong>Chonthichar</strong> · ACIT4603 Group 4 · Oslo Metropolitan University
    </div>
    """)

    # ── Event binding ─────────────────────────────────────────────────────────
    submit_btn.click(
        fn=_run,
        inputs=[image_input, season_input, location_input],
        outputs=[yolo_status, yolo_box, llm_status, llm_box, flag_btn]
    )
    clear_btn.add([image_input, yolo_box, llm_box])


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)