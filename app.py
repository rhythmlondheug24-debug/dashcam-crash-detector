"""
Vehicle Proximity Detection — Gradio Web UI v2
================================================
Premium dark-themed interface with:
  - Video upload + processed output
  - Live progress bar with TTC display
  - Post-processing summary panel (crash moments, stats)
  - Danger frame screenshot gallery

Launch: python app.py
"""

import sys
from pathlib import Path

import gradio as gr

# ── Ensure the detector module is importable ────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from vehicle_proximity_detector import process_video


# ═══════════════════════════════════════════════════════════════════════════
# PROCESSING WRAPPER
# ═══════════════════════════════════════════════════════════════════════════

def run_detection(video_path, progress=gr.Progress(track_tqdm=True)):
    """
    Gradio handler: upload → detect → return (video, summary, screenshots).
    Wrapped in try/except so errors display in the UI, not crash the server.
    """
    if video_path is None:
        raise gr.Error("⚠️ Please upload a dashcam video first.")

    try:
        input_path = Path(video_path).resolve()
        if not input_path.exists():
            raise gr.Error(f"Uploaded file not found: {input_path}")

        output_path = input_path.parent / f"{input_path.stem}_tracked.mp4"

        def gradio_progress(fraction, description):
            progress(fraction, desc=description)

        # Run the detection pipeline
        result = process_video(
            input_path=str(input_path),
            output_path=str(output_path),
            progress_callback=gradio_progress,
        )

        out_video = result["output_path"]
        summary = result["summary"]
        screenshots = result["screenshots"]

        if not Path(out_video).exists():
            raise gr.Error("Processing completed but output file was not created.")

        # Gradio Gallery expects list of (filepath, caption) or just filepaths
        gallery_items = []
        for s in screenshots:
            fname = Path(s).stem
            gallery_items.append((s, fname))

        return out_video, summary, gallery_items

    except gr.Error:
        raise  # Re-raise Gradio errors as-is
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"❌ Processing failed: {type(e).__name__}: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — All styling lives here (Gradio 6 compatible)
# ═══════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
/* ── Global dark theme ───────────────────────────────────────────────── */
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
    background: #0a0a1a !important;
}
.dark .gradio-container, body {
    background: #0a0a1a !important;
}

/* ── Blocks and panels ───────────────────────────────────────────────── */
.block, .gr-block, .gr-box, .gr-panel,
div[class*="block"] {
    background: #111128 !important;
    border-color: rgba(99, 102, 241, 0.15) !important;
}

/* ── Header ──────────────────────────────────────────────────────────── */
#app-title {
    text-align: center;
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0f0f23 100%) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    border-radius: 16px;
    padding: 28px 20px 20px 20px;
    margin-bottom: 20px;
}
#app-title h1 {
    background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin-bottom: 4px;
}
#app-title p {
    color: #94a3b8;
    font-size: 0.95rem;
    margin: 0;
}

/* ── Process Button ──────────────────────────────────────────────────── */
#process-btn {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    color: white !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    padding: 14px 0 !important;
    border-radius: 12px !important;
    letter-spacing: 0.5px;
    transition: all 0.3s ease !important;
    text-transform: uppercase;
}
#process-btn:hover {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    transform: translateY(-1px);
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4) !important;
}

/* ── Info and panels ─────────────────────────────────────────────────── */
#status-box {
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
    border-radius: 12px;
    padding: 12px 16px;
}

.video-container {
    border-radius: 12px;
    overflow: hidden;
}

#summary-panel {
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
    border-radius: 12px;
    padding: 16px;
    min-height: 200px;
}

/* ── Labels and text ─────────────────────────────────────────────────── */
label, .gr-label, span[data-testid="block-label"] {
    color: #c4b5fd !important;
}

/* ── Inputs ──────────────────────────────────────────────────────────── */
input, textarea, select {
    background: #1a1a3e !important;
}

/* ── Footer ──────────────────────────────────────────────────────────── */
#footer-text {
    text-align: center;
    opacity: 0.5;
    font-size: 0.8rem;
    margin-top: 8px;
}
"""


# ═══════════════════════════════════════════════════════════════════════════
# BUILD UI
# ═══════════════════════════════════════════════════════════════════════════

def create_ui():
    """Build the Gradio Blocks interface."""

    with gr.Blocks(title="Vehicle Proximity Detector") as app:

        # ── Header ──────────────────────────────────────────────────────
        with gr.Column(elem_id="app-title"):
            gr.Markdown(
                "# 🚗 Vehicle Proximity Detector v2\n"
                "Upload dashcam footage → AI detects vehicles → "
                "Crash analysis with TTC estimation"
            )

        # ── Row 1: Video Input / Output ─────────────────────────────────
        with gr.Row(equal_height=True):

            with gr.Column(scale=1):
                video_input = gr.Video(
                    label="📁 Upload Dashcam Footage",
                    sources=["upload"],
                    elem_classes=["video-container"],
                    height=400,
                )

                process_btn = gr.Button(
                    "⚡ Process Video",
                    variant="primary",
                    elem_id="process-btn",
                    size="lg",
                )

                gr.Markdown(
                    "**Supported:** `.MP4`, `.MOV` &nbsp;|&nbsp; "
                    "**Model:** YOLOv11s &nbsp;|&nbsp; "
                    "**Inference:** FP16 CUDA\n\n"
                    "🟢 Green box = Safe &nbsp;|&nbsp; "
                    "🔴 Red box = Threat &nbsp;|&nbsp; "
                    "🟠 Orange label = TTC warning",
                    elem_id="status-box",
                )

            with gr.Column(scale=1):
                video_output = gr.Video(
                    label="🎯 Processed Proximity Alert Video",
                    elem_classes=["video-container"],
                    height=400,
                    interactive=False,
                )

                summary_output = gr.Markdown(
                    value="*Summary will appear here after processing...*",
                    label="📊 Analysis Summary",
                    elem_id="summary-panel",
                )

        # ── Row 2: Danger Screenshots ───────────────────────────────────
        with gr.Row():
            danger_gallery = gr.Gallery(
                label="📸 Top Danger Frame Screenshots",
                columns=5,
                rows=1,
                height=220,
                object_fit="contain",
                allow_preview=True,
            )

        # ── Footer ──────────────────────────────────────────────────────
        gr.Markdown(
            "Built with YOLO11s · ByteTrack · OpenCV · Gradio &nbsp;|&nbsp; "
            "GPU-accelerated CUDA inference &nbsp;|&nbsp; "
            "TTC estimation · Crash moment detection",
            elem_id="footer-text",
        )

        # ── Event Wiring ────────────────────────────────────────────────
        process_btn.click(
            fn=run_detection,
            inputs=[video_input],
            outputs=[video_output, summary_output, danger_gallery],
        )

    return app


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = create_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        css=CUSTOM_CSS,
        max_file_size="2gb",
    )
