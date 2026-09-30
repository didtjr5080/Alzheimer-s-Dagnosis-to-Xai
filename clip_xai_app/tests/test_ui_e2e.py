"""True UI end-to-end tests: launch the actual `gr.Blocks` app returned by
`app.py::build_app` (the same object `main()` serves) on an ephemeral local
port and drive it over HTTP via `gradio_client`, the way a real browser
would. This is distinct from the rest of the suite (e.g.
test_merged_live_adapters.py), which calls the underlying handler functions
(`analyze_merged_scan`, `build_merged_full_pdf_report`, ...) directly and so
never exercises the actual `.click()` wiring, component types, or the
client/server serialization boundary."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "clip_xai_app"
sys.path.insert(0, str(APP_ROOT))

MERGED_ROOT = REPO_ROOT / "merged_project"
_merged_available = (MERGED_ROOT / "unified_manifest.csv").exists()

pytestmark = pytest.mark.skipif(not _merged_available, reason="merged_project/ not available locally")

SAMPLE_SCAN_ID = "OAS30564_MR_d0000"  # OASIS3, test split, true_label=CN


@pytest.fixture(scope="module")
def running_app():
    from app import build_app

    demo = build_app()
    demo.launch(prevent_thread_lock=True, quiet=True, show_error=True, inbrowser=False)
    try:
        yield demo
    finally:
        demo.close()


def test_merged_tab_scan_id_analysis_via_real_http_api(running_app):
    from gradio_client import Client

    client = Client(running_app.local_url)
    status_markdown, branch_table, gallery, pdf_path, images_zip_path, status_box = client.predict(
        SAMPLE_SCAN_ID, None, api_name="/analyze_merged_ui",
    )

    assert SAMPLE_SCAN_ID in status_markdown
    assert status_box == "Analysis completed."
    assert branch_table["data"]  # non-empty rows returned over the wire
    assert len(gallery) >= 3  # CLIP original/heatmap/overlay at minimum
    assert Path(pdf_path).exists()
    assert Path(pdf_path).suffix == ".pdf"
    assert Path(images_zip_path).exists()
    assert Path(images_zip_path).suffix == ".zip"


def test_merged_tab_image_upload_via_real_http_api(running_app):
    from gradio_client import Client, handle_file

    client = Client(running_app.local_url)
    image_path = str(MERGED_ROOT / "slices_multi" / f"{SAMPLE_SCAN_ID}_cor080.png")
    status_markdown, branch_table, gallery, pdf_path, images_zip_path, status_box = client.predict(
        "", handle_file(image_path), api_name="/analyze_merged_ui",
    )

    assert status_box == "Analysis completed."
    assert "사용자 업로드" in status_markdown or "업로드" in status_markdown
    assert branch_table["data"]
    assert Path(pdf_path).exists()
    assert Path(images_zip_path).exists()


def test_merged_tab_rejects_unknown_scan_id_without_crashing_the_server(running_app):
    from gradio_client import Client

    client = Client(running_app.local_url)
    status_markdown, _branch_table, gallery, pdf_path, images_zip_path, _status_box = client.predict(
        "definitely_not_a_real_scan_id", None, api_name="/analyze_merged_ui",
    )

    assert "Analysis failed" in status_markdown
    assert pdf_path is None
    assert images_zip_path is None
    assert gallery == []
