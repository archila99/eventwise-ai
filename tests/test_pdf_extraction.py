import re
from pathlib import Path

import pytest

from conftest import AI_INDEX, CREW, FESTIVAL, MULTI, extracted, require
from eventwise.extract import extract_document

BUDGET = FESTIVAL / "06_Commercial_and_Financial" / "14_Festival_Budget_2026_Actual_vs_Budget.pdf"
POST_EVENT = FESTIVAL / "08_Reports" / "18_Post_Event_Report_2026.pdf"
TRAVEL_POLICY = MULTI / "Travel-and-Expenses-Policy.pdf"
AI_SAMPLE = (13, 17, 18, 19, 26, 100)


def _issues(doc, code):
    return [i for i in doc.issues if i.code == code]


# X2: table cells keep word spacing in Word-generated PDFs
def test_table_cells_keep_word_spaces():
    doc = extracted(TRAVEL_POLICY, (3,))
    table = next(e for e in doc.elements if e.kind == "table")
    assert table.table.raw_cells[0][:2] == ["Person Type", "Description"]
    assert table.table.raw_cells[1][1].startswith("Individuals contracted to work")


def test_table_borders_are_not_figures():
    doc = extracted(TRAVEL_POLICY, (29,))
    assert [e.kind for e in doc.elements] == ["heading", "table"]


# X6: repeated headers/footers are not indexed as text
@pytest.mark.parametrize("path, pages", [(BUDGET, None), (POST_EVENT, None), (AI_INDEX, AI_SAMPLE)])
def test_headers_and_footers_removed(path, pages):
    doc = extracted(path, pages)
    heights = {p["page"]: p["height"] for p in doc.pages}
    furniture = re.compile(r"^(page \d+ of \d+|\d+|.*AI INDEX REPORT 2026)$", re.IGNORECASE)
    in_margin = lambda e: e.bbox[3] < 0.08 * heights[e.page] or e.bbox[1] > 0.92 * heights[e.page]  # noqa: E731
    offenders = [e.text for e in doc.elements
                 if e.kind in ("text", "heading") and furniture.match(e.text) and in_margin(e)]
    assert offenders == []
    assert doc.furniture, "repeated header/footer lines should be recorded, not silently dropped"


# X7: physical page index and printed page label are both kept
def test_ai_index_page_labels():
    doc = extracted(AI_INDEX, (18, 74, 100))
    labels = {p["page"]: p["label"] for p in doc.pages}
    assert labels == {18: "17", 74: "73-74", 100: "100"}  # PDF p. 74 is a double-width spread
    assert all(e.page in (18, 74, 100) for e in doc.elements)


# X8: the benchmark table embedded as an image on PDF p. 26 is found as a figure
def test_ai_index_raster_table_detected():
    doc = extracted(AI_INDEX, AI_SAMPLE)
    figures = [e for e in doc.elements if e.kind == "figure" and e.page == 26]
    assert len(figures) == 1
    assert figures[0].bbox == pytest.approx((41.05, 499.97, 465.1, 686.99), abs=2)
    assert figures[0].figure.source_type == "raster"
    assert figures[0].label == "Figure 1.1.15"


# X9: decorative full-page images are not figures
def test_no_page_sized_figures():
    doc = extracted(AI_INDEX, AI_SAMPLE)
    sizes = {p["page"]: p["width"] * p["height"] for p in doc.pages}
    for fig in doc.by_kind("figure"):
        x0, y0, x1, y1 = fig.bbox
        assert (x1 - x0) * (y1 - y0) < 0.95 * sizes[fig.page]
    assert _issues(doc, "background_image_skipped"), "skipped backgrounds should be reported"


def test_ai_index_vector_charts_and_captions():
    doc = extracted(AI_INDEX, AI_SAMPLE)
    figures = {e.label: e for e in doc.by_kind("figure") if e.page == 18}
    assert set(figures) == {"Figure 1.1.2", "Figure 1.1.3"}
    chart = figures["Figure 1.1.2"]
    assert chart.figure.source_type == "vector"
    assert "59, United States" in chart.figure.text_labels
    assert chart.caption_match.confidence == "high"
    assert chart.section_path[-2:] == ["1.1 Notable AI Models", "By National Affiliation2"]


def test_uncaptioned_figure_is_reported():
    doc = extracted(AI_INDEX, AI_SAMPLE)
    uncaptioned = next(i for i in doc.issues if i.code == "objects_without_caption")
    assert any(":p100:fig" in eid for eid in uncaptioned.location["element_ids"])


# X1 (PDF part): every caption in the festival PDFs is bound to an object
@pytest.mark.parametrize("path", sorted(FESTIVAL.rglob("*.pdf")) if FESTIVAL.exists() else [], ids=lambda p: p.name)
def test_festival_pdf_captions_bound(path):
    doc = extracted(path)
    assert doc.status == "ok"
    assert _issues(doc, "caption_unmatched") == []
    assert all(e.caption_match.confidence != "low" for e in doc.elements if e.caption)


def test_raw_and_normalised_table_values_are_separate():
    doc = extracted(BUDGET)
    table = next(e for e in doc.elements if e.label == "Table 2").table
    row = table.raw_cells[1]
    assert row[0] == "Ticket sales (incl. camping, parking & glamping\nadd-ons)"  # raw keeps line break
    assert row[3] == "(£40,400)"
    norm = table.normalised_cells[1]
    assert norm[0].text == "Ticket sales (incl. camping, parking & glamping add-ons)"
    assert (norm[3].value, norm[3].unit) == (-40400.0, "GBP")
    assert table.cell_bboxes[1][3] is not None


def test_source_metadata_on_every_element():
    doc = extracted(BUDGET)
    for e in doc.elements:
        assert e.page in (1, 2) and e.bbox and e.element_id.startswith(doc.doc_id[:10])
    table2 = next(e for e in doc.elements if e.label == "Table 2")
    assert table2.section_path[-1] == "2. Income"
    assert [e.order for e in doc.elements] == list(range(len(doc.elements)))


# X12 / refinement 4: cross-page tables are flagged, never merged
def test_cross_page_table_flagged_not_merged():
    doc = extracted(BUDGET)
    p1, p2 = (next(e for e in doc.elements if e.kind == "table" and e.page == p and e.text.startswith("Expenditure line"))
              for p in (1, 2))
    assert p1.element_id != p2.element_id
    hint = next(i for i in doc.issues if i.code == "possible_table_continuation")
    assert hint.location["element_ids"] == [p1.element_id, p2.element_id]


def test_cross_page_hint_with_different_column_counts():
    doc = extracted(CREW / "04_Dyffryn_Sound_and_Light_production_contractor" / "Crew_List.pdf")
    hints = [i for i in doc.issues if i.code == "possible_table_continuation"]
    assert hints and "different column count" in hints[0].location["evidence"]


def test_figure_crops_are_saved(tmp_path):
    doc = extract_document(require(BUDGET), out_dir=tmp_path)
    figures = doc.by_kind("figure")
    assert len(figures) == 2
    for fig in figures:
        image = Path(tmp_path, doc.doc_id, fig.figure.image_path)
        assert image.exists() and image.stat().st_size > 1000
