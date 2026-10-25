from pathlib import Path

import pytest

from conftest import CREW, FESTIVAL, MULTI, extracted, require
from eventwise.extract import extract_document

RISK_FORM = MULTI / "Travel-Risk-Assessment-Form-March-2026-v6.docx"
ERA = MULTI / "ERA_Travel-01.docx"
SURVEY = FESTIVAL / "08_Reports" / "19_Customer_Satisfaction_Survey_Results_2026.docx"
AGREEMENTS = sorted(CREW.glob("*/Agreement_*.docx")) if CREW.exists() else []


# X3: body order is preserved (heading immediately followed by its table)
@pytest.mark.parametrize("path", AGREEMENTS, ids=lambda p: p.parent.name[:20])
def test_signatures_heading_followed_by_table(path):
    doc = extracted(path)
    index = next(i for i, e in enumerate(doc.elements) if e.kind == "heading" and e.text == "Signatures")
    following = doc.elements[index + 1]
    if following.kind == "text" and "DRAFT" in following.text:  # draft notice paragraph
        following = doc.elements[index + 2]
    assert following.kind == "table"
    assert following.section_path == ["Signatures"]
    assert following.table.raw_cells[0][0].startswith("For Camp Wales Events Ltd")


# X4: text boxes are extracted exactly once (DrawingML + VML fallback deduplicated)
def test_text_box_extracted_once():
    doc = extracted(RISK_FORM)
    texts = [e.text for e in doc.elements] + [c for e in doc.by_kind("table") for row in e.table.raw_cells for c in row if c]
    assert sum(t.count("0292 874444") for t in texts) == 1
    box = next(e for e in doc.elements if "0292 874444" in e.text)
    assert box.location["container"] == "textbox"


# X5: legacy form checkboxes are detected with their state
def test_legacy_checkboxes():
    doc = extracted(ERA)
    boxes = [f for f in doc.form_fields if f["type"] == "legacy_form_checkbox"]
    assert len(boxes) == 5
    assert sorted(b["checked"] for b in boxes) == [False, False, False, False, True]
    checked = next(b for b in boxes if b["checked"])
    assert (checked["table"], checked["row"], checked["col"]) == (3, 2, 0)
    assert "\u2612" in doc.by_kind("table")[2].table.raw_cells[2][0]


def test_merged_cells_recorded_and_not_duplicated():
    doc = extracted(RISK_FORM)
    table = doc.by_kind("table")[2]
    assert table.table.merges
    for m in table.table.merges:
        assert table.table.raw_cells[m.row][m.col] is not None
        covered = [(r, c) for r in range(m.row, m.row + m.row_span) for c in range(m.col, m.col + m.col_span)
                   if (r, c) != (m.row, m.col)]
        assert all(table.table.raw_cells[r][c] is None for r, c in covered)
    assert not any(m.inferred for m in table.table.merges)


def test_numbered_bold_headings_without_heading_styles():
    doc = extracted(ERA)
    headings = [e.text for e in doc.by_kind("heading")]
    assert "1.General Information" in headings
    table = next(e for e in doc.by_kind("table") if e.text.startswith("Department"))
    assert table.section_path == ["1.General Information"]


def test_docx_figures_and_captions(tmp_path):
    doc = extract_document(require(SURVEY), out_dir=tmp_path)
    figures = doc.by_kind("figure")
    assert [f.label for f in figures] == ["Figure 1", "Figure 2", "Figure 3"]
    assert all(f.caption_match.confidence == "high" for f in figures)
    for f in figures:
        assert Path(tmp_path, doc.doc_id, f.figure.image_path).exists()
    assert [t.label for t in doc.by_kind("table")] == ["Table 1", "Table 2", "Table 3", "Table 4"]


def test_headers_footers_are_metadata_not_elements():
    doc = extracted(AGREEMENTS[0]) if AGREEMENTS else pytest.skip("no agreements")
    header = next(f["text"] for f in doc.furniture if f["kind"] == "header")
    assert all(header not in e.text for e in doc.elements)


def test_docx_locations():
    doc = extracted(SURVEY)
    for e in doc.elements:
        assert e.page is None and "block" in e.location
    table = doc.by_kind("table")[0]
    assert table.location["table"] == 1 and table.section_path


# X1 (DOCX part): every caption in the festival DOCX files is bound
@pytest.mark.parametrize("path", sorted(FESTIVAL.rglob("*.docx")) if FESTIVAL.exists() else [], ids=lambda p: p.name)
def test_festival_docx_captions_bound(path):
    doc = extracted(path)
    assert doc.status == "ok"
    assert [i for i in doc.issues if i.code == "caption_unmatched"] == []
