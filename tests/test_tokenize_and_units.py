from eventwise.config import IndexConfig
from eventwise.extraction.captions import find_label_mentions, label_key
from eventwise.indexing.units import build_units, document_title, units_fingerprint
from eventwise.retrieval.tokenize import tokenize
from retrieval_helpers import figure, make_doc, matched, table


def test_tokenize_keeps_identifiers_and_parts():
    tokens = tokenize("See agreement CW-AGR-2026-003 and invoice INV-CES-26-0388.")
    assert "cw-agr-2026-003" in tokens
    assert "ces-26-0388" in tokens  # a sub-identifier matches the shorter form used elsewhere
    assert {"cw", "agr", "ces"} <= set(tokens)


def test_tokenize_canonicalises_numbers_and_money():
    assert "176400" in tokenize("Total contract value £176,400 ex VAT")
    assert "50400" in tokenize("£50,400.00")
    assert "16.8" in tokenize("+16.8%")
    assert tokenize("£176,400") == tokenize("176400")


def test_tokenize_words():
    tokens = tokenize("The presentations and categories of the University's policies")
    assert tokens == ["presentation", "category", "university", "policy"]
    assert tokenize("Café ŵyr") == ["cafe", "wyr"]


def test_label_keys_and_mentions():
    assert label_key("Figure 1.1.2") == "figure:1.1.2"
    assert label_key("Tabl 3") == label_key("Table 3") == "table:3"
    assert find_label_mentions("as shown in Figure 3 and Table 2a, not Figure 30.") == {"figure:3", "table:2a",
                                                                                      "figure:30"}


def _doc():
    rows = [["Category", "Thu", "Fri", "Sat", "Sun"]] + [[f"Cat {i}", "1", "2", "3", str(i)] for i in range(20)]
    return make_doc("d" * 16, "report_2026.pdf", title="Annual Report 2026", elements=[
        {"kind": "heading", "text": "1. Safety", "heading_level": 1},
        {"kind": "text", "text": "Sunday was hottest; see Figure 3 for the daily maximum.", "section_path": ["1. Safety"]},
        {"kind": "figure", "text": "Figure 3: Maximum temperature.", "label": "Figure 3",
         "caption": "Figure 3: Maximum temperature.", "caption_match": matched(), "figure": figure(),
         "section_path": ["1. Safety"], "bbox": (10, 20, 300, 400)},
        {"kind": "table", "text": "big table", "label": "Table 1", "caption": "Table 1: Cases by day.",
         "table": table(rows), "section_path": ["1. Safety"], "page": 2},
        {"kind": "figure", "text": "", "figure": figure("images/p2_fig2.png"), "page": 2},
    ])


def test_units_keep_element_ids_and_metadata():
    doc = _doc()
    units = build_units(doc)
    ids = {e["element_id"] for e in doc["elements"]}
    assert {u.element_id for u in units} == ids  # every element is indexed, nothing invented
    fig = next(u for u in units if u.label == "Figure 3")
    assert fig.page == 1 and fig.bbox == [10, 20, 300, 400] and fig.caption_confidence == "high"
    assert fig.image_path == "dddddddddddddddd/images/p1_fig1.png"
    assert fig.section_path == ["1. Safety"]
    assert "Document: Annual Report 2026" in fig.search_text and "Section: 1. Safety" in fig.search_text


def test_figure_borrows_citing_text_and_links_both_ways():
    units = build_units(_doc())
    text = next(u for u in units if u.kind == "text")
    fig = next(u for u in units if u.label == "Figure 3")
    assert fig.refs == [] and fig.cited_by == [text.element_id] and text.refs == [fig.element_id]
    assert "Sunday was hottest" in fig.content and fig.context_sources == [text.element_id]


def test_large_table_gets_row_groups_linked_to_the_table():
    cfg = IndexConfig(table_rows_threshold=8, table_group_rows=6)
    units = [u for u in build_units(_doc(), cfg) if u.kind == "table"]
    whole, groups = units[0], units[1:]
    assert whole.rows is None and whole.is_primary and "Cat 19" in whole.content
    assert len(groups) == 4 and all(g.element_id == whole.element_id for g in groups)
    assert groups[0].rows == [1, 6] and groups[-1].rows == [19, 20]
    for g in groups:  # each group repeats caption and column header so it stands alone
        assert g.content.startswith("Table 1: Cases by day.\nColumns: Category | Thu | Fri | Sat | Sun")
    covered = {r for g in groups for r in range(g.rows[0], g.rows[1] + 1)}
    assert covered == set(range(1, 21))


def test_element_without_text_is_flagged_not_dropped():
    units = build_units(_doc())
    bare = next(u for u in units if u.kind == "figure" and u.label is None)
    assert bare.low_text and bare.content  # placeholder text, but still retrievable/inspectable


def test_long_text_is_windowed():
    words = " ".join(f"w{i}" for i in range(400))
    doc = make_doc("e" * 16, "long.pdf", [{"kind": "text", "text": words}])
    units = build_units(doc, IndexConfig(max_unit_words=160, chunk_overlap_words=30))
    assert len(units) == 3 and {u.element_id for u in units} == {doc["elements"][0]["element_id"]}
    assert units[0].part == "chunk 1/3" and units[0].is_primary and not units[1].is_primary
    assert "w399" in units[-1].content


def test_table_of_contents_lines_flagged():
    doc = make_doc("f" * 16, "toc.pdf", [{"kind": "text", "text": "3.1 Requests for Travel ........ 19"},
                                          {"kind": "text", "text": "Travel must be booked early."}])
    toc, body = build_units(doc)
    assert toc.navigation and not body.navigation


def test_fingerprint_changes_with_content_and_config():
    doc = _doc()
    a = units_fingerprint(build_units(doc), IndexConfig())
    assert a == units_fingerprint(build_units(doc), IndexConfig())
    assert a != units_fingerprint(build_units(doc), IndexConfig(context_words=10))
    doc["elements"][1]["text"] += " Changed."
    assert a != units_fingerprint(build_units(doc), IndexConfig())


def test_document_title_falls_back_to_file_name():
    assert document_title({"doc_name": "14_Festival_Budget.pdf", "metadata": {"title": "Microsoft Word - x.docx"}}) \
        == "14 Festival Budget"
    assert document_title({"doc_name": "a.pdf", "metadata": {"title": "Festival Budget"}}) == "Festival Budget (a)"
