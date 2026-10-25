import pytest

from eventwise.captions import match_by_adjacency, match_by_geometry, parse_caption
from eventwise.normalise import clean_text, normalise_cell, parse_number


# X10: money / percentage parsing
@pytest.mark.parametrize("text, expected", [
    ("(£40,400)", (-40400.0, "GBP")),
    ("£1,479,600", (1479600.0, "GBP")),
    ("16.8%", (16.8, "%")),
    ("-2.7%", (-2.7, "%")),
    ("-£780.00", (-780.0, "GBP")),
    ("£5m", (5_000_000.0, "GBP")),
    ("632", (632.0, None)),
    (" 1,812 ", (1812.0, None)),
])
def test_parse_number(text, expected):
    assert parse_number(text) == expected


@pytest.mark.parametrize("text", [
    "9,420 (Sat 18 July, 20:00)", "10 m", "Per commission rate", "", None, "(£40", "12 Jun",
])
def test_parse_number_rejects_non_numeric(text):
    assert parse_number(text) is None


def test_normalised_cell_keeps_text_and_value_separately():
    cell = normalise_cell("(£40,400)\n")
    assert cell.text == "(£40,400)" and cell.value == -40400.0 and cell.unit == "GBP"
    assert normalise_cell(None) is None


def test_clean_text_expands_ligatures_and_whitespace():
    assert clean_text("e\ufb03cient  \n data") == "efficient data"


@pytest.mark.parametrize("text, kind, label", [
    ("Table 3: Expenditure, budget vs actual.", "table", "Table 3"),
    ("Figure 1.1.2", "figure", "Figure 1.1.2"),
    ("Tabl 1: Mesurau Cynllun Teithio", "table", "Tabl 1"),
    ("Ffigur 2: Hierarchaeth Teithio", "figure", "Ffigur 2"),
    ("Table 2a: Materials Camp Wales will buy", "table", "Table 2a"),
])
def test_parse_caption(text, kind, label):
    caption = parse_caption(text)
    assert caption and caption.kind == kind and caption.label == label


@pytest.mark.parametrize("text", [
    "Table 4 is an example rota for the Info Point team.",
    "See Figure 2 for details",
    "Tables and chairs are provided",
])
def test_running_text_is_not_a_caption(text):
    assert parse_caption(text) is None


def test_geometry_matches_caption_beside_object():
    # Caption to the right of a chart's bottom edge (AI Index style).
    chart = (36, 88, 332, 366)
    other = (36, 419, 576, 705)
    caption = parse_caption("Figure 1.1.2")
    result = match_by_geometry([(caption, (344, 355, 390, 366))], [("figure", chart), ("figure", other)])
    assert len(result) == 1 and result[0].object_index == 0


def test_geometry_leaves_distant_caption_unmatched():
    caption = parse_caption("Table 1: Far away")
    assert match_by_geometry([(caption, (0, 700, 100, 710))], [("table", (0, 0, 100, 100))]) == []


def test_adjacency_reports_ambiguity():
    cap = parse_caption("Table 2a: First")
    sequence = [("table", None), ("caption", cap), ("table", None)]
    assignments, ambiguous = match_by_adjacency(sequence)
    assert assignments[0].object_index == 0 and assignments[0].confidence == "medium"
    assert ambiguous == [1]
