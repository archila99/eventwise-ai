import base64
import json
from types import SimpleNamespace

from eventwise.answer import (
    ABSTAIN,
    DEFAULT_MODEL,
    MAX_IMAGES,
    TABLE_TRUNCATION_NOTE,
    answer_question,
)
from eventwise.ask import main
from eventwise.retrieve import ContextItem, RetrievalResponse, RetrievedEvidence, format_citation
from eventwise.units import IndexUnit

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def evidence(element_id="t1", kind="text", citation="Budget.pdf — p. 1", content="Security cost £176,400.",
             **overrides) -> RetrievedEvidence:
    fields = dict(
        rank=1, element_id=element_id, doc_id="d1", doc_name="Budget.pdf", source_path="data/Budget.pdf",
        kind=kind, score=0.2, selection_score=0.2, bm25_rank=1, bm25_score=2.0, dense_rank=1, dense_score=0.4,
        citation=citation, page=1, page_label=None, section_path=["Expenditure"], location={},
        label=None, caption=None, caption_confidence=None, content=content, matched_parts=[],
        image_path=None, element_ref={"document_json": "document.json", "element_id": element_id},
    )
    fields.update(overrides)
    return RetrievedEvidence(**fields)


def context(element_id, relation="cites", kind="figure", snippet="see the figure", page=99, **overrides) -> ContextItem:
    fields = dict(
        element_id=element_id, kind=kind, relation=relation, page=page, page_label="99",
        label="Wrong label", caption="Wrong caption", snippet=snippet, image_path=None, result_rank=None,
    )
    fields.update(overrides)
    return ContextItem(**fields)


def unit(element_id, **overrides) -> IndexUnit:
    fields = dict(
        unit_id=element_id, element_id=element_id, doc_id="d1", doc_name="Report.pdf", kind="figure",
        order=0, content="Figure 3", context="Document: Report", page=3, page_label="17",
        section_path=["Safety"], label="Figure 3", location={"page": 3}, low_text=True,
    )
    fields.update(overrides)
    return IndexUnit(**fields)


def response(results, query="What was the security spend?", warnings=None) -> RetrievalResponse:
    return RetrievalResponse(
        query=query, mode="hybrid", results=results, filters={}, config={}, timings_ms={},
        warnings=list(warnings or []), suppressed=[],
    )


class FakeRetriever:
    def __init__(self, results, elements=None, units=None, warnings=None, query="What was the security spend?"):
        self.response = response(results, query=query, warnings=warnings)
        self.elements = elements or {}
        self.units = list(units or [])
        self.primary = {}
        for index, item in enumerate(self.units):
            self.primary.setdefault(item.element_id, index)
        self.calls = []

    def search(self, query, top_k=None, documents=None, kinds=None, mode="hybrid", config=None):
        self.calls.append({"query": query, "top_k": top_k, "documents": documents, "mode": mode})
        return self.response

    def get_element(self, element_id):
        if element_id not in self.elements:
            raise KeyError(element_id)
        return self.elements[element_id]


class FakeClient:
    def __init__(self, content):
        self.completions = self
        self.calls = 0
        self.kwargs = None
        self._content = content
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        message = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class ExplodingClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        raise AssertionError("the API should not be called")


def model_json(**overrides) -> str:
    payload = {"answer": "Security cost £176,400.", "citations": ["t1"], "sufficient": True}
    payload.update(overrides)
    return json.dumps(payload)


def user_text(kwargs) -> str:
    content = kwargs["messages"][1]["content"]
    if isinstance(content, str):
        return content
    return "\n".join(part.get("text", "") for part in content if part.get("type") == "text")


def image_parts(kwargs) -> list[dict]:
    content = kwargs["messages"][1]["content"]
    if isinstance(content, str):
        return []
    return [part for part in content if part.get("type") == "image_url"]


def write_png(directory, name) -> str:
    path = directory / name
    path.write_bytes(PNG)
    return str(path)


def ask(retriever, client, query="What was the security spend?", **kwargs):
    return answer_question(retriever, query, client, **kwargs)


def test_prompt_includes_citation_content_and_context():
    figure = context("fig3", snippet="temperature rose through the weekend")
    hit = evidence(content="Security cost £176,400.", context=[figure], also_in=[{"element_id": "dup"}])
    source = unit("fig3", content="Figure 3 temperatures")
    retriever = FakeRetriever([hit], units=[source])
    client = FakeClient(model_json(citations=["t1", "fig3"]))
    result = ask(retriever, client)

    text = user_text(client.kwargs)
    assert "What was the security spend?" in text
    assert "Security cost £176,400." in text
    assert "Budget.pdf — p. 1" in text
    assert "temperature rose through the weekend" in text
    expected = format_citation("Report.pdf", 3, "17", ["Safety"], "Figure 3", {"page": 3})
    assert expected in text
    assert "99" not in text
    assert "Wrong label" not in text
    assert [item.element_id for item in result.citations] == ["t1", "fig3"]
    assert result.citations[0].citation == "Budget.pdf — p. 1"
    assert result.citations[1].citation == expected
    assert "JSON" in client.kwargs["messages"][0]["content"]


def test_result_citation_is_not_rebuilt_from_the_index_unit():
    hit = evidence(citation="KEEP THIS CITATION", context=[context("t1", relation="cites", result_rank=1)])
    retriever = FakeRetriever([hit], units=[unit("t1", page=99, page_label="99", doc_name="Other.pdf")])
    client = FakeClient(model_json())
    result = ask(retriever, client)
    assert result.citations[0].citation == "KEEP THIS CITATION"
    assert "99" not in user_text(client.kwargs)
    assert "Other.pdf" not in user_text(client.kwargs)


def test_table_grid_is_used_for_a_partial_hit():
    grid = [
        [{"text": "Line"}, {"text": "Actual"}],
        [{"text": "Security"}, {"text": "£176,400"}],
        [{"text": "Medical"}, {"text": "£58,300"}],
    ]
    hit = evidence(
        kind="table", citation="Budget.pdf — p. 1 — Table 3", content="PARTIAL ROW",
        caption="Table 3: Expenditure", matched_parts=[{"part": "rows 2-2 of 3"}],
    )
    retriever = FakeRetriever([hit], elements={"t1": {"table": {"normalised_cells": grid}}})
    client = FakeClient(model_json())
    ask(retriever, client)
    text = user_text(client.kwargs)
    assert "Medical | £58,300" in text
    assert "PARTIAL ROW" not in text
    assert text.count("Table 3: Expenditure") == 1
    assert "Matched rows 2-2 of 3" in text


def test_continuation_table_uses_its_own_source():
    parent = evidence(
        kind="table", content="Day | Count",
        context=[context("day2", relation="table_continuation", kind="table", snippet="continued", page=99)],
    )
    elements = {
        "t1": {"table": {"normalised_cells": [[{"text": "Day"}, {"text": "Count"}], [{"text": "Friday"}, {"text": "8"}]]}},
        "day2": {"table": {"normalised_cells": [[{"text": "Saturday"}, {"text": "20"}]]}},
    }
    day2 = unit(
        "day2", kind="table", doc_name="Incident.pdf", page=1, page_label=None, section_path=["Log"],
        label="Table 1", location={}, content="Saturday | 20", low_text=False,
    )
    retriever = FakeRetriever([parent], elements=elements, units=[day2])
    client = FakeClient(model_json(citations=["day2"]))
    result = ask(retriever, client)
    expected = format_citation("Incident.pdf", 1, None, ["Log"], "Table 1", {})
    assert "Saturday | 20" in user_text(client.kwargs)
    assert expected in user_text(client.kwargs)
    assert "99" not in user_text(client.kwargs)
    assert result.citations[0].citation == expected


def test_long_table_is_truncated():
    grid = [[{"text": f"r{i}"}, {"text": "value"}] for i in range(800)]
    hit = evidence(kind="table", content="PARTIAL")
    retriever = FakeRetriever([hit], elements={"t1": {"table": {"normalised_cells": grid}}})
    client = FakeClient(model_json())
    result = ask(retriever, client)
    text = user_text(client.kwargs)
    assert TABLE_TRUNCATION_NOTE in text
    assert "r0 | value" in text
    assert "r799" not in text
    assert any("truncated" in warning for warning in result.warnings)


def test_missing_table_falls_back_to_retrieved_text():
    hit = evidence(kind="table", content="Security | £176,400")
    retriever = FakeRetriever([hit])
    client = FakeClient(model_json())
    result = ask(retriever, client)
    assert "Security | £176,400" in user_text(client.kwargs)
    assert any("could not be loaded" in warning for warning in result.warnings)


def test_unknown_citations_are_dropped_and_ids_are_normalised():
    hit = evidence(also_in=[{"element_id": "dup"}])
    retriever = FakeRetriever([hit])
    client = FakeClient(model_json(citations=["[t1]", "nope", "t1", "dup", 3]))
    result = ask(retriever, client)
    assert [item.element_id for item in result.citations] == ["t1"]
    assert result.sufficient is True
    dropped = next(warning for warning in result.warnings if warning.startswith("Dropped"))
    assert "nope" in dropped and "dup" in dropped


def test_unresolvable_context_id_is_not_given_a_citation():
    hit = evidence(context=[context("ghost", snippet="temperature rose", page=99)])
    retriever = FakeRetriever([hit])
    client = FakeClient(model_json(citations=["ghost"], answer="It was 99 degrees.", sufficient=True))
    result = ask(retriever, client)
    text = user_text(client.kwargs)
    assert "Source location unavailable" in text
    assert "99" not in text
    assert result.citations == []
    assert result.sufficient is False
    assert any("ghost" in warning for warning in result.warnings)


def test_docx_context_citation_has_no_invented_page():
    hit = evidence(context=[context("fig2", snippet="survey scores", page=4)])
    docx = unit(
        "fig2", doc_name="Survey.docx", page=None, page_label=None, section_path=["Results"],
        label="Figure 2", location={"paragraph": 4, "block": 2}, low_text=False, content="Figure 2 scores",
    )
    retriever = FakeRetriever([hit], units=[docx])
    client = FakeClient(model_json(citations=["fig2"]))
    result = ask(retriever, client)
    citation = result.citations[0].citation
    expected = format_citation("Survey.docx", None, None, ["Results"], "Figure 2", {"paragraph": 4, "block": 2})
    assert citation == expected
    assert "p." not in citation
    assert "paragraph 4" in citation
    assert "p. 4" not in user_text(client.kwargs)


def test_abstention_is_preserved_with_its_citation():
    retriever = FakeRetriever([evidence()])
    client = FakeClient(model_json(
        answer="Per-act fees are not stated. Artist fees in total were £512,000.",
        citations=["t1"], sufficient=False,
    ))
    result = ask(retriever, client)
    assert result.sufficient is False
    assert result.citations[0].element_id == "t1"
    assert "£512,000" in result.answer


def test_invalid_json_abstains_after_one_call():
    retriever = FakeRetriever([evidence()])
    client = FakeClient("I cannot answer that.")
    result = ask(retriever, client)
    assert client.calls == 1
    assert result.sufficient is False
    assert result.answer == ABSTAIN
    assert result.citations == []
    assert any("not valid JSON" in warning for warning in result.warnings)


def test_fenced_json_is_accepted():
    retriever = FakeRetriever([evidence()])
    body = model_json(answer="The cost was £176,400.")
    client = FakeClient(f"```json\n{body}\n```")
    result = ask(retriever, client)
    assert result.sufficient is True
    assert result.answer == "The cost was £176,400."


def test_sufficient_answer_without_citations_is_cleared():
    retriever = FakeRetriever([evidence()])
    client = FakeClient(model_json(citations=[], sufficient=True))
    result = ask(retriever, client)
    assert result.sufficient is False
    assert any("not treated as sufficiently supported" in warning for warning in result.warnings)


def test_empty_retrieval_does_not_call_the_api():
    retriever = FakeRetriever([], warnings=["No evidence found for this query with the current filters"])
    result = ask(retriever, ExplodingClient())
    assert result.sufficient is False
    assert result.answer == ABSTAIN
    assert result.model is None
    assert result.warnings == ["No evidence found for this query with the current filters"]


def test_text_question_does_not_attach_images(tmp_path):
    image = write_png(tmp_path, "fig.png")
    cited = write_png(tmp_path, "cited.png")
    figure = evidence(
        "fig", kind="figure", content="Figure 2: Largest expenditure lines.", low_text=False, image_path=image,
        context=[context("cited", image_path=cited, snippet="see Figure 9")],
    )
    text = evidence(context=[])
    cited_unit = unit("cited", content="Figure 9: a labelled diagram", low_text=False)
    retriever = FakeRetriever([text, figure], units=[cited_unit])
    client = FakeClient(model_json())
    result = ask(retriever, client)
    assert image_parts(client.kwargs) == []
    assert isinstance(client.kwargs["messages"][1]["content"], str)
    assert result.images == []


def test_low_text_figure_is_attached_for_an_ordinary_question(tmp_path):
    image = write_png(tmp_path, "temp.png")
    figure = evidence(
        "fig3", kind="figure", content="Figure without caption or text (raster)", low_text=True,
        image_path=image, citation="Report.pdf — p. 3 — Figure 3",
    )
    retriever = FakeRetriever([figure])
    client = FakeClient(model_json(citations=["fig3"], answer="Friday peaked at 22 °C."))
    result = ask(retriever, client, query="What was the maximum daytime temperature on Friday?")
    parts = image_parts(client.kwargs)
    assert len(parts) == 1
    assert parts[0]["image_url"]["detail"] == "high"
    assert base64.b64decode(parts[0]["image_url"]["url"].split(",", 1)[1]) == PNG
    assert "Image for element fig3:" in user_text(client.kwargs)
    assert result.images == ["fig3"]


def test_visual_question_attaches_a_figure_that_already_has_text(tmp_path):
    image = write_png(tmp_path, "models.png")
    figure = evidence(
        "fig112", kind="figure", low_text=False, image_path=image,
        content="Figure 1.1.2\nText in figure: United States, China, Europe",
    )
    retriever = FakeRetriever([figure])
    client = FakeClient(model_json(citations=["fig112"]))
    ask(retriever, client, query="According to Figure 1.1.2, how many models came from the United States?")
    assert len(image_parts(client.kwargs)) == 1


def test_extracted_chart_data_is_not_sent_as_an_image(tmp_path):
    image = write_png(tmp_path, "chart.png")
    figure = evidence(
        "chart", kind="figure", low_text=False, image_path=image,
        content="Chart data Score: Toilets: 3.1, Bar: 3.2",
    )
    retriever = FakeRetriever([figure])
    client = FakeClient(model_json(citations=["chart"]))
    result = ask(retriever, client, query="What does the chart show for toilets?")
    assert result.images == []
    assert image_parts(client.kwargs) == []


def test_image_cap_skips_a_missing_file_and_stops_at_two(tmp_path):
    first = tmp_path / "missing.png"
    second = write_png(tmp_path, "b.png")
    third = write_png(tmp_path, "c.png")
    fourth = write_png(tmp_path, "d.png")
    figures = [
        evidence("a", kind="figure", low_text=True, image_path=str(first), rank=1),
        evidence("b", kind="figure", low_text=True, image_path=second, rank=2),
        evidence("c", kind="figure", low_text=True, image_path=third, rank=3),
        evidence("d", kind="figure", low_text=True, image_path=fourth, rank=4),
    ]
    retriever = FakeRetriever(figures)
    client = FakeClient(model_json(citations=["b"]))
    result = ask(retriever, client, query="What was the temperature?")
    assert result.images == ["b", "c"]
    assert len(image_parts(client.kwargs)) == MAX_IMAGES
    assert any("missing" in warning and "a" in warning for warning in result.warnings)
    assert any("limit 2" in warning for warning in result.warnings)
    assert "d" not in result.images


def test_no_vision_sends_no_images(tmp_path):
    image = write_png(tmp_path, "temp.png")
    figure = evidence("fig3", kind="figure", low_text=True, image_path=image, content="Figure 3")
    retriever = FakeRetriever([figure])
    client = FakeClient(model_json(citations=["fig3"], sufficient=False, answer="The figure was not readable."))
    result = ask(retriever, client, query="What was the Friday temperature?", vision=False)
    assert isinstance(client.kwargs["messages"][1]["content"], str)
    assert "Figure images were not provided" in client.kwargs["messages"][1]["content"]
    assert result.images == []


def test_cited_low_text_figure_is_attached_and_a_neighbour_is_not(tmp_path):
    cited_path = write_png(tmp_path, "cited.png")
    neighbour_path = write_png(tmp_path, "next.png")
    hit = evidence(context=[
        context("cited-fig", image_path=cited_path, snippet="as shown in the figure"),
        context("next-fig", relation="next", image_path=neighbour_path, snippet="next paragraph"),
    ])
    units = [
        unit("cited-fig", content="Figure without caption or text (raster)", low_text=True),
        unit("next-fig", content="Figure without caption or text (raster)", low_text=True),
    ]
    retriever = FakeRetriever([hit], units=units)
    client = FakeClient(model_json(citations=["cited-fig"]))
    result = ask(retriever, client)
    assert result.images == ["cited-fig"]
    assert "Image for element next-fig" not in user_text(client.kwargs)
    assert len(image_parts(client.kwargs)) == 1


def test_request_uses_the_default_model_without_reasoning():
    retriever = FakeRetriever([evidence()])
    client = FakeClient(model_json())
    result = ask(retriever, client)
    assert DEFAULT_MODEL == "gpt-5.4-mini"
    assert client.kwargs["model"] == "gpt-5.4-mini"
    assert client.kwargs["reasoning_effort"] == "none"
    assert client.kwargs["response_format"] == {"type": "json_object"}
    assert client.kwargs["store"] is False
    assert client.calls == 1
    assert result.model == "gpt-5.4-mini"
    assert retriever.calls == [{"query": "What was the security spend?", "top_k": 10, "documents": None, "mode": "hybrid"}]


def test_model_override_and_search_filters():
    retriever = FakeRetriever([evidence()])
    client = FakeClient(model_json())
    ask(retriever, client, model="gpt-4o-mini", top_k=4, documents=["budget"])
    assert client.kwargs["model"] == "gpt-4o-mini"
    assert retriever.calls[0]["top_k"] == 4
    assert retriever.calls[0]["documents"] == ["budget"]


def test_cli_requires_an_api_key(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    assert main(["What was the security spend?"]) == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_cli_loads_dotenv_without_printing_the_key(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-test-not-a-real-key\n")
    assert main(["What was the security spend?"]) == 2
    captured = capsys.readouterr()
    assert "OPENAI_API_KEY is not set" not in captured.err
    assert "sk-test-not-a-real-key" not in captured.out
    assert "sk-test-not-a-real-key" not in captured.err


def test_non_boolean_sufficient_flag_abstains():
    retriever = FakeRetriever([evidence()])
    client = FakeClient(model_json(sufficient="yes"))
    result = ask(retriever, client)
    assert result.sufficient is False
    assert any("boolean" in warning for warning in result.warnings)
