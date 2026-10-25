# EventWise AI

EventWise AI answers questions about a local collection of PDF and DOCX documents. Every answer is tied to retrieved evidence, and citations are checked before they are shown. It is an MVP for the Hartree Centre Cardiff Hub AI Pollination Student Competition, Technical Challenge 1 (Knowledge Management).

The competition documents stay on your machine. This repository contains the application, not the source files.

## Problem

The documents mix prose, tables, and figures. A useful answer has to say where a fact came from — a page, section, table, or figure — and it has to say when the documents do not contain the answer. EventWise searches the extracted documents locally, then asks a model to write the answer only from the passages it was given.

## Architecture

1. **Extraction.** PDF and DOCX files are split into headings, text, tables, and figures. Each element keeps its location. Figures are saved as PNG crops.
2. **Local hybrid retrieval.** The elements are indexed with BM25 and with local MiniLM embeddings (`sentence-transformers/all-MiniLM-L6-v2`). A query searches both indexes and fuses the rankings with reciprocal rank fusion.
3. **Answering.** `python -m eventwise.ask` sends the retrieved evidence to OpenAI (`gpt-5.4-mini` by default, reasoning effort `none`) and asks for a JSON answer with element IDs.
4. **Citation validation.** An element ID is kept only when it was in the prompt. Result citations use the string built at retrieval time. Context citations are rebuilt from the indexed source metadata. A missing location is dropped rather than guessed.

Retrieval and indexing do not call OpenAI. The model sees the retrieved excerpts and, when a figure needs to be read, at most two images.

## Stack

- Python 3.14 (the version this project is developed with), on macOS
- PyMuPDF and python-docx for extraction
- rank-bm25, sentence-transformers, and PyTorch (CPU) for local search
- The OpenAI Python SDK for grounded answering
- python-dotenv for a local `.env` file
- pytest

There is no web UI, agent framework, or calculation tool.

## Install on macOS

From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The first install downloads PyTorch and the other pinned packages. It does not need an API key.

## Configure the API key

Create a file named `.env` in the project root, next to `requirements.txt`:

```bash
cp .env.example .env
```

Open `.env` in a local editor and replace `your_api_key_here` with your OpenAI key. The file should contain one line:

```text
OPENAI_API_KEY=your_api_key_here
```

`.env` is listed in `.gitignore`. Do not commit it, paste the key into a chat, or put it in the README. `python -m eventwise.ask` loads this file from the current directory. A key that is already set in the environment is left unchanged. The program does not print the key.

Commands other than `eventwise.ask` do not read the key.

## Documents and the local index

Put the PDF and DOCX files in a local `data/` directory. That directory is gitignored.

Build or update the index from the project root. This extracts any document that is not already extracted, then embeds new or changed documents. The first run can download MiniLM into `.cache/models/` if it is not already there.

```bash
.venv/bin/python -m eventwise.index data/
```

Useful variants:

```bash
.venv/bin/python -m eventwise.extract data/ --out .cache/extracted
.venv/bin/python -m eventwise.index data/ --rebuild
.venv/bin/python -m eventwise.index data/ --prune
```

`--rebuild` rewrites the embeddings. `--prune` drops indexed documents that are not under the paths you pass.

Inspect retrieval without calling the API:

```bash
.venv/bin/python -m eventwise.retrieve "your question" -k 10 --doc "some-file.pdf" --kind table --mode hybrid
```

`--doc` and `--kind` can be repeated. `--mode` is `hybrid` (the default), `bm25`, or `dense`. `--json` prints the structured result.

The index, extracted JSON, figure PNGs, and the embedding model live under `.cache/`. That directory is disposable and gitignored. Run these commands from the project root.

## Ask a question

```bash
.venv/bin/python -m eventwise.ask "your question"
```

Options that exist today:

```bash
.venv/bin/python -m eventwise.ask "your question" -k 10 --doc "some-file.pdf" --no-vision --json --model gpt-5.4-mini
```

- `-k` is how many retrieved items are sent to the model (default 10).
- `--doc` limits retrieval to documents whose id, name, or path matches. Repeat it to add names.
- `--no-vision` does not send figure images.
- `--json` prints the answer, citations, warnings, and image ids as JSON.
- `--model` overrides `gpt-5.4-mini`.

The command sends document text, and sometimes figure images, to OpenAI. It makes one request and does not retry. A missing key or a missing index exits with an error and does not call the API.

## Tests

```bash
.venv/bin/python -m pytest -q
```

The suite covers extraction, indexing, retrieval, prompt building, citation checks, abstention, and image selection. Answering tests use a fake client. They do not call OpenAI.

Retrieval quality on the local corpus can be scored without a model:

```bash
.venv/bin/python eval/retrieval_eval.py
```

That script reads `eval/retrieval_gold.json` and writes `eval/results/`. The results directory is gitignored because a saved retrieval log quotes document text.

## Limitations

- Chart and figure answers have not been checked against the live API. The vision request is implemented and covered with mocked tests only.
- Extraction does not OCR images. A value that exists only inside a figure is available to the model when that figure is one of the images attached to the question.
- At most two images are sent. An image is attached when the question refers to a figure or chart, or when a retrieved figure has little or no text of its own. Native chart data that is already stored as text is not sent again as an image.
- Abstention is the model's reading of the evidence. Retrieval scores are not used as a cutoff.
- Tables without ruling lines can be missed. There is no arithmetic tool; the model sees the table as text.
- DOCX citations use paragraph, block, table, row, and column locations. Those files have no page numbers.
- Welsh text is retrieved mainly through BM25.
- The answer is one model call. There is no chat history, web interface, or cross-document inconsistency checker.
