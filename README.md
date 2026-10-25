# EventWise AI

EventWise AI answers questions about a local collection of PDF and DOCX documents. It retrieves evidence from those documents and asks a model to answer only from that evidence. Citations are checked before they are shown.

It was built for the Hartree Centre Cardiff Hub AI Pollination Student Competition, Challenge 1: Knowledge Management. This repository contains the application. It does not contain the competition documents, the ground-truth answers, or the generated search index.

## Problem

The source material mixes prose, tables, and figures, sometimes across many files and in more than one language. A useful answer has to point to the page, section, table, or figure it came from, and it has to say when the documents do not contain the answer. EventWise keeps the original files on the machine that runs it, searches an extracted index locally, and sends only the retrieved passages to the model.

## Key features

- PDF and DOCX extraction of headings, text, tables, and figures, with a location kept for each element. Figure crops are saved as PNG files.
- Search units built from that extraction, including row groups for long tables.
- Local embeddings with Sentence Transformers (`all-MiniLM-L6-v2`, 384 dimensions).
- Hybrid retrieval: BM25 and dense search, combined with reciprocal rank fusion.
- Evidence-grounded answers through the OpenAI API (`gpt-5.4-mini`, reasoning effort `none`).
- Citations for a page, section, table, or figure. An element ID is kept only when it was in the prompt. A missing location is dropped rather than guessed.
- A FastAPI backend and a browser page for asking questions.
- Upload of individual PDF and DOCX files, or a folder, into one shared knowledge base, with indexing in a background thread.

## Architecture

```text
PDF / DOCX under data/
        │
        ▼
extraction          headings, text, tables, figure crops
        │
        ▼
indexing            search units and MiniLM embeddings
        │
        ▼
.cache/index        BM25 statistics and dense vectors
        │
        ▼
retrieval           hybrid ranking, top 10 by default
        │
        ▼
answer              one model call, then citation checks
        │
        ▼
browser or CLI      answer plus the citations that survived
```

Folder names under `data/` are only for organisation. Every PDF and DOCX file, including files in nested folders, belongs to the same index.

The web process loads the index and MiniLM once. An upload is saved under `data/`, then a background thread rebuilds the index in `.cache/index-staging`. The live index is replaced only after the new index loads and still contains the documents that were already indexed. The in-memory retriever is swapped after that publish. A failed rebuild leaves the previous index in place.

Indexing and retrieval do not call OpenAI. The model sees the retrieved excerpts and, when a figure needs to be read, at most two images.

```text
eventwise-ai/
├── eventwise/
│   ├── extraction/          # PDF and DOCX parsing
│   ├── indexing/            # Search units, embeddings, on-disk index
│   ├── retrieval/           # BM25 + dense search
│   ├── ingestion/           # Uploads and background index publishing
│   ├── api/                 # FastAPI application
│   ├── answer.py            # Grounded answers and citation checks
│   ├── ask.py               # Command-line questions
│   ├── models.py            # Shared document dataclasses
│   └── config.py            # Index and retrieval settings
├── frontend/                # Browser page served by the API
├── eval/                    # Evaluation scripts; gold files stay local
├── tests/                   # Offline tests
├── analysis/                # Dataset inspection utility
├── requirements.txt
└── pytest.ini
```

## Technology stack

- Python 3.14, the version this project is developed and tested with
- PyMuPDF and python-docx for extraction
- rank-bm25, sentence-transformers, and PyTorch (CPU) for local search
- The OpenAI Python SDK for grounded answering
- python-dotenv for a local `.env` file
- FastAPI, Uvicorn, and python-multipart for the local web application
- A browser page in HTML, CSS, and JavaScript, with no frontend framework
- pytest

There is no account system, database, agent framework, or calculation tool.

## Prerequisites

- macOS or Linux
- Python 3.14
- A network connection for the first install and for the one-time MiniLM download
- An OpenAI API key, only for asking questions
- Enough disk space for the PyTorch install, the embedding model, and the extracted cache

The commands below were checked on macOS. The same shell commands apply on Linux.

## Local installation and configuration

From a terminal:

```bash
git clone https://github.com/archila99/eventwise-ai.git
cd eventwise-ai
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Open `.env` and replace `your_api_key_here` with your OpenAI key. The file contains one setting:

```text
OPENAI_API_KEY=your_api_key_here
```

`.env` is gitignored. Do not commit it. A key already set in the environment is left unchanged. The application reads `.env` from the project root when the process starts, so restart the server after changing the key. Indexing, retrieval, and the offline tests do not need the key.

Create a directory for the original files and build the first index. This extracts new documents, downloads MiniLM into `.cache/models/` if it is not already there, and writes `.cache/index/`.

```bash
mkdir -p data
# Copy your PDF and DOCX files into data/, including nested folders if you want them.
.venv/bin/python -m eventwise.indexing data/
```

The first embedding run is the command that is allowed to download the model. Later runs, and the web application, load MiniLM from `.cache/models/` without downloading it. Run these commands from the project root so `.env` and `.cache/` resolve.

## Running the application

With the index already built:

```bash
.venv/bin/python -m eventwise.api
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/). The server listens on `127.0.0.1` only. Each question uses the same answer pipeline as the command line. The API key stays on the server.

The command-line equivalent is:

```bash
.venv/bin/python -m eventwise.ask "your question"
```

Options:

```bash
.venv/bin/python -m eventwise.ask "your question" -k 10 --doc "some-file.pdf" --no-vision --json --model gpt-5.4-mini
```

- `-k` is how many retrieved items are sent to the model. The default is 10.
- `--doc` limits retrieval to documents whose id, name, or path matches. Repeat it to add another document.
- `--no-vision` does not send figure images.
- `--json` prints the answer, citations, warnings, and image ids.
- `--model` overrides `gpt-5.4-mini`.

The ask command sends document text, and sometimes figure images, to OpenAI. It makes one request and does not retry.

Other local commands:

```bash
.venv/bin/python -m eventwise.extraction data/ --out .cache/extracted
.venv/bin/python -m eventwise.indexing data/ --rebuild
.venv/bin/python -m eventwise.indexing data/ --prune
.venv/bin/python -m eventwise.retrieval "your question" -k 10 --doc "some-file.pdf" --kind table --mode hybrid
```

`--rebuild` rewrites the embeddings. `--prune` drops indexed documents that are not under the paths you pass. `--mode` is `hybrid` (the default), `bm25`, or `dense`. Retrieval does not call OpenAI.

## Uploading documents and asking questions

On the web page, use **Add files** or **Add folder**, then **Upload**. Selected files and their count are shown before the upload starts. Files are stored under `data/`, keeping the relative folder path the browser provides. The page does not show absolute server paths.

The page shows:

- **Processing** while extraction and embedding run. The Ask button is disabled.
- **Ready** when the new index has been checked and the running search is using it. The document list refreshes.
- **Failed** when the update did not finish. If a previous index is still loaded, the message says those documents can still be searched, and Ask stays available. The failed update is not published. Files from that failed upload are removed so they do not block the next one. Documents that were already indexed stay on disk.

**All documents** searches the shared knowledge base. The document list uses the file path, because the same filename can exist in more than one folder. Choosing one row searches that document.

Limits:

- Only `.pdf` and `.docx` files are accepted.
- A file cannot replace one already stored at the same relative path.
- 50 MB per file, and 200 MB per upload.
- One indexing job at a time. A second upload while a job is running is rejected.

To index files that are already in `data/` without using the browser, run `.venv/bin/python -m eventwise.indexing data/` again, then restart the web server so it loads the updated index. The browser upload path reloads the running retriever itself.

## Testing and evaluation

Offline tests, from the project root:

```bash
.venv/bin/python -m pytest -q
```

These tests use temporary directories and a fake model client. They do not call OpenAI, and they do not read `data/` or `.cache/`. They cover extraction, indexing, retrieval, prompt construction, citation checks, abstention, image selection, the API, and uploads. A local run on 25 October 2026 passed 193 tests. That count is a unit and integration result. It is not a measure of answer accuracy.

Two separate local benchmarks were run against the competition corpus. The corpus, the ground-truth files (`eval/qa_gold.json` and `eval/retrieval_gold.json`), and the reports under `eval/results/` are gitignored because the reports quote document text. A fresh clone does not contain them, so these numbers cannot be reproduced from the repository alone.

Question answering, 25 October 2026, report `eval/results/qa_eval_20261025T115617Z.json`:

- 20 questions, one request each, `gpt-5.4-mini`, reasoning effort `none`, hybrid retrieval, top 10, no automatic retries.
- 18 strict passes and 2 reviews. Reviews are not counted as correct. Strict accuracy was 90%.
- Mean citation score 0.917.
- Passes by category: factual text 4/4, table 4/4, chart 4/4, calculation 3/3, unanswerable 2/2, cross-document 1/3. The two cross-document reviews are the items that were not passed.
- An earlier run the same day, before later answer-pipeline changes, scored 13/20 (65%) with a mean citation score of 0.889.

Retrieval only, 22 October 2026, with no model call. Hybrid Recall@10 was 100% on the 8 priority questions, 97% on the 16 development questions, and 100% on the 10 held-out questions. Development questions were used while tuning the ranking. One setting was changed after the first held-out run, so that set is no longer strictly unseen.

To rerun retrieval scoring on a machine that already has the private gold file and the index:

```bash
.venv/bin/python eval/retrieval_eval.py
```

`eval/run_qa_eval.py` calls the paid API. Do not run it unless you intend to make those requests. `--dry-run` scores retrieval and prompt construction without calling the API, and it still needs `eval/qa_gold.json`.

## Known limitations

- This is a local competition MVP. It is not a production service. Answers can be wrong, incomplete, or abstain when the evidence is present.
- There is no authentication. Anyone who can open the local page can upload files and ask questions. Uploaded documents are not isolated between users.
- The web application does not download MiniLM. Build the index once with `python -m eventwise.indexing` so the model is cached before starting the server or uploading.
- Extraction does not OCR scanned pages. A value that exists only inside a figure is available when that figure is one of the images attached to the question.
- At most two images are sent. Tables without ruling lines can be missed. There is no arithmetic tool; the model sees a table as text.
- DOCX citations use paragraph, block, table, row, and column locations. Those files have no page numbers.
- Welsh text is retrieved mainly through BM25. The embedding model is multilingual only to the extent MiniLM supports it.
- The answer is one model call. There is no chat history and no checker for contradictions across documents.
- One ingestion job runs at a time. A failed job can leave unused files in `.cache/extracted`. The live index is kept.
- A `.docx` file is recognised by its suffix and a ZIP signature. A renamed ZIP can pass that check and then fail in the parser.

## Future improvements

- Let the first web upload download MiniLM when `.cache/models/` is empty.
- Allow a stored file to be replaced, and queue more than one indexing job.
- Add OCR for scanned pages and stronger detection for tables without ruling lines.
- Improve retrieval for Welsh and other non-English text.
- Add authentication and separate document collections if the application is used by more than one person.
- Keep a short conversation history without weakening the citation checks.
