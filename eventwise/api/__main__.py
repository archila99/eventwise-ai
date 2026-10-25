"""Start the local web application.

    python -m eventwise.api

Run this from the project root so ``.env`` and ``.cache/`` resolve.
"""

import uvicorn


def main() -> None:
    uvicorn.run("eventwise.api.app:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
