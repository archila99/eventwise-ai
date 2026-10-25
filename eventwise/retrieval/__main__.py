"""Command line: python -m eventwise.retrieval "question" [-k 10] [--doc NAME] [--kind KIND]."""

import sys

from .retrieve import main

sys.exit(main())
