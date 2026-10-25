"""Command line: python -m eventwise.indexing PATH [PATH ...] [--prune] [--rebuild]."""

import sys

from .index import main

sys.exit(main())
