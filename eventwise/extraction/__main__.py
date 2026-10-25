"""Command line: python -m eventwise.extraction PATH [PATH ...] [--out DIR] [--pages SPEC]."""

import sys

from .extract import main

sys.exit(main())
