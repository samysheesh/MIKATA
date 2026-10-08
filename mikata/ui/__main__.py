"""`python -m mikata.ui` : interface de MIKATA (mascotte, réglages), processus à part."""

from __future__ import annotations

import sys

from mikata.ui.qt.app import main

if __name__ == "__main__":
    sys.exit(main())
