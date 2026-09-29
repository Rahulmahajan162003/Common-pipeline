"""
Entrypoint for `python -m benchmark_harness`.
"""

import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
