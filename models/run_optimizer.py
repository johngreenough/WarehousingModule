#!/usr/bin/env python3
import sys
from pathlib import Path

# Allow running from models/ while keeping package imports stable.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.optimizer import main  # noqa: E402


if __name__ == "__main__":
    main()
