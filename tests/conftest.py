import logging
import os
import sys
from pathlib import Path

# Make sure the installed/editable package is found even if someone runs
# ``pytest`` without activating the venv.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

logging.basicConfig(level=logging.WARNING)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
