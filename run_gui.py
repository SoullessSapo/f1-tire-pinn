"""Open the graphical interface in the browser.

    python run_gui.py

Equivalent to `streamlit run gui/app.py`, without having to remember the path.
"""

import sys
from pathlib import Path

from streamlit.web import cli

if __name__ == "__main__":
    app = Path(__file__).resolve().parent / "gui" / "app.py"
    sys.argv = ["streamlit", "run", str(app), *sys.argv[1:]]
    raise SystemExit(cli.main())
