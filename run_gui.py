"""Open the graphical interface in the browser.

    python run_gui.py

Equivalent to `streamlit run gui/app.py`, without having to remember the path.
"""

import os
import sys
from pathlib import Path

from streamlit.web import cli

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    app = root / "gui" / "app.py"
    # Streamlit looks for .streamlit/config.toml (the theme) in the working directory.
    os.chdir(root)
    sys.argv = ["streamlit", "run", str(app), *sys.argv[1:]]
    raise SystemExit(cli.main())
