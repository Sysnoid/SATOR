"""Dev-only helper: import a demo module and redirect plt.show() to savefig.

Run: python examples/_render_smoke.py demo_09_pharma_tablet_pca
Produces examples/responses/<name>_fig_<n>.png for each figure drawn.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python examples/_render_smoke.py <demo_module_name>")
        sys.exit(1)
    mod_name = sys.argv[1]
    os.environ.setdefault("MPLBACKEND", "Agg")

    sys.path.insert(0, str(Path(__file__).parent))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path("examples/responses")
    out_dir.mkdir(parents=True, exist_ok=True)
    counter = {"i": 0}

    original_show = plt.show

    def _save_show(*_args, **_kwargs) -> None:
        counter["i"] += 1
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            fname = out_dir / f"{mod_name}_fig_{counter['i']}_{fig_num}.png"
            fig.savefig(fname, bbox_inches="tight")
            print(f"[render] saved {fname}")
        plt.close("all")

    plt.show = _save_show  # type: ignore[assignment]

    import importlib

    demo = importlib.import_module(mod_name)
    if hasattr(demo, "main"):
        demo.main()

    plt.show = original_show  # type: ignore[assignment]


if __name__ == "__main__":
    main()
