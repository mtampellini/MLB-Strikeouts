"""Phase 5: picks.json + tier file writers."""
from __future__ import annotations

import json
from pathlib import Path

from .engine import PickResult


def write_picks(result: PickResult, output_dir: Path | str) -> dict[str, Path]:
    """Write the four canonical files into ``output_dir``:

    - ``picks.json``: primary tier (the live slate)
    - ``secondary_picks.json``: secondary tier (above price cap or below rank 10)
    - ``shadow_picks.json``: shadow tier (10-20% edge)
    - ``all_picks_debug.json``: everything (primary + secondary + shadow + skipped + metadata)

    Returns a dict mapping file label -> Path of written file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    paths["primary"] = out_dir / "picks.json"
    paths["secondary"] = out_dir / "secondary_picks.json"
    paths["shadow"] = out_dir / "shadow_picks.json"
    paths["debug"] = out_dir / "all_picks_debug.json"

    paths["primary"].write_text(
        json.dumps({"metadata": result.metadata, "picks": result.primary},
                    indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths["secondary"].write_text(
        json.dumps({"metadata": result.metadata, "picks": result.secondary},
                    indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths["shadow"].write_text(
        json.dumps({"metadata": result.metadata, "picks": result.shadow},
                    indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths["debug"].write_text(
        json.dumps({
            "metadata": result.metadata,
            "primary": result.primary,
            "secondary": result.secondary,
            "shadow": result.shadow,
            "skipped": result.skipped,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return paths
