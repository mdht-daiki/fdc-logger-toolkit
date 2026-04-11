from __future__ import annotations

import sys
from pathlib import Path

from importlinter.cli import lint_imports_command


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    sys.path.insert(0, str(src_dir))

    args = ["--config", str(repo_root / "pyproject.toml")]
    lint_imports_command.main(args=args, standalone_mode=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
