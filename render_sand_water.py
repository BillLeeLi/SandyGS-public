#!/usr/bin/env python3
"""Primary command-line entry for SandyGS sand-water rendering."""

import runpy


def main() -> None:
    runpy.run_module("sandygs.render_app", run_name="__main__")


if __name__ == "__main__":
    main()
