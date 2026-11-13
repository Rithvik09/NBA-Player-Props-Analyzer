"""Build the expanded points matrices for a season list that includes the live season.

build_expanded_points.SEASONS stops at 2025-26, so a crawled live season would be
left out of the build while it reported success. That module cannot simply be
edited: expanded_points_v4 and expanded_points_v6 pin its content, and v4 is closed,
so a drifted pin there has no honest accounting -- a closure is written once.

Its run() reads SEASONS from module scope each time it is called, so this sets that
list for the duration of one call and restores it afterwards. The builder stays
byte-identical, both studies keep verifiable provenance, and the new build's manifest
still records the seasons it actually used, because run() reads the same name.
"""
import argparse
from contextlib import contextmanager
from pathlib import Path
from scripts import build_expanded_points as builder


@contextmanager
def seasons(requested):
    """Point the builder at `requested` for one call, then put its default back."""
    requested = list(requested)
    if not requested:
        raise ValueError('No seasons requested; the builder would produce nothing')
    duplicated = sorted({s for s in requested if requested.count(s) > 1})
    if duplicated:
        raise ValueError('Season listed twice: '+', '.join(duplicated))
    original = builder.SEASONS
    builder.SEASONS = requested
    try:
        yield requested
    finally:
        builder.SEASONS = original


def run(output, workers=2, requested=None):
    """Build `requested` seasons, defaulting to the builder's own list plus nothing."""
    with seasons(requested or builder.SEASONS) as used:
        builder.run(Path(output), workers)
        return used


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--seasons', nargs='+', default=None,
                   help=f'seasons to build; default {builder.SEASONS}')
    a = p.parse_args()
    print(run(a.output, a.workers, a.seasons))
