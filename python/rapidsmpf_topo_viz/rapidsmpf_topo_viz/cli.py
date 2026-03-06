# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Command-line interface for rapidsmpf-topo-viz.

Usage
-----
Discover and print enriched topology JSON::

    rapidsmpf-topo-viz discover
    rapidsmpf-topo-viz discover --output topology.json

Render a diagram from live discovery::

    rapidsmpf-topo-viz render --output topology.png

Render a diagram from a saved JSON file::

    rapidsmpf-topo-viz render --json topology.json --output topology.svg --format svg
"""

from __future__ import annotations

import sys

import click

SUPPORTED_FORMATS = ("png", "svg", "pdf")


@click.group()
def main() -> None:
    """Discover and visualize NVIDIA multi-GPU system topology."""


@main.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Write JSON to a file instead of stdout.",
)
def discover(output: str | None) -> None:
    """Discover system topology and output enriched JSON."""
    from rapidsmpf_topo_viz.topology import TopologyViz

    viz = TopologyViz()
    if not viz.discover():
        click.echo("Error: topology discovery failed", err=True)
        sys.exit(1)

    json_str = viz.to_json(indent=2)

    if output is not None:
        with open(output, "w") as f:  # noqa: PTH123
            f.write(json_str)
        click.echo(f"Topology written to {output}")
    else:
        click.echo(json_str)


@main.command()
@click.option(
    "--json",
    "json_path",
    type=click.Path(exists=True),
    default=None,
    help="Load topology from a JSON file instead of live discovery.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    required=True,
    help="Destination path for the rendered diagram.",
)
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(SUPPORTED_FORMATS, case_sensitive=False),
    default="png",
    show_default=True,
    help="Output image format.",
)
def render(
    json_path: str | None,
    output: str,
    fmt: str,
) -> None:
    """
    Render a topology block diagram.

    If --json is not provided, runs live discovery first.
    """
    import rapidsmpf_topo_viz as tv

    if json_path is not None:
        topo = tv.load_json(json_path)
    else:
        topo = tv.discover()

    result = tv.render(topo, output, fmt=fmt)
    click.echo(f"Diagram written to {result}")
