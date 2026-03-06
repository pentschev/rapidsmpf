# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Topology visualization for NVIDIA multi-GPU systems.

This package provides a Python API for discovering system topology
(GPUs, CPUs, NICs, NVLink, PCIe) and rendering it as a block diagram.

Discovery and enrichment are performed by the C++ ``topology_viz`` library
via Cython bindings; rendering uses Graphviz.

Examples
--------
Discover the live system topology and render a diagram:

>>> import rapidsmpf_topo_viz as tv
>>> topo = tv.discover()
>>> tv.render(topo, "topology.png")

Render from a previously-saved JSON file:

>>> topo = tv.load_json("topology.json")
>>> tv.render(topo, "topology.svg", fmt="svg")

Build a Graphviz graph object for interactive use (e.g., Jupyter):

>>> graph = tv.build_graph(topo)
>>> graph  # renders inline as SVG in Jupyter
"""

from __future__ import annotations

import json as _json
from typing import TYPE_CHECKING, Any

from rapidsmpf_topo_viz.renderer import (
    build_graph as build_graph,
    render_topology as render_topology,
)
from rapidsmpf_topo_viz.topology import TopologyViz as TopologyViz

if TYPE_CHECKING:
    from pathlib import Path


def discover(*, enrich: bool = True) -> dict[str, Any]:
    """
    Discover the enriched system topology.

    Calls into the C++ ``topology_viz`` library directly (via Cython) to
    run cuCascade topology discovery and, optionally, bandwidth / naming
    enrichment.

    Parameters
    ----------
    enrich
        If ``True`` (default), perform full enrichment (PCIe bandwidth,
        NVLink connections, CPU/NIC names).  If ``False``, only the base
        cuCascade discovery is run and enrichment fields are left at
        their defaults.

    Returns
    -------
    dict
        The topology as a nested dictionary matching the JSON schema
        of ``system_topology``.

    Raises
    ------
    RuntimeError
        If cuCascade topology discovery fails.
    """
    viz = TopologyViz()
    if not viz.discover():
        msg = "Topology discovery failed"
        raise RuntimeError(msg)
    if not enrich:
        return _json.loads(viz.to_json(indent=0))
    return viz.to_dict()


def load_json(path: str | Path) -> dict[str, Any]:
    """
    Load a topology from a JSON file.

    Accepts both the enriched format produced by ``discover()`` and the
    original cuCascade ``topology_discovery`` format.  Missing enrichment
    fields are left at their defaults (zero / empty).

    Parameters
    ----------
    path
        Filesystem path to a JSON file.

    Returns
    -------
    dict
        The topology as a nested dictionary.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    RuntimeError
        If the file is not valid JSON or required keys are missing.
    """
    from pathlib import Path as _Path

    p = _Path(path)
    if not p.exists():
        msg = f"File not found: {p}"
        raise FileNotFoundError(msg)

    viz = TopologyViz()
    if not viz.load_json_file(str(p)):
        msg = f"Failed to load JSON file: {p}"
        raise RuntimeError(msg)
    return viz.to_dict()


def render(
    topology: dict[str, Any],
    output: str | Path,
    *,
    fmt: str = "png",
) -> Path:
    """
    Render a topology dictionary as a block diagram.

    Parameters
    ----------
    topology
        A topology dictionary (from ``discover()`` or ``load_json()``).
    output
        Destination file path for the rendered image.  The file
        extension is *not* used to infer the format -- use ``fmt``.
    fmt
        Output format: ``"png"`` (default), ``"svg"``, or ``"pdf"``.

    Returns
    -------
    Path
        The absolute path to the written file.

    Raises
    ------
    ValueError
        If ``fmt`` is not one of the supported formats.
    RuntimeError
        If Graphviz rendering fails (e.g., ``dot`` is not installed).
    """
    return render_topology(topology, output, fmt=fmt)


def discover_and_render(
    output: str | Path,
    *,
    fmt: str = "png",
    enrich: bool = True,
) -> Path:
    """
    Convenience: discover live topology and render a diagram.

    Equivalent to ``render(discover(enrich=enrich), output, fmt=fmt)``.

    Parameters
    ----------
    output
        Destination file path for the rendered image.
    fmt
        Output format: ``"png"``, ``"svg"``, or ``"pdf"``.
    enrich
        Whether to perform full bandwidth/naming enrichment.

    Returns
    -------
    Path
        The absolute path to the written file.
    """
    topo = discover(enrich=enrich)
    return render(topo, output, fmt=fmt)
