# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Graphviz-based topology diagram renderer.

Produces hierarchical block diagrams from a ``system_topology`` dictionary.
Components are grouped by NUMA node, connections are color-coded by type
(NVLink, PCIe, network), and bandwidth labels are shown when available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    import graphviz


def render_topology(
    topology: dict[str, Any],
    output: str | Path,
    *,
    fmt: str = "png",
    engine: str = "dot",
) -> Path:
    """
    Render a topology dictionary as a Graphviz block diagram.

    Parameters
    ----------
    topology
        Enriched topology dictionary (as produced by the C++ ``topology_viz``
        tool or loaded from JSON).  Supports partial data -- missing bandwidth
        or name fields result in labels without those details.
    output
        Destination file path.  The Graphviz ``render()`` call appends the
        format extension automatically; this function removes a redundant
        extension if present (e.g., ``"out.png"`` will not become
        ``"out.png.png"``).
    fmt
        Output format passed to Graphviz: ``"png"`` (default), ``"svg"``,
        or ``"pdf"``.
    engine
        Graphviz layout engine.  ``"dot"`` (default) produces top-to-bottom
        hierarchical layouts best suited for topology diagrams.

    Returns
    -------
    Path
        Absolute path to the rendered file.

    Raises
    ------
    ValueError
        If *fmt* is not one of ``{"png", "svg", "pdf"}``.
    RuntimeError
        If the Graphviz ``dot`` executable is not found.

    Notes
    -----
    Visual conventions:

    - **NUMA nodes** are drawn as dashed-border subgraph clusters.
    - **GPUs** are green rectangles (#76B900) with white text.
    - **CPUs** are blue rectangles (#0076CE) with white text.
    - **NICs** are orange rectangles (#FF6F00) with white text.
    - **NVLink** edges are thick green lines.
    - **PCIe** edges are dashed blue lines.
    - **Network speed** is annotated on an edge to an abstract *Network*
      node, or directly on the NIC label when speed is known.

    When a bandwidth or name field is zero / empty the corresponding label
    element is simply omitted (graceful degradation for partial JSON input).
    """


def build_graph(topology: dict[str, Any], *, engine: str = "dot") -> graphviz.Digraph:
    """
    Build a Graphviz ``Digraph`` from a topology dictionary.

    Returns the graph object without writing any file, which is useful for
    interactive exploration (e.g., Jupyter notebooks where ``Digraph``
    objects render inline via ``_repr_svg_()``).

    Parameters
    ----------
    topology
        Enriched topology dictionary (as produced by the C++ ``topology_viz``
        tool or loaded from JSON).  Supports partial data.
    engine
        Graphviz layout engine name.  ``"dot"`` (default) produces
        top-to-bottom hierarchical layouts.

    Returns
    -------
    graphviz.Digraph
        A fully constructed graph.  Can be displayed directly in a Jupyter
        notebook, rendered to a file with ``.render()``, or inspected via
        ``.source``.

    Examples
    --------
    In a Jupyter notebook:

    >>> import rapidsmpf_topo_viz as tv
    >>> topo = tv.discover()
    >>> graph = tv.build_graph(topo)
    >>> graph  # renders inline as SVG
    """
