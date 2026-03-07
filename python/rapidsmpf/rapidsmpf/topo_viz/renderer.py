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

GPU_COLOR = "#76B900"
CPU_COLOR = "#0076CE"
NIC_COLOR = "#FF6F00"
SWITCH_COLOR = "#9B59B6"
NVLINK_COLOR = "#76B900"
PCIE_COLOR = "#4A90D9"
NETWORK_COLOR = "#888888"
CLUSTER_BG = "#F5F5F5"
CLUSTER_BORDER = "#CCCCCC"
FONT_COLOR_LIGHT = "white"
FONT_NAME = "Helvetica"


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
    - **PCIe Switches** are purple rectangles (#9B59B6) with white text.
    - **NVLink** edges are thick green lines.
    - **PCIe** edges are dashed blue lines (endpoint to switch, switch to CPU).
    - **Network speed** is annotated on an edge to an abstract *Network*
      node.

    When a bandwidth or name field is zero / empty the corresponding label
    element is simply omitted (graceful degradation for partial JSON input).
    """
    from pathlib import Path as _Path

    supported = {"png", "svg", "pdf"}
    if fmt not in supported:
        msg = f"Unsupported format {fmt!r}, must be one of {supported}"
        raise ValueError(msg)

    graph = build_graph(topology, engine=engine)

    out = _Path(output)
    stem = str(out)
    if stem.endswith(f".{fmt}"):
        stem = stem[: -(len(fmt) + 1)]

    rendered = graph.render(filename=stem, format=fmt, cleanup=True)
    return _Path(rendered).resolve()


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
    import graphviz as gv

    hostname = topology.get("system", {}).get("hostname", "System")
    graph = gv.Digraph(
        name=f"topology_{hostname}",
        engine=engine,
        graph_attr={
            "rankdir": "TB",
            "label": f"System Topology: {hostname}",
            "labelloc": "t",
            "fontname": FONT_NAME,
            "fontsize": "16",
            "compound": "true",
            "nodesep": "0.6",
            "ranksep": "0.8",
        },
        node_attr={"fontname": FONT_NAME, "fontsize": "10"},
        edge_attr={"fontname": FONT_NAME, "fontsize": "9"},
    )

    gpus = topology.get("gpus", [])
    nics = topology.get("network_devices", [])
    cpus = topology.get("cpus", [])
    switches = topology.get("pcie_switches", [])

    gpus.sort(key=lambda g: (g.get("numa_node", -1), g.get("id", 0)))
    nics.sort(key=lambda n: (n.get("numa_node", -1), n.get("name", "")))
    switches.sort(key=lambda s: (s.get("numa_node", -1), s.get("pci_bus_id", "")))

    # Build lookup: which switch does each GPU / NIC belong to?
    gpu_to_switch: dict[int, str] = {}
    nic_to_switch: dict[str, str] = {}
    for sw in switches:
        sw_id = sw["pci_bus_id"]
        for gid in sw.get("gpu_ids", []):
            gpu_to_switch[gid] = sw_id
        for nn in sw.get("nic_names", []):
            nic_to_switch[nn] = sw_id

    numa_gpus: dict[int, list[dict]] = {}
    for gpu in gpus:
        node = gpu.get("numa_node", -1)
        numa_gpus.setdefault(node, []).append(gpu)

    numa_nics: dict[int, list[dict]] = {}
    for nic in nics:
        node = nic.get("numa_node", -1)
        numa_nics.setdefault(node, []).append(nic)

    numa_switches: dict[int, list[dict]] = {}
    for sw in switches:
        node = sw.get("numa_node", -1)
        numa_switches.setdefault(node, []).append(sw)

    cpu_by_numa: dict[int, dict] = {}
    for cpu in cpus:
        cpu_by_numa[cpu.get("numa_node", -1)] = cpu

    all_numa = sorted(
        set(
            list(numa_gpus.keys()) + list(numa_nics.keys()) + list(numa_switches.keys())
        )
    )

    for numa_node in all_numa:
        cluster_name = f"cluster_numa{numa_node}"
        numa_label = f"NUMA Node {numa_node}" if numa_node >= 0 else "NUMA Unknown"

        with graph.subgraph(name=cluster_name) as sub:
            sub.attr(
                label=numa_label,
                style="dashed,rounded",
                color=CLUSTER_BORDER,
                bgcolor=CLUSTER_BG,
                fontname=FONT_NAME,
                fontsize="12",
            )

            # Tier 1: GPUs (top)
            local_gpu_ids: list[str] = []
            for gpu in numa_gpus.get(numa_node, []):
                gpu_id = f"gpu_{gpu['id']}"
                sub.node(
                    gpu_id,
                    label=_gpu_label(gpu),
                    shape="box",
                    style="filled,rounded",
                    fillcolor=GPU_COLOR,
                    fontcolor=FONT_COLOR_LIGHT,
                )
                local_gpu_ids.append(gpu_id)

            # Tier 2: CPU
            cpu_id = None
            cpu_info = cpu_by_numa.get(numa_node)
            if cpu_info is not None:
                cpu_label = _cpu_label(cpu_info)
                cpu_id = f"cpu_numa{numa_node}"
                sub.node(
                    cpu_id,
                    label=cpu_label,
                    shape="box",
                    style="filled,rounded",
                    fillcolor=CPU_COLOR,
                    fontcolor=FONT_COLOR_LIGHT,
                )

            # Tier 3: PCIe Switches
            local_sw_ids: list[str] = []
            for sw in numa_switches.get(numa_node, []):
                sw_id = f"sw_{_sanitize_bdf(sw['pci_bus_id'])}"
                sub.node(
                    sw_id,
                    label=_switch_label(sw),
                    shape="box",
                    style="filled,rounded",
                    fillcolor=SWITCH_COLOR,
                    fontcolor=FONT_COLOR_LIGHT,
                )
                local_sw_ids.append(sw_id)

            # Tier 4: NICs (bottom)
            local_nic_ids: list[str] = []
            for nic in numa_nics.get(numa_node, []):
                nic_id = f"nic_{nic['name']}"
                sub.node(
                    nic_id,
                    label=_nic_label(nic),
                    shape="box",
                    style="filled,rounded",
                    fillcolor=NIC_COLOR,
                    fontcolor=FONT_COLOR_LIGHT,
                )
                local_nic_ids.append(nic_id)

            # Invisible edges to enforce tier ordering within cluster.
            # Use middle elements as anchors to horizontally center
            # dependent tiers.
            mid_gpu = local_gpu_ids[len(local_gpu_ids) // 2] if local_gpu_ids else None
            mid_sw = local_sw_ids[len(local_sw_ids) // 2] if local_sw_ids else None
            # GPUs -> CPU -> Switches -> NICs
            if mid_gpu and cpu_id:
                sub.edge(mid_gpu, cpu_id, style="invis")
            for sw_id in local_sw_ids:
                if cpu_id:
                    sub.edge(cpu_id, sw_id, style="invis")
                elif mid_gpu:
                    sub.edge(mid_gpu, sw_id, style="invis")
            for nic_id in local_nic_ids:
                if mid_sw:
                    sub.edge(mid_sw, nic_id, style="invis")
                elif cpu_id:
                    sub.edge(cpu_id, nic_id, style="invis")
                elif mid_gpu:
                    sub.edge(mid_gpu, nic_id, style="invis")

    # PCIe edges: GPU -> Switch (or GPU -> CPU if no switch)
    for gpu in gpus:
        gpu_id_num = gpu.get("id", 0)
        gpu_node = f"gpu_{gpu_id_num}"
        sw_bdf = gpu_to_switch.get(gpu_id_num)
        if sw_bdf:
            label = _pcie_label(gpu.get("pcie", {}))
            graph.edge(
                gpu_node,
                f"sw_{_sanitize_bdf(sw_bdf)}",
                label=label,
                style="dashed",
                color=PCIE_COLOR,
                fontcolor=PCIE_COLOR,
                constraint="false",
            )
        else:
            numa_node = gpu.get("numa_node", -1)
            if numa_node in cpu_by_numa:
                label = _pcie_label(gpu.get("pcie", {}))
                graph.edge(
                    gpu_node,
                    f"cpu_numa{numa_node}",
                    label=label,
                    style="dashed",
                    color=PCIE_COLOR,
                    fontcolor=PCIE_COLOR,
                )

    # PCIe edges: NIC -> Switch (or NIC -> CPU if no switch)
    for nic in nics:
        nic_name = nic.get("name", "")
        nic_node = f"nic_{nic_name}"
        sw_bdf = nic_to_switch.get(nic_name)
        if sw_bdf:
            label = _pcie_label(nic.get("pcie", {}))
            graph.edge(
                nic_node,
                f"sw_{_sanitize_bdf(sw_bdf)}",
                label=label,
                style="dashed",
                color=PCIE_COLOR,
                fontcolor=PCIE_COLOR,
                constraint="false",
            )
        else:
            numa_node = nic.get("numa_node", -1)
            if numa_node in cpu_by_numa:
                label = _pcie_label(nic.get("pcie", {}))
                graph.edge(
                    nic_node,
                    f"cpu_numa{numa_node}",
                    label=label,
                    style="dashed",
                    color=PCIE_COLOR,
                    fontcolor=PCIE_COLOR,
                    constraint="false",
                )

    # PCIe edges: Switch -> CPU (upstream link)
    for sw in switches:
        sw_node = f"sw_{_sanitize_bdf(sw['pci_bus_id'])}"
        numa_node = sw.get("numa_node", -1)
        if numa_node in cpu_by_numa:
            label = _pcie_label(sw.get("pcie", {}))
            graph.edge(
                sw_node,
                f"cpu_numa{numa_node}",
                label=label,
                style="dashed",
                color=PCIE_COLOR,
                fontcolor=PCIE_COLOR,
                constraint="false",
            )

    # NVLink edges (deduplicated: only draw A->B where A < B)
    drawn_nvlinks: set[tuple[int, int]] = set()
    for gpu in gpus:
        src_id = gpu.get("id", 0)
        for peer in gpu.get("nvlink_peers", []):
            dst_id = peer.get("peer_gpu_id", 0)
            edge_key = (min(src_id, dst_id), max(src_id, dst_id))
            if edge_key in drawn_nvlinks:
                continue
            drawn_nvlinks.add(edge_key)

            label = _nvlink_label(peer)
            graph.edge(
                f"gpu_{src_id}",
                f"gpu_{dst_id}",
                label=label,
                color=NVLINK_COLOR,
                penwidth="2.5",
                fontcolor=NVLINK_COLOR,
                dir="both",
                constraint="false",
            )

    # Network node for NICs with speed
    nics_with_speed = [n for n in nics if n.get("bandwidth_gbps", 0) > 0]
    if nics_with_speed:
        graph.node(
            "network",
            label="Network",
            shape="ellipse",
            style="filled",
            fillcolor=NETWORK_COLOR,
            fontcolor=FONT_COLOR_LIGHT,
        )
        # Single invisible constraining edge to place Network below NICs
        graph.edge(
            f"nic_{nics_with_speed[0]['name']}",
            "network",
            style="invis",
        )
        for nic in nics_with_speed:
            bw = nic["bandwidth_gbps"]
            speed_bits = bw * 8
            label = f"{speed_bits:.0f} Gb/s"
            graph.edge(
                f"nic_{nic['name']}",
                "network",
                label=label,
                color=NETWORK_COLOR,
                fontcolor=NETWORK_COLOR,
                style="bold",
                constraint="false",
            )

    return graph


def _sanitize_bdf(bdf: str) -> str:
    """Replace characters that Graphviz interprets as port separators."""
    return bdf.replace(":", "_").replace(".", "_")


def _gpu_label(gpu: dict[str, Any]) -> str:
    name = gpu.get("name", "")
    gpu_id = gpu.get("id", "?")
    if name:
        return f"GPU {gpu_id}\\n{name}"
    return f"GPU {gpu_id}"


def _cpu_label(cpu: dict[str, Any]) -> str:
    name = cpu.get("model_name", "")
    numa = cpu.get("numa_node", "?")
    cores = cpu.get("core_count", 0)
    parts = [f"CPU (NUMA {numa})"]
    if name:
        parts.append(name)
    if cores > 0:
        parts.append(f"{cores} cores")
    return "\\n".join(parts)


def _nic_label(nic: dict[str, Any]) -> str:
    name = nic.get("name", "?")
    model = nic.get("model_name", "")
    if model:
        return f"{name}\\n{model}"
    return str(name)


def _switch_label(sw: dict[str, Any]) -> str:
    bdf = sw.get("pci_bus_id", "?")
    return f"PCIe Switch\\n{bdf}"


def _pcie_label(pcie: dict[str, Any]) -> str:
    gen = pcie.get("generation", 0)
    width = pcie.get("width", 0)
    bw = pcie.get("bandwidth_gbps", 0)
    if gen > 0 and width > 0 and bw > 0:
        return f"PCIe Gen{gen} x{width}\\n{bw:.1f} GB/s"
    if gen > 0 and width > 0:
        return f"PCIe Gen{gen} x{width}"
    return ""


def _nvlink_label(peer: dict[str, Any]) -> str:
    version = peer.get("nvlink_version", 0)
    count = peer.get("link_count", 0)
    bw = peer.get("bandwidth_gbps", 0)
    parts = []
    if version > 0:
        parts.append(f"NVLink {version}")
    else:
        parts.append("NVLink")
    if count > 0:
        parts.append(f"x{count}")
    label = " ".join(parts)
    if bw > 0:
        label += f"\\n{bw:.0f} GB/s"
    return label
