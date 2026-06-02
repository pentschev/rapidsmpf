/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "dashboard.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

namespace rapidsmpf::benchmark::dashboard {

namespace {

std::string json_escape(std::string_view s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\b':
            out += "\\b";
            break;
        case '\f':
            out += "\\f";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                out += "\\u00";
                static constexpr char hex[] = "0123456789abcdef";
                auto v = static_cast<unsigned char>(c);
                out.push_back(hex[v >> 4]);
                out.push_back(hex[v & 0x0f]);
            } else {
                out.push_back(c);
            }
        }
    }
    return out;
}

std::string json_string(std::string_view s) {
    return "\"" + json_escape(s) + "\"";
}

bool write_all(int fd, std::string_view data) {
    while (!data.empty()) {
        ssize_t written = ::send(fd, data.data(), data.size(), MSG_NOSIGNAL);
        if (written <= 0) {
            return false;
        }
        data.remove_prefix(static_cast<std::size_t>(written));
    }
    return true;
}

std::string direction_string(rapidsmpf::ucxx::UCXX::TelemetryEvent::Direction direction) {
    switch (direction) {
    case rapidsmpf::ucxx::UCXX::TelemetryEvent::Direction::Send:
        return "send";
    case rapidsmpf::ucxx::UCXX::TelemetryEvent::Direction::Recv:
        return "recv";
    }
    return "unknown";
}

using TelemetryEvent = rapidsmpf::ucxx::UCXX::TelemetryEvent;

struct TransportFlags {
    bool cuda_ipc{false};
    bool pcie{false};
    bool infiniband{false};
};

TransportFlags transport_flags(std::string_view debug_string);
char const* json_bool(bool value);

void append_transfer_fields(
    std::ostringstream& ss,
    TelemetryEvent const& event,
    std::uint64_t count = 0,
    std::string_view sample_debug_string = {}
) {
    ss << "\"local_rank\":" << event.local_rank << ",\"peer_rank\":" << event.peer_rank
       << ",\"tag\":" << event.tag
       << ",\"direction\":" << json_string(direction_string(event.direction))
       << ",\"bytes\":" << event.bytes << ",\"start_time_us\":" << event.start_time_us
       << ",\"end_time_us\":" << event.end_time_us
       << ",\"duration_seconds\":" << event.duration_seconds
       << ",\"memory_type\":" << json_string(event.memory_type)
       << ",\"debug_string\":" << json_string(event.debug_string);
    if (count > 0) {
        ss << ",\"count\":" << count;
    }
    if (!sample_debug_string.empty()) {
        ss << ",\"sample_debug_string\":" << json_string(sample_debug_string);
    }
    auto const flags = transport_flags(
        sample_debug_string.empty() ? std::string_view{event.debug_string}
                                    : sample_debug_string
    );
    ss << ",\"transport_flags\":{\"cuda_ipc\":" << json_bool(flags.cuda_ipc)
       << ",\"pcie\":" << json_bool(flags.pcie)
       << ",\"infiniband\":" << json_bool(flags.infiniband) << "}";
}

std::string transfer_json(TelemetryEvent const& event) {
    std::ostringstream ss;
    ss << "{\"type\":\"transfer\",";
    append_transfer_fields(ss, event);
    ss << "}";
    return ss.str();
}

std::string lower_copy(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (char c : value) {
        auto const uc = static_cast<unsigned char>(c);
        out.push_back(static_cast<char>(std::tolower(uc)));
    }
    return out;
}

TransportFlags transport_flags(std::string_view debug_string) {
    auto const lower = lower_copy(debug_string);
    return TransportFlags{
        .cuda_ipc = lower.find("cuda_ipc") != std::string::npos
                    || lower.find("cuda ipc") != std::string::npos,
        .pcie = lower.find("cuda_copy") != std::string::npos
                || lower.find("gdr_copy") != std::string::npos
                || lower.find("pcie") != std::string::npos,
        .infiniband = lower.find("infiniband") != std::string::npos
                      || lower.find("mlx5") != std::string::npos
                      || lower.find("/ib") != std::string::npos
                      || lower.find("rc_") != std::string::npos
                      || lower.find("dc_") != std::string::npos
    };
}

char const* json_bool(bool value) {
    return value ? "true" : "false";
}

std::string normalized_debug_key(std::string_view debug_string) {
    auto const lower = lower_copy(debug_string);

    std::vector<std::string> mlx_devices;
    std::size_t pos = 0;
    while ((pos = lower.find("mlx5_", pos)) != std::string::npos) {
        auto end = pos + std::string_view{"mlx5_"}.size();
        while (end < lower.size() && std::isdigit(static_cast<unsigned char>(lower[end])))
        {
            ++end;
        }
        mlx_devices.push_back(lower.substr(pos, end - pos));
        pos = end;
    }
    if (!mlx_devices.empty()) {
        std::sort(mlx_devices.begin(), mlx_devices.end());
        mlx_devices.erase(
            std::unique(mlx_devices.begin(), mlx_devices.end()), mlx_devices.end()
        );
        std::ostringstream ss;
        ss << "infiniband";
        for (auto const& device : mlx_devices) {
            ss << ' ' << device;
        }
        return ss.str();
    }
    if (lower.find("/ib") != std::string::npos || lower.find("rc_") != std::string::npos
        || lower.find("dc_") != std::string::npos)
    {
        return "infiniband";
    }
    if (lower.find("cuda_copy") != std::string::npos
        || lower.find("gdr_copy") != std::string::npos)
    {
        return "pcie";
    }
    if (lower.find("cuda_ipc") != std::string::npos
        || lower.find("cuda ipc") != std::string::npos)
    {
        return "cuda_ipc";
    }
    return std::string{debug_string};
}

char const* index_html() {
    return R"HTML(<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RapidsMPF topology dashboard</title>
<style>
:root {
  color-scheme: dark;
  --bg: #0b0f14;
  --panel: #151b23;
  --frame: #0f1620;
  --line: #30363d;
  --text: #e6edf3;
  --muted: #8b949e;
  --gpu: #55c2a2;
  --nic: #7aa2f7;
  --cpu: #e3b341;
  --rank: #f0883e;
  --switch: #c678dd;
  --fabric: #86e1fc;
}
* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; background: var(--bg); color: var(--text); font: 13px/1.4 system-ui, sans-serif; }
body { overflow: hidden; }
.shell { display: grid; grid-template-rows: auto 1fr; height: 100%; }
.bar { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; padding: 10px 14px; border-bottom: 1px solid var(--line); background: var(--panel); }
.title { font-weight: 650; font-size: 15px; }
.stat { color: var(--muted); white-space: nowrap; }
.stat strong { color: var(--text); font-weight: 600; }
.spacer { flex: 1 1 auto; }
.legend { display: flex; gap: 12px; align-items: center; color: var(--muted); white-space: nowrap; }
.key { display: inline-flex; gap: 5px; align-items: center; }
.swatch { width: 18px; height: 3px; border-radius: 2px; background: #667085; }
.swatch.nvlink { background: var(--gpu); }
.swatch.pcie { background: var(--switch); }
.swatch.transfer { background: var(--rank); }
.swatch.infiniband { background: #ff8a65; }
.tool { border: 1px solid var(--line); background: #21262d; color: var(--text); border-radius: 6px; padding: 4px 9px; font: inherit; cursor: pointer; }
.tool:hover { background: #2b3139; }
#stage { width: 100%; height: 100%; touch-action: none; cursor: grab; }
#stage.dragging { cursor: grabbing; }
.host-frame rect { fill: var(--frame); stroke: #2f3b48; stroke-width: 1.2; }
.host-title { fill: var(--text); font-size: 14px; font-weight: 650; }
.rail { stroke: #26313d; stroke-width: 1; stroke-dasharray: 5 7; }
.rail-label { fill: var(--muted); font-size: 11px; }
.edge { stroke: #667085; stroke-width: 1.5; opacity: .58; fill: none; stroke-linecap: round; stroke-linejoin: round; }
.edge.contains { display: none; }
.edge.affinity { stroke: #7d8590; stroke-width: 1.2; opacity: .34; stroke-dasharray: 4 7; }
.edge.rank_gpu { stroke: #f0883e; stroke-width: 1.7; opacity: .65; stroke-dasharray: 5 5; }
.edge.nvlink { stroke: var(--gpu); stroke-width: 2.4; opacity: .82; }
.edge.nvlink_fabric { stroke: var(--fabric); stroke-width: 2.1; opacity: .72; }
.edge.pcie { stroke: var(--switch); stroke-width: 1.7; stroke-dasharray: 9 6; opacity: .66; }
.edge.infiniband { stroke: #ff8a65; stroke-width: 2.1; opacity: .72; }
.edge.transfer { stroke: var(--rank); stroke-width: 2.6; opacity: .92; }
.edge.transfer.infiniband { stroke: #ff8a65; }
.edge.transfer.pcie { stroke: #b889ff; }
.edge.active { opacity: .98; }
.edge.nvlink.active { stroke-width: 4.2; }
.edge.nvlink_fabric.active { stroke-width: 3.8; }
.edge.pcie.active { stroke-width: 3.2; }
.edge.infiniband.active { stroke-width: 3.8; }
.edge.unknown { stroke-dasharray: 4 6; }
.node rect { stroke: rgba(255,255,255,.24); stroke-width: 1.2; rx: 6; }
.node text { fill: var(--text); font-size: 11px; text-anchor: middle; dominant-baseline: middle; pointer-events: none; }
.node .subtitle { fill: rgba(230,237,243,.7); font-size: 9.5px; }
.node.cpu rect { fill: #b58e35; }
.node.gpu rect { fill: #3b927d; }
.node.nic rect { fill: #4e72c6; }
.node.switch rect { fill: #8a55a6; }
.node.nvswitch rect { fill: #3a91aa; }
.node.fabric rect { fill: #3a91aa; }
.node.rank rect { fill: #b7642e; }
.node.unknown rect { fill: #586069; }
.node.pinned rect { stroke: #ffffff; stroke-width: 1.8; }
.node.disconnected rect { opacity: .55; stroke-dasharray: 5 4; }
.label { fill: var(--muted); font-size: 11px; paint-order: stroke; stroke: rgba(11,15,20,.92); stroke-width: 4px; }
.edge-label { fill: var(--text); font-size: 11px; paint-order: stroke; stroke: rgba(11,15,20,.95); stroke-width: 4px; }
.empty { fill: var(--muted); font-size: 14px; }
</style>
</head>
<body>
<div class="shell">
  <div class="bar">
    <div class="title">RapidsMPF topology dashboard</div>
    <div class="stat">events <strong id="eventCount">0</strong></div>
    <div class="stat">transfer <strong id="totalBytes">0 B</strong></div>
    <div class="stat">active avg <strong id="windowRate">0 B/s</strong></div>
    <div class="stat" id="status">connecting</div>
    <div class="spacer"></div>
    <div class="legend">
      <span class="key"><span class="swatch nvlink"></span>CUDA IPC</span>
      <span class="key"><span class="swatch pcie"></span>PCIe</span>
      <span class="key"><span class="swatch infiniband"></span>InfiniBand</span>
    </div>
    <button class="tool" id="fitView" type="button">Fit</button>
  </div>
  <svg id="stage" role="img" aria-label="Topology transfer graph">
    <g id="viewport">
      <g id="frames"></g>
      <g id="edges"></g>
      <g id="labels"></g>
      <g id="nodes"></g>
    </g>
  </svg>
</div>
<script>
const SVG_NS = 'http://www.w3.org/2000/svg';
const svg = document.getElementById('stage');
const viewport = document.getElementById('viewport');
const framesLayer = document.getElementById('frames');
const edgesLayer = document.getElementById('edges');
const labelsLayer = document.getElementById('labels');
const nodesLayer = document.getElementById('nodes');
const statusEl = document.getElementById('status');
const eventCountEl = document.getElementById('eventCount');
const totalBytesEl = document.getElementById('totalBytes');
const windowRateEl = document.getElementById('windowRate');
const fitViewEl = document.getElementById('fitView');
const state = {
  nodes: new Map(),
  edges: new Map(),
  hostFrames: new Map(),
  rankToGpu: new Map(),
  rankToHost: new Map(),
  recent: [],
  eventCount: 0,
  totalBytes: 0,
  latestEventTimeUs: 0,
  latestEventWallTimeMs: 0,
  view: {x: 0, y: 0, k: 1},
  dragging: null,
  panning: null,
  layoutDirty: true,
  autoFit: true,
  userView: false
};
const nodeBox = {
  cpu: [128, 38],
  gpu: [88, 42],
  nic: [112, 38],
  switch: [120, 34],
  nvswitch: [132, 42],
  fabric: [138, 38],
  rank: [44, 28],
  unknown: [86, 32],
  host: [0, 0]
};
const linkLength = {
  pcie: 105,
  nvlink: 130,
  nvlink_fabric: 115,
  affinity: 175,
  transfer: 245,
  rank_gpu: 68
};
const recentWindowUs = 5_000_000;
function svgEl(tag) {
  return document.createElementNS(SVG_NS, tag);
}
function fmtBytes(v) {
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  let n = Math.max(0, v), i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}
function hash(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return h >>> 0;
}
function normPci(pci) {
  const parts = String(pci || '').toLowerCase().split(':');
  if (parts.length >= 3) {
    parts[0] = parts[0].slice(-4).padStart(4, '0');
    return parts.join(':');
  }
  return String(pci || '').toLowerCase();
}
function gpuNodeId(host, gpu) {
  if (gpu.pci_bus_id) return `gpu:${host}:${normPci(gpu.pci_bus_id)}`;
  return `gpu:${host}:id${gpu.id}`;
}
function shortPci(pci) {
  const parts = String(pci || '').split(':');
  return parts.length >= 3 ? `${parts[1]}:${parts[2]}` : String(pci || '');
}
function naturalKey(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : String(v || '');
}
function compareNode(a, b) {
  const an = Number.isInteger(a.attrs?.numa) ? a.attrs.numa : 9999;
  const bn = Number.isInteger(b.attrs?.numa) ? b.attrs.numa : 9999;
  if (an !== bn) return an - bn;
  const ag = a.attrs?.gpuId ?? a.attrs?.rank ?? a.label ?? a.id;
  const bg = b.attrs?.gpuId ?? b.attrs?.rank ?? b.label ?? b.id;
  const ak = naturalKey(ag), bk = naturalKey(bg);
  return typeof ak === 'number' && typeof bk === 'number'
    ? ak - bk
    : String(ak).localeCompare(String(bk), undefined, {numeric: true});
}
function nodeSize(node) {
  return nodeBox[node.type] || nodeBox.unknown;
}
function collisionRadius(node) {
  const [w, h] = nodeSize(node);
  return Math.max(w, h) * 0.56;
}
function hostOf(node) {
  return node.attrs?.host || (node.type === 'host' ? node.label : 'unassigned');
}
function setTarget(node, x, y) {
  node.tx = x;
  node.ty = y;
  if (!node.placed) {
    node.x = x;
    node.y = y;
    node.vx = 0;
    node.vy = 0;
    node.placed = true;
  }
}
function markLayoutDirty(autoFit = true) {
  state.layoutDirty = true;
  if (autoFit && !state.userView) state.autoFit = true;
}
function ensureNode(id, type, label, attrs = {}) {
  if (!state.nodes.has(id)) {
    const h = hash(id);
    const angle = (h % 6283) / 1000;
    const radius = 240 + (h % 180);
    state.nodes.set(id, {
      id, type, label, attrs,
      x: Math.cos(angle) * radius,
      y: Math.sin(angle) * radius,
      tx: Math.cos(angle) * radius,
      ty: Math.sin(angle) * radius,
      vx: 0, vy: 0, pinned: false, placed: false
    });
    markLayoutDirty();
  } else {
    const node = state.nodes.get(id);
    const oldType = node.type;
    const oldLabel = node.label;
    node.type = type || node.type;
    node.label = label || node.label;
    node.attrs = {...node.attrs, ...attrs};
    if (oldType !== node.type || oldLabel !== node.label) markLayoutDirty();
  }
  return state.nodes.get(id);
}
function ensureEdge(id, source, target, type, attrs = {}) {
  if (source === target) return;
  if (!state.edges.has(id)) {
    state.edges.set(id, {id, source, target, type, attrs, bytes: 0, count: 0, rate: 0, recent: []});
    markLayoutDirty(type !== 'transfer');
  }
  const edge = state.edges.get(id);
  edge.type = type || edge.type;
  edge.attrs = {...edge.attrs, ...attrs};
  return edge;
}
function nvlinkPairs(topo, gpuIds) {
  const pairs = new Map();
  for (const gpu of topo.gpus || []) {
    if (!gpuIds.has(gpu.id)) continue;
    for (const peer of gpu.nvlink_peers || []) {
      if (!gpuIds.has(peer.peer_gpu_id) || gpu.id === peer.peer_gpu_id) continue;
      const lo = Math.min(gpu.id, peer.peer_gpu_id);
      const hi = Math.max(gpu.id, peer.peer_gpu_id);
      const id = `${lo}:${hi}`;
      const existing = pairs.get(id) || {a: lo, b: hi, bandwidth: 0, links: 0};
      existing.bandwidth = Math.max(existing.bandwidth, peer.bandwidth_gbps || 0);
      existing.links = Math.max(existing.links, peer.link_count || 0);
      pairs.set(id, existing);
    }
  }
  return Array.from(pairs.values());
}
function inferNvSwitchFabric(topo, gpuIds) {
  const gpuCount = gpuIds.size;
  if (gpuCount < 4) return null;
  const pairs = nvlinkPairs(topo, gpuIds);
  if (pairs.length !== gpuCount - 1) return null;

  const degree = new Map();
  for (const id of gpuIds.keys()) degree.set(id, 0);
  for (const pair of pairs) {
    degree.set(pair.a, (degree.get(pair.a) || 0) + 1);
    degree.set(pair.b, (degree.get(pair.b) || 0) + 1);
  }
  const center = Array.from(degree.entries()).find(([, value]) => value === gpuCount - 1)?.[0];
  if (center === undefined) return null;

  const perGpu = new Map();
  let fallbackBandwidth = 0;
  let fallbackLinks = 0;
  for (const pair of pairs) {
    if (pair.a !== center && pair.b !== center) return null;
    const gpu = pair.a === center ? pair.b : pair.a;
    const attrs = {bandwidth: pair.bandwidth, links: pair.links, inferredNvSwitch: true};
    perGpu.set(gpu, attrs);
    fallbackBandwidth = Math.max(fallbackBandwidth, pair.bandwidth || 0);
    fallbackLinks = Math.max(fallbackLinks, pair.links || 0);
  }
  perGpu.set(center, {bandwidth: fallbackBandwidth, links: fallbackLinks, inferredNvSwitch: true});
  return {center, perGpu};
}
function keepDirectNvlinkEdgesVisible(gpuIds) {
  const gpuSet = new Set(gpuIds.values());
  const nvlinks = Array.from(state.edges.values()).filter(edge =>
    edge.type === 'nvlink' && gpuSet.has(edge.source) && gpuSet.has(edge.target)
  );
  for (const edge of nvlinks) edge.attrs.hidden = false;
}
function processTopology(topo) {
  const host = topo.system?.hostname || topo.hostname || 'host';
  ensureNode(`host:${host}`, 'host', host, {host});
  const gpuIds = new Map();
  for (const gpu of topo.gpus || []) gpuIds.set(gpu.id, gpuNodeId(host, gpu));
  const nvSwitch = inferNvSwitchFabric(topo, gpuIds);
  for (const cpu of topo.cpus || []) {
    const id = `cpu:${host}:${cpu.numa_node}`;
    ensureNode(id, 'cpu', `CPU ${cpu.numa_node}`, {
      host,
      numa: cpu.numa_node,
      cores: cpu.core_count
    });
    ensureEdge(`contains:${host}:${id}`, `host:${host}`, id, 'contains');
  }
  for (const sw of topo.pcie_switches || []) {
    const id = `switch:${host}:${sw.pci_bus_id}`;
    ensureNode(id, 'switch', shortPci(sw.pci_bus_id) || 'PCIe switch', {
      host,
      numa: sw.numa_node,
      pci: sw.pci_bus_id
    });
    ensureEdge(`contains:${host}:${id}`, `host:${host}`, id, 'contains');
  }
  for (const gpu of topo.gpus || []) {
    const id = gpuIds.get(gpu.id);
    ensureNode(id, 'gpu', `GPU ${gpu.id}`, {
      host,
      pci: gpu.pci_bus_id,
      gpuId: gpu.id,
      numa: gpu.numa_node,
      name: gpu.name
    });
    ensureEdge(`contains:${host}:${id}`, `host:${host}`, id, 'contains');
    if (Number.isInteger(gpu.numa_node)) ensureEdge(`affinity:${host}:gpu${gpu.id}:numa${gpu.numa_node}`, id, `cpu:${host}:${gpu.numa_node}`, 'affinity');
    if (!nvSwitch) {
      for (const peer of gpu.nvlink_peers || []) {
        if (!gpuIds.has(peer.peer_gpu_id)) continue;
        const a = gpu.id < peer.peer_gpu_id ? gpuIds.get(gpu.id) : gpuIds.get(peer.peer_gpu_id);
        const b = gpu.id < peer.peer_gpu_id ? gpuIds.get(peer.peer_gpu_id) : gpuIds.get(gpu.id);
        ensureEdge(`nvlink:${host}:${a}:${b}`, a, b, 'nvlink', {
          bandwidth: peer.bandwidth_gbps,
          links: peer.link_count
        });
      }
    }
    for (const nic of gpu.network_devices || []) ensureEdge(`affinity:${host}:gpu${gpu.id}:${nic}`, id, `nic:${host}:${nic}`, 'affinity');
  }
  if (nvSwitch) {
    const sid = `nvswitch:${host}:0`;
    ensureNode(sid, 'nvswitch', 'NVSwitch', {host, inferred: true, centerGpu: nvSwitch.center});
    ensureEdge(`contains:${host}:${sid}`, `host:${host}`, sid, 'contains');
    for (const [gpuId, nodeId] of gpuIds.entries()) {
      const attrs = nvSwitch.perGpu.get(gpuId) || {};
      ensureEdge(`nvlink:${host}:${sid}:${nodeId}`, sid, nodeId, 'nvlink', attrs);
    }
  }
  for (const nic of topo.network_devices || []) {
    const id = `nic:${host}:${nic.name}`;
    ensureNode(id, 'nic', nic.name, {
      host,
      pci: nic.pci_bus_id,
      numa: nic.numa_node,
      bandwidth: nic.bandwidth_gbps,
      model: nic.model_name
    });
    ensureEdge(`contains:${host}:${id}`, `host:${host}`, id, 'contains');
    if (Number.isInteger(nic.numa_node)) ensureEdge(`affinity:${host}:${nic.name}:numa${nic.numa_node}`, id, `cpu:${host}:${nic.numa_node}`, 'affinity');
  }
  for (const sw of topo.pcie_switches || []) {
    const sid = `switch:${host}:${sw.pci_bus_id}`;
    for (const gid of sw.gpu_ids || []) {
      if (gpuIds.has(gid)) ensureEdge(`pcie:${host}:${sw.pci_bus_id}:gpu${gid}`, sid, gpuIds.get(gid), 'pcie');
    }
    for (const nic of sw.nic_names || []) ensureEdge(`pcie:${host}:${sw.pci_bus_id}:${nic}`, sid, `nic:${host}:${nic}`, 'pcie');
  }
  keepDirectNvlinkEdgesVisible(gpuIds);
  markLayoutDirty();
}
function processRank(ev) {
  const host = ev.hostname || 'host';
  ensureNode(`host:${host}`, 'host', host, {host});
  const rid = `rank:${ev.rank}`;
  ensureNode(rid, 'rank', `R${ev.rank}`, {host, rank: ev.rank, cudaDevice: ev.cuda_device});
  state.rankToHost.set(ev.rank, host);
  ensureEdge(`contains:${host}:${rid}`, `host:${host}`, rid, 'contains');
  if (ev.gpu_pci_bus_id) {
    const gid = `gpu:${host}:${normPci(ev.gpu_pci_bus_id)}`;
    ensureNode(gid, 'gpu', `GPU ${shortPci(ev.gpu_pci_bus_id)}`, {
      host,
      pci: ev.gpu_pci_bus_id,
      visibleDevice: ev.cuda_device
    });
    state.rankToGpu.set(ev.rank, gid);
    ensureEdge(`rank_gpu:${ev.rank}`, rid, gid, 'rank_gpu');
  }
  markLayoutDirty();
}
function debugText(ev) {
  return `${ev.debug_string || ''} ${ev.sample_debug_string || ''}`.toLowerCase();
}
function classifyTransfer(ev) {
  const d = debugText(ev);
  if (d.includes('infiniband') || d.includes('mlx5') || d.includes('/ib') || d.includes('rc_') || d.includes('dc_')) return 'infiniband';
  if (d.includes('cuda_copy') || d.includes('gdr_copy') || d.includes('pcie')) return 'pcie';
  if (d.includes('cuda_ipc') || d.includes('cuda ipc')) return 'cuda_ipc';
  return 'unknown';
}
function transportLabel(transport) {
  if (transport === 'cuda_ipc') return 'CUDA IPC';
  if (transport === 'infiniband') return 'InfiniBand';
  if (transport === 'pcie') return 'PCIe';
  return 'Transfer';
}
function uniqueEdges(edges) {
  const seen = new Set();
  return edges.filter(edge => {
    if (!edge || seen.has(edge.id)) return false;
    seen.add(edge.id);
    return true;
  });
}
function endpointEdges(nodeId, type) {
  return Array.from(state.edges.values()).filter(edge =>
    edge.type === type &&
    !edge.attrs?.hidden &&
    (edge.source === nodeId || edge.target === nodeId)
  );
}
function transferGpus(ev) {
  return [state.rankToGpu.get(ev.local_rank), state.rankToGpu.get(ev.peer_rank)];
}
function pcieRouteForGpus(gpuA, gpuB) {
  return uniqueEdges([
    ...endpointEdges(gpuA, 'pcie'),
    ...endpointEdges(gpuB, 'pcie')
  ]);
}
function eventEndUs(ev) {
  const end = Number(ev.end_time_us);
  if (Number.isFinite(end) && end > 0) return end;
  return Date.now() * 1000;
}
function eventDurationSeconds(ev) {
  const duration = Number(ev.duration_seconds);
  if (Number.isFinite(duration) && duration > 0) return duration;
  const start = Number(ev.start_time_us);
  const end = Number(ev.end_time_us);
  if (Number.isFinite(start) && Number.isFinite(end) && end > start) {
    return (end - start) / 1_000_000;
  }
  return 0;
}
function eventStartUs(ev) {
  const start = Number(ev.start_time_us);
  if (Number.isFinite(start) && start > 0) return start;
  const duration = eventDurationSeconds(ev);
  return eventEndUs(ev) - duration * 1_000_000;
}
function transferSample(ev) {
  return {
    start: eventStartUs(ev),
    t: eventEndUs(ev),
    bytes: ev.bytes || 0,
    duration: eventDurationSeconds(ev)
  };
}
function rateFromSamples(samples) {
  if (samples.length === 0) return 0;
  const bytes = samples.reduce((sum, item) => sum + item.bytes, 0);
  const start = Math.min(...samples.map(item => item.start));
  const end = Math.max(...samples.map(item => item.t));
  const activeSeconds = Number.isFinite(start) && Number.isFinite(end) && end > start
    ? (end - start) / 1_000_000
    : 0;
  const fallbackSeconds = samples.reduce((sum, item) => sum + item.duration, 0);
  const seconds = activeSeconds > 0 ? activeSeconds : fallbackSeconds;
  return seconds > 0 ? bytes / seconds : 0;
}
function shortestEdgePath(source, target, type) {
  if (source === target) return [];
  const seen = new Set([source]);
  const previous = new Map();
  const queue = [source];
  for (let head = 0; head < queue.length; head++) {
    const nodeId = queue[head];
    for (const edge of state.edges.values()) {
      if (edge.type !== type || edge.attrs?.hidden) continue;
      const next = edge.source === nodeId ? edge.target : edge.target === nodeId ? edge.source : undefined;
      if (!next || seen.has(next)) continue;
      seen.add(next);
      previous.set(next, {node: nodeId, edge});
      if (next === target) {
        const path = [];
        for (let cur = target; cur !== source;) {
          const step = previous.get(cur);
          if (!step) return [];
          path.push(step.edge);
          cur = step.node;
        }
        return path.reverse();
      }
      queue.push(next);
    }
  }
  return [];
}
function cudaIpcRoute(ev) {
  const [gpuA, gpuB] = transferGpus(ev);
  if (!gpuA || !gpuB) return [];
  return shortestEdgePath(gpuA, gpuB, 'nvlink');
}
function parseMlxDevices(ev) {
  return Array.from(new Set(debugText(ev).match(/mlx5_\d+/g) || []));
}
function nicNodesByName(name) {
  return Array.from(state.nodes.values()).filter(node =>
    node.type === 'nic' && (node.label === name || node.id.endsWith(`:${name}`))
  );
}
function chooseNicForRank(rank) {
  const gpu = state.nodes.get(state.rankToGpu.get(rank));
  const host = gpu?.attrs?.host || state.rankToHost.get(rank);
  const numa = gpu?.attrs?.numa;
  const candidates = Array.from(state.nodes.values()).filter(node =>
    node.type === 'nic' && (!host || node.attrs?.host === host)
  );
  candidates.sort((a, b) => {
    const an = a.attrs?.numa === numa ? 0 : 1;
    const bn = b.attrs?.numa === numa ? 0 : 1;
    if (an !== bn) return an - bn;
    return compareNode(a, b);
  });
  return candidates[0];
}
function infinibandRoute(ev) {
  const nics = parseMlxDevices(ev).flatMap(name => nicNodesByName(name));
  if (nics.length === 0) {
    const localNic = chooseNicForRank(ev.local_rank);
    const peerNic = chooseNicForRank(ev.peer_rank);
    if (localNic) nics.push(localNic);
    if (peerNic && peerNic.id !== localNic?.id) nics.push(peerNic);
  }

  const edges = [];
  for (const nic of nics) {
    edges.push(...endpointEdges(nic.id, 'pcie'));
  }
  if (nics.length >= 2) {
    for (let i = 0; i < nics.length; i++) {
      for (let j = i + 1; j < nics.length; j++) {
        if (nics[i].id === nics[j].id) continue;
        edges.push(ensureEdge(
          `infiniband:${nics[i].id}:${nics[j].id}`,
          nics[i].id,
          nics[j].id,
          'infiniband',
          {transport: 'infiniband'}
        ));
      }
    }
  }
  return uniqueEdges(edges);
}
function routeTransfer(ev, transport) {
  if (transport === 'cuda_ipc') return cudaIpcRoute(ev);
  if (transport === 'pcie') {
    const [gpuA, gpuB] = transferGpus(ev);
    if (gpuA && gpuB) return pcieRouteForGpus(gpuA, gpuB);
  }
  if (transport === 'infiniband') return infinibandRoute(ev);
  return [];
}
function addTransferToPhysicalEdge(edge, ev, transport) {
  edge.bytes += ev.bytes || 0;
  edge.count += 1;
  edge.recent.push(transferSample(ev));
  edge.attrs.transport = transport;
}
function processTransfer(ev) {
  ensureNode(`rank:${ev.local_rank}`, 'rank', `R${ev.local_rank}`, {rank: ev.local_rank});
  ensureNode(`rank:${ev.peer_rank}`, 'rank', `R${ev.peer_rank}`, {rank: ev.peer_rank});
  if (ev.direction !== 'send') return;
  const a = Math.min(ev.local_rank, ev.peer_rank);
  const b = Math.max(ev.local_rank, ev.peer_rank);
  const transport = classifyTransfer(ev);
  let mappedEdges = routeTransfer(ev, transport);
  if (mappedEdges.length === 0 && transport !== 'cuda_ipc') {
    const [gpuA, gpuB] = transferGpus(ev);
    mappedEdges = [
      ensureEdge(
        `transfer:${transport}:${a}:${b}`,
        gpuA || `rank:${a}`,
        gpuB || `rank:${b}`,
        'transfer',
        {transport}
      )
    ];
  }
  for (const edge of mappedEdges) addTransferToPhysicalEdge(edge, ev, transport);
  state.totalBytes += ev.bytes || 0;
  const sample = transferSample(ev);
  state.recent.push(sample);
  if (sample.t >= (state.latestEventTimeUs || 0)) {
    state.latestEventTimeUs = sample.t;
    state.latestEventWallTimeMs = Date.now();
  }
}
function processEvent(ev) {
  state.eventCount++;
  if (ev.type === 'topology') processTopology(ev.topology);
  else if (ev.type === 'rank') processRank(ev);
  else if (ev.type === 'transfer') processTransfer(ev);
  else if (ev.type === 'transfer_batch') {
    const events = Array.isArray(ev.events) ? ev.events : [];
    const count = events.reduce((sum, item) => sum + Math.max(1, Number(item.count) || 1), 0);
    state.eventCount += Math.max(0, count - 1);
    for (const item of events) processTransfer(item);
  }
  else if (ev.type === 'topology_error') statusEl.textContent = `topology: ${ev.message || 'error'}`;
  eventCountEl.textContent = String(state.eventCount);
  totalBytesEl.textContent = fmtBytes(state.totalBytes);
}
function edgeClass(e) {
  return `edge ${e.type || ''} ${e.attrs?.transport || ''} ${e.bytes > 0 ? 'active' : ''} ${e.attrs?.confidence || ''}`;
}
function visibleNodes() {
  return Array.from(state.nodes.values()).filter(node => node.type !== 'host');
}
function visibleEdges() {
  return Array.from(state.edges.values()).filter(edge =>
    edge.type !== 'contains' && !edge.attrs?.hidden &&
    state.nodes.has(edge.source) && state.nodes.has(edge.target)
  );
}
function buildHostSpecs() {
  const hosts = new Map();
  for (const node of visibleNodes()) {
    const host = hostOf(node);
    if (!hosts.has(host)) hosts.set(host, []);
    hosts.get(host).push(node);
  }
  for (const node of state.nodes.values()) {
    if (node.type === 'host' && !hosts.has(hostOf(node))) hosts.set(hostOf(node), []);
  }
  return Array.from(hosts.entries()).sort((a, b) => a[0].localeCompare(b[0], undefined, {numeric: true})).map(([host, nodes]) => {
    const numaSet = new Set();
    for (const node of nodes) if (Number.isInteger(node.attrs?.numa)) numaSet.add(node.attrs.numa);
    if (numaSet.size === 0) numaSet.add(0);
    const numa = Array.from(numaSet).sort((a, b) => a - b);
    const gpuCount = nodes.filter(node => node.type === 'gpu').length;
    const nicCount = nodes.filter(node => node.type === 'nic').length;
    const rankCount = nodes.filter(node => node.type === 'rank').length;
    const switchCount = nodes.filter(node => node.type === 'switch').length;
    const width = Math.max(
      760,
      numa.length * 280 + 130,
      Math.ceil(Math.max(1, gpuCount) / 2) * 150 + 230,
      Math.max(1, nicCount) * 125 + 180,
      Math.max(1, rankCount) * 58 + 180
    );
    const rankRows = Math.max(0, Math.ceil(rankCount / Math.max(1, gpuCount || 4)) - 1);
    const height = Math.max(560, 500 + rankRows * 34 + Math.max(0, switchCount - 4) * 12);
    return {host, nodes, numa, width, height};
  });
}
function packHosts(specs) {
  const rect = svg.getBoundingClientRect();
  const maxRowWidth = Math.max(900, Math.min(2600, (rect.width || 1200) / Math.max(0.35, state.view.k)));
  const gap = 120;
  let x = 0, y = 0, rowHeight = 0, rowStart = 0;
  const placed = [];
  for (const spec of specs) {
    if (x > rowStart && x + spec.width > maxRowWidth) {
      x = 0;
      y += rowHeight + gap;
      rowStart = 0;
      rowHeight = 0;
    }
    placed.push({...spec, left: x, top: y});
    x += spec.width + gap;
    rowHeight = Math.max(rowHeight, spec.height);
  }
  if (placed.length === 0) return placed;
  const minX = Math.min(...placed.map(p => p.left));
  const maxX = Math.max(...placed.map(p => p.left + p.width));
  const minY = Math.min(...placed.map(p => p.top));
  const maxY = Math.max(...placed.map(p => p.top + p.height));
  const dx = -(minX + maxX) / 2;
  const dy = -(minY + maxY) / 2;
  return placed.map(p => ({...p, left: p.left + dx, top: p.top + dy}));
}
function laneMap(spec, left, innerLeft, innerRight) {
  const lanes = new Map();
  const count = Math.max(1, spec.numa.length);
  for (let i = 0; i < spec.numa.length; i++) {
    const x = count === 1 ? (innerLeft + innerRight) / 2 : innerLeft + i * (innerRight - innerLeft) / (count - 1);
    lanes.set(spec.numa[i], x);
  }
  return lanes;
}
function targetForNuma(node, lanes, fallback) {
  return lanes.get(node.attrs?.numa) ?? fallback;
}
function layoutLine(nodes, y, minX, maxX, idealX) {
  const sorted = nodes.slice().sort((a, b) => {
    const ax = idealX(a), bx = idealX(b);
    if (Math.abs(ax - bx) > 1) return ax - bx;
    return compareNode(a, b);
  });
  if (sorted.length === 0) return;
  if (sorted.length === 1) {
    setTarget(sorted[0], Math.max(minX, Math.min(maxX, idealX(sorted[0]))), y);
    return;
  }
  const step = Math.min(150, Math.max(88, (maxX - minX) / Math.max(1, sorted.length - 1)));
  const total = step * (sorted.length - 1);
  const start = (minX + maxX) / 2 - total / 2;
  for (let i = 0; i < sorted.length; i++) {
    const x = Math.max(minX, Math.min(maxX, start + i * step));
    setTarget(sorted[i], x, y);
  }
}
function layoutGpuGroup(nodes, centerX, centerY, width, rowsHint) {
  const sorted = nodes.slice().sort(compareNode);
  if (sorted.length === 0) return;
  const rows = rowsHint || (sorted.length > 4 ? 2 : 1);
  const cols = Math.ceil(sorted.length / rows);
  const stepX = cols <= 1 ? 0 : Math.min(140, Math.max(96, width / Math.max(1, cols - 1)));
  const total = stepX * (cols - 1);
  const x0 = centerX - total / 2;
  for (let i = 0; i < sorted.length; i++) {
    const row = Math.floor(i / cols);
    const col = i % cols;
    const y = centerY + (rows === 1 ? 0 : (row === 0 ? -66 : 66));
    setTarget(sorted[i], x0 + col * stepX, y);
  }
}
function layoutNvlinkMesh(nodes, lanes, centerX, centerY, width, numaValues) {
  const groups = new Map();
  for (const gpu of nodes) {
    const numa = Number.isInteger(gpu.attrs?.numa) ? gpu.attrs.numa : numaValues[0];
    if (!groups.has(numa)) groups.set(numa, []);
    groups.get(numa).push(gpu);
  }
  if (groups.size <= 1) {
    layoutGpuGroup(nodes, centerX, centerY, width);
    return;
  }
  const laneWidth = Math.max(180, width / Math.max(1, groups.size));
  for (const [numa, gpus] of groups.entries()) {
    layoutGpuGroup(
      gpus,
      lanes.get(numa) ?? centerX,
      centerY,
      laneWidth * 0.78,
      gpus.length > 2 ? 2 : 1
    );
  }
}
function connectedTargetX(node, fallbackX) {
  const xs = [];
  for (const edge of state.edges.values()) {
    if (edge.source !== node.id && edge.target !== node.id) continue;
    const other = state.nodes.get(edge.source === node.id ? edge.target : edge.source);
    if (other && Number.isFinite(other.tx)) xs.push(other.tx);
  }
  return xs.length ? xs.reduce((a, x) => a + x, 0) / xs.length : fallbackX;
}
function placeHost(spec) {
  const left = spec.left;
  const top = spec.top;
  const width = spec.width;
  const height = spec.height;
  const centerX = left + width / 2;
  const marginX = 82;
  const innerLeft = left + marginX;
  const innerRight = left + width - marginX;
  const lanes = laneMap(spec, left, innerLeft, innerRight);
  state.hostFrames.set(spec.host, {
    x: left,
    y: top,
    w: width,
    h: height,
    host: spec.host,
    rails: Array.from(lanes.entries()).map(([numa, x]) => ({numa, x}))
  });

  const cpus = spec.nodes.filter(node => node.type === 'cpu').sort(compareNode);
  for (const cpu of cpus) setTarget(cpu, targetForNuma(cpu, lanes, centerX), top + 70);

  const fabric = spec.nodes.find(node => node.type === 'fabric');
  const nvSwitch = spec.nodes.find(node => node.type === 'nvswitch');
  const gpus = spec.nodes.filter(node => node.type === 'gpu').sort(compareNode);
  const gpuY = top + height * 0.47;
  if (nvSwitch && gpus.length >= 2) {
    setTarget(nvSwitch, centerX, gpuY);
    const rows = gpus.length > 4 ? 2 : 1;
    const cols = Math.ceil(gpus.length / rows);
    const stepX = cols <= 1 ? 0 : Math.min(150, Math.max(105, (innerRight - innerLeft) / Math.max(1, cols - 1)));
    const total = stepX * (cols - 1);
    const x0 = centerX - total / 2;
    for (let i = 0; i < gpus.length; i++) {
      const row = Math.floor(i / cols);
      const col = i % cols;
      const y = rows === 1 ? gpuY + 96 : gpuY + (row === 0 ? -96 : 96);
      setTarget(gpus[i], x0 + col * stepX, y);
    }
  } else if (fabric && gpus.length >= 6) {
    setTarget(fabric, centerX, gpuY);
    const rows = gpus.length > 4 ? 2 : 1;
    const cols = Math.ceil(gpus.length / rows);
    const stepX = cols <= 1 ? 0 : Math.min(150, Math.max(105, (innerRight - innerLeft) / Math.max(1, cols - 1)));
    const total = stepX * (cols - 1);
    const x0 = centerX - total / 2;
    for (let i = 0; i < gpus.length; i++) {
      const row = Math.floor(i / cols);
      const col = i % cols;
      const y = rows === 1 ? gpuY + 90 : gpuY + (row === 0 ? -92 : 92);
      setTarget(gpus[i], x0 + col * stepX, y);
    }
  } else if (
    gpus.length >= 6 &&
    visibleEdges().some(edge =>
      edge.type === 'nvlink' &&
      gpus.some(gpu => gpu.id === edge.source) &&
      gpus.some(gpu => gpu.id === edge.target)
    )
  ) {
    layoutNvlinkMesh(gpus, lanes, centerX, gpuY, innerRight - innerLeft, spec.numa);
  } else {
    const groups = new Map();
    for (const gpu of gpus) {
      const numa = Number.isInteger(gpu.attrs?.numa) ? gpu.attrs.numa : spec.numa[0];
      if (!groups.has(numa)) groups.set(numa, []);
      groups.get(numa).push(gpu);
    }
    const laneWidth = Math.max(140, (innerRight - innerLeft) / Math.max(1, spec.numa.length));
    for (const [numa, nodes] of groups.entries()) {
      layoutGpuGroup(nodes, lanes.get(numa) ?? centerX, gpuY, laneWidth * 0.82);
    }
  }

  const switches = spec.nodes.filter(node => node.type === 'switch').sort(compareNode);
  layoutLine(
    switches,
    top + 165,
    innerLeft,
    innerRight,
    node => connectedTargetX(node, targetForNuma(node, lanes, centerX))
  );

  const nics = spec.nodes.filter(node => node.type === 'nic').sort(compareNode);
  layoutLine(
    nics,
    top + height - 116,
    innerLeft,
    innerRight,
    node => connectedTargetX(node, targetForNuma(node, lanes, centerX))
  );

  const ranks = spec.nodes.filter(node => node.type === 'rank').sort(compareNode);
  const ranksByGpu = new Map();
  const unboundRanks = [];
  for (const rank of ranks) {
    const edge = Array.from(state.edges.values()).find(e => e.type === 'rank_gpu' && e.source === rank.id);
    const gpu = edge ? state.nodes.get(edge.target) : undefined;
    if (!gpu || !Number.isFinite(gpu.tx)) {
      unboundRanks.push(rank);
      continue;
    }
    if (!ranksByGpu.has(gpu.id)) ranksByGpu.set(gpu.id, []);
    ranksByGpu.get(gpu.id).push(rank);
  }
  for (const [gpuId, gpuRanks] of ranksByGpu.entries()) {
    const gpu = state.nodes.get(gpuId);
    gpuRanks.sort(compareNode);
    const row = gpu.ty >= gpuY ? 1 : -1;
    const baseY = gpu.ty + row * 62;
    const startX = gpu.tx - (gpuRanks.length - 1) * 28 / 2;
    for (let i = 0; i < gpuRanks.length; i++) setTarget(gpuRanks[i], startX + i * 28, baseY);
  }
  layoutLine(unboundRanks, top + height - 54, innerLeft, innerRight, () => centerX);

  const leftovers = spec.nodes.filter(node => !Number.isFinite(node.tx) || !Number.isFinite(node.ty));
  layoutLine(leftovers, top + height - 54, innerLeft, innerRight, () => centerX);
}
function computeLayout() {
  const specs = packHosts(buildHostSpecs());
  state.hostFrames.clear();
  for (const spec of specs) placeHost(spec);
  state.layoutDirty = false;
}
function relaxLayout() {
  const nodes = visibleNodes();
  const edges = visibleEdges();
  for (let step = 0; step < 2; step++) {
    for (const edge of edges) {
      const a = state.nodes.get(edge.source), b = state.nodes.get(edge.target);
      if (!a || !b) continue;
      const dx = b.x - a.x, dy = b.y - a.y;
      const dist = Math.max(1, Math.hypot(dx, dy));
      const target = linkLength[edge.type] || 140;
      const strength = edge.type === 'transfer' ? 0.0008 : 0.0018;
      const force = (dist - target) * strength;
      const fx = dx / dist * force, fy = dy / dist * force;
      if (!a.pinned) { a.vx += fx; a.vy += fy; }
      if (!b.pinned) { b.vx -= fx; b.vy -= fy; }
    }
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i], b = nodes[j];
        const dx = b.x - a.x, dy = b.y - a.y;
        const dist = Math.max(1, Math.hypot(dx, dy));
        const minDist = collisionRadius(a) + collisionRadius(b) + (hostOf(a) === hostOf(b) ? 8 : 34);
        if (dist >= minDist) continue;
        const push = (minDist - dist) * 0.025;
        const fx = dx / dist * push, fy = dy / dist * push;
        if (!a.pinned) { a.vx -= fx; a.vy -= fy; }
        if (!b.pinned) { b.vx += fx; b.vy += fy; }
      }
    }
    for (const n of nodes) {
      if (n.pinned) continue;
      n.vx += (n.tx - n.x) * 0.16;
      n.vy += (n.ty - n.y) * 0.16;
      n.x += n.vx;
      n.y += n.vy;
      n.vx *= 0.68;
      n.vy *= 0.68;
    }
  }
}
function contentBounds() {
  const xs = [], ys = [];
  for (const frame of state.hostFrames.values()) {
    xs.push(frame.x, frame.x + frame.w);
    ys.push(frame.y, frame.y + frame.h);
  }
  for (const node of visibleNodes()) {
    const [w, h] = nodeSize(node);
    xs.push(node.x - w / 2, node.x + w / 2);
    ys.push(node.y - h / 2, node.y + h / 2);
  }
  if (xs.length === 0) return null;
  return {minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys)};
}
function fitView() {
  const bounds = contentBounds();
  const rect = svg.getBoundingClientRect();
  if (!bounds || rect.width <= 0 || rect.height <= 0) return;
  const pad = 52;
  const width = Math.max(1, bounds.maxX - bounds.minX);
  const height = Math.max(1, bounds.maxY - bounds.minY);
  const scale = Math.max(0.12, Math.min(1.7, Math.min((rect.width - pad * 2) / width, (rect.height - pad * 2) / height)));
  state.view.k = scale;
  state.view.x = rect.width / 2 - (bounds.minX + bounds.maxX) * scale / 2;
  state.view.y = rect.height / 2 - (bounds.minY + bounds.maxY) * scale / 2;
}
function edgePoint(from, to) {
  const [w, h] = nodeSize(from);
  const dx = to.x - from.x, dy = to.y - from.y;
  if (Math.abs(dx) < 1e-6 && Math.abs(dy) < 1e-6) return {x: from.x, y: from.y};
  const sx = Math.abs(dx) > 1e-6 ? (w / 2) / Math.abs(dx) : Infinity;
  const sy = Math.abs(dy) > 1e-6 ? (h / 2) / Math.abs(dy) : Infinity;
  const s = Math.min(sx, sy, 1);
  return {x: from.x + dx * s, y: from.y + dy * s};
}
function edgePath(edge, a, b) {
  const p0 = edgePoint(a, b);
  const p1 = edgePoint(b, a);
  if (edge.type === 'transfer') {
    const dx = p1.x - p0.x, dy = p1.y - p0.y;
    const dist = Math.max(1, Math.hypot(dx, dy));
    const bend = Math.min(120, dist * 0.24) * (hash(edge.id) % 2 ? 1 : -1);
    const mx = (p0.x + p1.x) / 2 - dy / dist * bend;
    const my = (p0.y + p1.y) / 2 + dx / dist * bend;
    return `M ${p0.x} ${p0.y} Q ${mx} ${my} ${p1.x} ${p1.y}`;
  }
  if (edge.type === 'pcie' || edge.type === 'affinity' || edge.type === 'rank_gpu' || edge.type === 'nvlink_fabric') {
    const midY = (p0.y + p1.y) / 2;
    return `M ${p0.x} ${p0.y} C ${p0.x} ${midY} ${p1.x} ${midY} ${p1.x} ${p1.y}`;
  }
  if (edge.type === 'nvlink' && Math.abs(p0.y - p1.y) > 28) {
    const midX = (p0.x + p1.x) / 2;
    return `M ${p0.x} ${p0.y} C ${midX} ${p0.y} ${midX} ${p1.y} ${p1.x} ${p1.y}`;
  }
  return `M ${p0.x} ${p0.y} L ${p1.x} ${p1.y}`;
}
function truncate(text, maxLen) {
  const s = String(text || '');
  return s.length > maxLen ? s.slice(0, Math.max(0, maxLen - 1)) + '...' : s;
}
function nodeSubtitle(node) {
  if (node.type === 'cpu') return Number.isFinite(node.attrs?.cores) && node.attrs.cores > 0 ? `${node.attrs.cores} cores` : `NUMA ${node.attrs?.numa ?? '?'}`;
  if (node.type === 'gpu') return node.attrs?.pci ? shortPci(node.attrs.pci) : '';
  if (node.type === 'nic') return node.attrs?.bandwidth ? `${node.attrs.bandwidth} GB/s` : 'disconnected';
  if (node.type === 'switch') return 'PCIe switch';
  if (node.type === 'nvswitch') return 'NVLink fabric';
  if (node.type === 'fabric') return 'aggregated links';
  if (node.type === 'rank') return 'rank';
  return '';
}
function render() {
  if (state.layoutDirty) computeLayout();
  relaxLayout();
  if (state.autoFit) {
    fitView();
    state.autoFit = false;
  }
  const now = Date.now();
  const eventNowUs = state.latestEventTimeUs > 0
    ? state.latestEventTimeUs + Math.max(0, now - (state.latestEventWallTimeMs || now)) * 1000
    : now * 1000;
  const cutoffUs = eventNowUs - recentWindowUs;
  state.recent = state.recent.filter(x => x.t >= cutoffUs);
  windowRateEl.textContent = fmtBytes(rateFromSamples(state.recent)) + '/s';
  viewport.setAttribute('transform', `translate(${state.view.x} ${state.view.y}) scale(${state.view.k})`);
  framesLayer.replaceChildren();
  edgesLayer.replaceChildren();
  labelsLayer.replaceChildren();
  nodesLayer.replaceChildren();
  if (state.nodes.size === 0) {
    const t = svgEl('text');
    t.setAttribute('class', 'empty');
    t.setAttribute('x', 24);
    t.setAttribute('y', 44);
    t.textContent = 'Waiting for topology and transfer events';
    nodesLayer.appendChild(t);
  }
  for (const frame of state.hostFrames.values()) {
    const g = svgEl('g');
    g.setAttribute('class', 'host-frame');
    const rect = svgEl('rect');
    rect.setAttribute('x', frame.x);
    rect.setAttribute('y', frame.y);
    rect.setAttribute('width', frame.w);
    rect.setAttribute('height', frame.h);
    rect.setAttribute('rx', 10);
    const title = svgEl('text');
    title.setAttribute('class', 'host-title');
    title.setAttribute('x', frame.x + 18);
    title.setAttribute('y', frame.y + 26);
    title.textContent = frame.host;
    g.append(rect, title);
    for (const rail of frame.rails) {
      const line = svgEl('line');
      line.setAttribute('class', 'rail');
      line.setAttribute('x1', rail.x);
      line.setAttribute('x2', rail.x);
      line.setAttribute('y1', frame.y + 42);
      line.setAttribute('y2', frame.y + frame.h - 28);
      const label = svgEl('text');
      label.setAttribute('class', 'rail-label');
      label.setAttribute('x', rail.x + 8);
      label.setAttribute('y', frame.y + 48);
      label.textContent = `NUMA ${rail.numa}`;
      g.append(line, label);
    }
    framesLayer.appendChild(g);
  }
  for (const e of visibleEdges()) {
    const a = state.nodes.get(e.source), b = state.nodes.get(e.target);
    if (!a || !b) continue;
    const path = svgEl('path');
    path.setAttribute('class', edgeClass(e));
    path.setAttribute('d', edgePath(e, a, b));
    if (e.type === 'nvlink' && e.attrs?.bandwidth) {
      path.setAttribute('stroke-width', String(1.8 + Math.min(3.8, e.attrs.bandwidth / 35) + (e.bytes > 0 ? 1.4 : 0)));
    }
    if (e.type === 'pcie' && e.bytes > 0) {
      path.setAttribute('stroke-width', String(2.4 + Math.min(4, Math.log2(Math.max(1, e.bytes)) / 7)));
    }
    if ((e.type === 'transfer' || e.type === 'infiniband' || e.type === 'nvlink_fabric') && e.bytes > 0) {
      path.setAttribute('stroke-width', String(2.0 + Math.min(6, Math.log2(Math.max(1, e.bytes)) / 5)));
    }
    edgesLayer.appendChild(path);
    if (e.bytes > 0) {
      e.recent = e.recent.filter(x => x.t >= cutoffUs);
      e.rate = rateFromSamples(e.recent);
      const label = svgEl('text');
      label.setAttribute('class', 'edge-label');
      label.setAttribute('x', (a.x + b.x) / 2 + 4);
      label.setAttribute('y', (a.y + b.y) / 2 - 4);
      label.textContent = `${transportLabel(e.attrs?.transport)} ${fmtBytes(e.bytes)} ${fmtBytes(e.rate)}/s`;
      labelsLayer.appendChild(label);
    }
  }
  for (const n of visibleNodes()) {
    const g = svgEl('g');
    const disconnected = n.type === 'nic' && !(n.attrs?.bandwidth > 0);
    g.setAttribute('class', `node ${n.type || 'unknown'}${n.pinned ? ' pinned' : ''}${disconnected ? ' disconnected' : ''}`);
    g.setAttribute('transform', `translate(${n.x} ${n.y})`);
    g.dataset.id = n.id;
    const [w, h] = nodeSize(n);
    const box = svgEl('rect');
    box.setAttribute('x', -w / 2);
    box.setAttribute('y', -h / 2);
    box.setAttribute('width', w);
    box.setAttribute('height', h);
    box.setAttribute('rx', 6);
    const title = svgEl('title');
    title.textContent = n.id;
    const txt = svgEl('text');
    const subtitleText = nodeSubtitle(n);
    txt.setAttribute('y', subtitleText ? -6 : 1);
    txt.textContent = truncate(n.label || n.id, n.type === 'rank' ? 4 : 16);
    g.append(title, box, txt);
    if (subtitleText) {
      const sub = svgEl('text');
      sub.setAttribute('class', 'subtitle');
      sub.setAttribute('y', 10);
      sub.textContent = truncate(subtitleText, 18);
      g.appendChild(sub);
    }
    nodesLayer.appendChild(g);
  }
  requestAnimationFrame(render);
}
function svgPoint(ev) {
  const rect = svg.getBoundingClientRect();
  return {x: (ev.clientX - rect.left - state.view.x) / state.view.k, y: (ev.clientY - rect.top - state.view.y) / state.view.k};
}
svg.addEventListener('pointerdown', ev => {
  const nodeEl = ev.target.closest?.('.node');
  svg.setPointerCapture(ev.pointerId);
  if (nodeEl) {
    const node = state.nodes.get(nodeEl.dataset.id);
    const p = svgPoint(ev);
    state.dragging = {node, dx: node.x - p.x, dy: node.y - p.y};
    node.pinned = true;
  } else {
    state.panning = {x: ev.clientX, y: ev.clientY, vx: state.view.x, vy: state.view.y};
    state.userView = true;
    svg.classList.add('dragging');
  }
});
svg.addEventListener('pointermove', ev => {
  if (state.dragging) {
    const p = svgPoint(ev), n = state.dragging.node;
    n.x = p.x + state.dragging.dx; n.y = p.y + state.dragging.dy; n.vx = 0; n.vy = 0;
  } else if (state.panning) {
    state.view.x = state.panning.vx + ev.clientX - state.panning.x;
    state.view.y = state.panning.vy + ev.clientY - state.panning.y;
  }
});
svg.addEventListener('pointerup', ev => { state.dragging = null; state.panning = null; svg.classList.remove('dragging'); });
svg.addEventListener('dblclick', ev => {
  const nodeEl = ev.target.closest?.('.node');
  if (nodeEl) state.nodes.get(nodeEl.dataset.id).pinned = false;
});
svg.addEventListener('wheel', ev => {
  ev.preventDefault();
  const factor = ev.deltaY < 0 ? 1.12 : 0.89;
  const rect = svg.getBoundingClientRect();
  const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
  state.view.x = mx - (mx - state.view.x) * factor;
  state.view.y = my - (my - state.view.y) * factor;
  state.view.k = Math.max(0.15, Math.min(4, state.view.k * factor));
  state.userView = true;
}, {passive: false});
fitViewEl.addEventListener('click', () => {
  if (state.layoutDirty) computeLayout();
  fitView();
  state.userView = false;
});
window.addEventListener('resize', () => {
  if (!state.userView) state.autoFit = true;
});
const events = new EventSource('/events');
events.onopen = () => { statusEl.textContent = 'connected'; };
events.onerror = () => { statusEl.textContent = 'reconnecting'; };
events.onmessage = msg => {
  try { processEvent(JSON.parse(msg.data)); }
  catch (err) { console.warn('bad event', err, msg.data); }
};
render();
</script>
</body>
</html>)HTML";
}

}  // namespace

JsonlEventSink::JsonlEventSink(std::filesystem::path path) : path_{std::move(path)} {}

std::filesystem::path const& JsonlEventSink::path() const noexcept {
    return path_;
}

void JsonlEventSink::clear() const {
    std::ofstream out(path_, std::ios::trunc);
    if (!out) {
        throw std::runtime_error(
            "failed to truncate dashboard event file: " + path_.string()
        );
    }
}

void JsonlEventSink::publish_raw(std::string const& json) const {
    int fd = ::open(path_.c_str(), O_CREAT | O_WRONLY | O_APPEND, 0644);
    if (fd < 0) {
        return;
    }
    std::string line = json;
    line.push_back('\n');
    std::ignore = ::write(fd, line.data(), line.size());
    ::close(fd);
}

void JsonlEventSink::publish_rank(
    rapidsmpf::Rank rank,
    rapidsmpf::Rank nranks,
    std::string const& hostname,
    int cuda_device,
    std::string const& gpu_pci_bus_id
) const {
    std::ostringstream ss;
    ss << "{\"type\":\"rank\",\"rank\":" << rank << ",\"nranks\":" << nranks
       << ",\"hostname\":" << json_string(hostname) << ",\"cuda_device\":" << cuda_device
       << ",\"gpu_pci_bus_id\":" << json_string(gpu_pci_bus_id) << "}";
    publish_raw(ss.str());
}

void JsonlEventSink::publish_topology(std::string const& topology_json) const {
    publish_raw("{\"type\":\"topology\",\"topology\":" + topology_json + "}");
}

void JsonlEventSink::publish_topology_error(std::string const& message) const {
    publish_raw("{\"type\":\"topology_error\",\"message\":" + json_string(message) + "}");
}

void JsonlEventSink::publish_transfer(
    rapidsmpf::ucxx::UCXX::TelemetryEvent const& event
) const {
    publish_raw(transfer_json(event));
}

class TelemetryBatcher::Impl {
  public:
    Impl(std::shared_ptr<JsonlEventSink> sink, std::chrono::milliseconds interval)
        : sink_{std::move(sink)}, interval_{interval}, last_flush_{Clock::now()} {}

    ~Impl() {
        flush();
    }

    void ingest(TelemetryEvent const& event) {
        if (event.direction != TelemetryEvent::Direction::Send) {
            return;
        }

        std::vector<Bucket> batch;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            add_locked(event);
            auto const now = Clock::now();
            if (now - last_flush_ < interval_) {
                return;
            }
            batch = drain_locked(now);
        }
        publish(batch);
    }

    void flush() {
        std::vector<Bucket> batch;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            batch = drain_locked(Clock::now());
        }
        publish(batch);
    }

  private:
    using Clock = std::chrono::steady_clock;

    struct Bucket {
        TelemetryEvent event{};
        std::uint64_t count{0};
        std::string sample_debug_string;
    };

    static std::string key_for(
        TelemetryEvent const& event, std::string const& debug_key
    ) {
        std::ostringstream ss;
        ss << event.local_rank << '\x1f' << event.peer_rank << '\x1f' << event.tag
           << '\x1f' << static_cast<int>(event.direction) << '\x1f' << event.memory_type
           << '\x1f' << debug_key;
        return ss.str();
    }

    void add_locked(TelemetryEvent const& event) {
        auto debug_key = normalized_debug_key(event.debug_string);
        auto& bucket = buckets_[key_for(event, debug_key)];
        if (bucket.count == 0) {
            bucket.event = event;
            bucket.sample_debug_string = event.debug_string;
            bucket.event.debug_string = std::move(debug_key);
        } else {
            auto& aggregate = bucket.event;
            aggregate.bytes += event.bytes;
            if (event.start_time_us > 0
                && (aggregate.start_time_us == 0
                    || event.start_time_us < aggregate.start_time_us))
            {
                aggregate.start_time_us = event.start_time_us;
            }
            aggregate.end_time_us = std::max(aggregate.end_time_us, event.end_time_us);
            aggregate.duration_seconds += event.duration_seconds;
        }
        ++bucket.count;
    }

    std::vector<Bucket> drain_locked(Clock::time_point now) {
        std::vector<Bucket> batch;
        batch.reserve(buckets_.size());
        for (auto& item : buckets_) {
            batch.push_back(std::move(item.second));
        }
        buckets_.clear();
        last_flush_ = now;
        return batch;
    }

    void publish(std::vector<Bucket> const& batch) const {
        if (!sink_ || batch.empty()) {
            return;
        }

        std::ostringstream ss;
        ss << "{\"type\":\"transfer_batch\",\"events\":[";
        bool first = true;
        for (auto const& bucket : batch) {
            if (!first) {
                ss << ',';
            }
            first = false;
            ss << '{';
            append_transfer_fields(
                ss, bucket.event, bucket.count, bucket.sample_debug_string
            );
            ss << '}';
        }
        ss << "]}";
        sink_->publish_raw(ss.str());
    }

    std::shared_ptr<JsonlEventSink> sink_;
    std::chrono::milliseconds interval_;
    Clock::time_point last_flush_;
    std::mutex mutex_;
    std::unordered_map<std::string, Bucket> buckets_;
};

TelemetryBatcher::TelemetryBatcher(
    std::shared_ptr<JsonlEventSink> sink, std::chrono::milliseconds interval
)
    : impl_{std::make_unique<Impl>(std::move(sink), interval)} {}

TelemetryBatcher::~TelemetryBatcher() = default;

void TelemetryBatcher::ingest(rapidsmpf::ucxx::UCXX::TelemetryEvent const& event) {
    impl_->ingest(event);
}

void TelemetryBatcher::flush() {
    impl_->flush();
}

class Server::Impl {
  public:
    Impl(std::filesystem::path event_file, std::uint16_t requested_port)
        : event_file_{std::move(event_file)} {
        listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd_ < 0) {
            throw std::runtime_error("failed to create dashboard socket");
        }

        int opt = 1;
        setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = htons(requested_port);
        if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
            ::close(listen_fd_);
            listen_fd_ = -1;
            throw std::runtime_error("failed to bind dashboard socket");
        }
        if (::listen(listen_fd_, 16) != 0) {
            ::close(listen_fd_);
            listen_fd_ = -1;
            throw std::runtime_error("failed to listen on dashboard socket");
        }

        sockaddr_in actual{};
        socklen_t len = sizeof(actual);
        if (::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&actual), &len) == 0) {
            port_ = ntohs(actual.sin_port);
        } else {
            port_ = requested_port;
        }

        thread_ = std::thread([this]() { accept_loop(); });
    }

    ~Impl() {
        stop_ = true;
        if (listen_fd_ >= 0) {
            ::shutdown(listen_fd_, SHUT_RDWR);
            ::close(listen_fd_);
            listen_fd_ = -1;
        }
        if (thread_.joinable()) {
            thread_.join();
        }
        std::lock_guard<std::mutex> lock(client_threads_mutex_);
        for (auto& thread : client_threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

    [[nodiscard]] std::uint16_t port() const noexcept {
        return port_;
    }

  private:
    void accept_loop() {
        while (!stop_) {
            fd_set rfds;
            FD_ZERO(&rfds);
            FD_SET(listen_fd_, &rfds);
            timeval tv{0, 200000};
            int ready = ::select(listen_fd_ + 1, &rfds, nullptr, nullptr, &tv);
            if (ready <= 0 || stop_) {
                continue;
            }
            int client_fd = ::accept(listen_fd_, nullptr, nullptr);
            if (client_fd < 0) {
                continue;
            }
            std::lock_guard<std::mutex> lock(client_threads_mutex_);
            client_threads_.emplace_back([this, client_fd]() {
                handle_client(client_fd);
            });
        }
    }

    void handle_client(int client_fd) {
        std::array<char, 2048> buffer{};
        ssize_t n = ::recv(client_fd, buffer.data(), buffer.size() - 1, 0);
        if (n <= 0) {
            ::close(client_fd);
            return;
        }
        std::string request(buffer.data(), static_cast<std::size_t>(n));
        std::string path = "/";
        auto first_space = request.find(' ');
        if (first_space != std::string::npos) {
            auto second_space = request.find(' ', first_space + 1);
            if (second_space != std::string::npos) {
                path = request.substr(first_space + 1, second_space - first_space - 1);
            }
        }
        if (path == "/events") {
            stream_events(client_fd);
        } else if (path == "/health") {
            std::string body = "ok\n";
            std::ostringstream response;
            response << "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: "
                     << body.size() << "\r\nConnection: close\r\n\r\n"
                     << body;
            std::ignore = write_all(client_fd, response.str());
        } else {
            std::string body = index_html();
            std::ostringstream response;
            response << "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n"
                     << "Content-Length: " << body.size()
                     << "\r\nConnection: close\r\n\r\n"
                     << body;
            std::ignore = write_all(client_fd, response.str());
        }
        ::close(client_fd);
    }

    void stream_events(int client_fd) {
        std::string headers =
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\nConnection: keep-alive\r\n"
            "Access-Control-Allow-Origin: *\r\n\r\n";
        if (!write_all(client_fd, headers)) {
            return;
        }

        std::uintmax_t offset = 0;
        while (!stop_) {
            std::ifstream in(event_file_);
            if (in) {
                in.seekg(static_cast<std::streamoff>(offset));
                std::string line;
                while (std::getline(in, line)) {
                    offset += line.size() + 1;
                    if (line.empty()) {
                        continue;
                    }
                    if (!write_all(client_fd, "data: " + line + "\n\n")) {
                        return;
                    }
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds{250});
        }
    }

    std::filesystem::path event_file_;
    int listen_fd_{-1};
    std::uint16_t port_{0};
    std::atomic<bool> stop_{false};
    std::thread thread_;
    std::mutex client_threads_mutex_;
    std::vector<std::thread> client_threads_;
};

Server::Server(std::filesystem::path event_file, std::uint16_t requested_port)
    : impl_{std::make_unique<Impl>(std::move(event_file), requested_port)} {}

Server::~Server() = default;

std::uint16_t Server::port() const noexcept {
    return impl_->port();
}

std::string Server::url() const {
    return "http://127.0.0.1:" + std::to_string(port()) + "/";
}

std::string default_event_file() {
    return "/tmp/rapidsmpf_bench_comm_dashboard.jsonl";
}

std::string hostname() {
    std::array<char, 256> name{};
    if (::gethostname(name.data(), name.size() - 1) == 0) {
        return std::string{name.data()};
    }
    return "unknown";
}

}  // namespace rapidsmpf::benchmark::dashboard
