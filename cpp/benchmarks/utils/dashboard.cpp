/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "dashboard.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <tuple>
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
  --bg: #0d1117;
  --panel: #151b23;
  --line: #30363d;
  --text: #e6edf3;
  --muted: #8b949e;
  --gpu: #55c2a2;
  --nic: #7aa2f7;
  --cpu: #e3b341;
  --rank: #f0883e;
  --switch: #c678dd;
}
* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; background: var(--bg); color: var(--text); font: 13px/1.4 system-ui, sans-serif; }
body { overflow: hidden; }
.shell { display: grid; grid-template-rows: auto 1fr; height: 100%; }
.bar { display: flex; gap: 20px; align-items: center; padding: 10px 14px; border-bottom: 1px solid var(--line); background: var(--panel); }
.title { font-weight: 650; font-size: 15px; }
.stat { color: var(--muted); }
.stat strong { color: var(--text); font-weight: 600; }
#stage { width: 100%; height: 100%; touch-action: none; cursor: grab; }
#stage.dragging { cursor: grabbing; }
.edge { stroke: #667085; stroke-width: 1.4; opacity: .58; fill: none; }
.edge.nvlink { stroke: #55c2a2; stroke-width: 2.1; opacity: .8; }
.edge.pcie { stroke: #c678dd; stroke-dasharray: 8 5; }
.edge.transfer { stroke: #f0883e; stroke-width: 2.4; opacity: .9; }
.edge.unknown { stroke-dasharray: 4 6; }
.node circle { stroke: rgba(255,255,255,.22); stroke-width: 1.2; }
.node text { fill: var(--text); paint-order: stroke; stroke: rgba(13,17,23,.88); stroke-width: 4px; font-size: 12px; pointer-events: none; }
.node.host circle { fill: #30363d; }
.node.cpu circle { fill: var(--cpu); }
.node.gpu circle { fill: var(--gpu); }
.node.nic circle { fill: var(--nic); }
.node.switch circle { fill: var(--switch); }
.node.rank circle { fill: var(--rank); }
.node.unknown circle { fill: #8b949e; }
.label { fill: var(--muted); font-size: 11px; paint-order: stroke; stroke: rgba(13,17,23,.9); stroke-width: 4px; }
.empty { fill: var(--muted); font-size: 14px; }
</style>
</head>
<body>
<div class="shell">
  <div class="bar">
    <div class="title">RapidsMPF topology dashboard</div>
    <div class="stat">events <strong id="eventCount">0</strong></div>
    <div class="stat">transfer <strong id="totalBytes">0 B</strong></div>
    <div class="stat">window <strong id="windowRate">0 B/s</strong></div>
    <div class="stat" id="status">connecting</div>
  </div>
  <svg id="stage" role="img" aria-label="Topology transfer graph">
    <g id="viewport">
      <g id="edges"></g>
      <g id="labels"></g>
      <g id="nodes"></g>
    </g>
  </svg>
</div>
<script>
const svg = document.getElementById('stage');
const viewport = document.getElementById('viewport');
const edgesLayer = document.getElementById('edges');
const labelsLayer = document.getElementById('labels');
const nodesLayer = document.getElementById('nodes');
const statusEl = document.getElementById('status');
const eventCountEl = document.getElementById('eventCount');
const totalBytesEl = document.getElementById('totalBytes');
const windowRateEl = document.getElementById('windowRate');
const state = {
  nodes: new Map(),
  edges: new Map(),
  recent: [],
  eventCount: 0,
  totalBytes: 0,
  view: {x: 0, y: 0, k: 1},
  dragging: null,
  panning: null
};
const typeRadius = {host: 24, cpu: 17, gpu: 18, nic: 17, switch: 15, rank: 13, unknown: 12};
const linkLength = {contains: 90, pcie: 110, nvlink: 78, affinity: 95, transfer: 150, rank_gpu: 58};
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
function ensureNode(id, type, label, attrs = {}) {
  if (!state.nodes.has(id)) {
    const h = hash(id);
    const angle = (h % 6283) / 1000;
    const radius = 120 + (h % 220);
    state.nodes.set(id, {
      id, type, label, attrs,
      x: Math.cos(angle) * radius,
      y: Math.sin(angle) * radius,
      vx: 0, vy: 0, pinned: false
    });
  } else {
    const node = state.nodes.get(id);
    node.type = type || node.type;
    node.label = label || node.label;
    node.attrs = {...node.attrs, ...attrs};
  }
  return state.nodes.get(id);
}
function ensureEdge(id, source, target, type, attrs = {}) {
  if (source === target) return;
  if (!state.edges.has(id)) state.edges.set(id, {id, source, target, type, attrs, bytes: 0, count: 0, rate: 0});
  const edge = state.edges.get(id);
  edge.type = type || edge.type;
  edge.attrs = {...edge.attrs, ...attrs};
  return edge;
}
function processTopology(topo) {
  const host = topo.system?.hostname || 'host';
  ensureNode(`host:${host}`, 'host', host, {host});
  const gpuIds = new Map();
  for (const gpu of topo.gpus || []) gpuIds.set(gpu.id, gpuNodeId(host, gpu));
  for (const cpu of topo.cpus || []) {
    const id = `cpu:${host}:${cpu.numa_node}`;
    ensureNode(id, 'cpu', `NUMA ${cpu.numa_node}`, {host});
    ensureEdge(`contains:${host}:${id}`, `host:${host}`, id, 'contains');
  }
  for (const sw of topo.pcie_switches || []) {
    const id = `switch:${host}:${sw.pci_bus_id}`;
    ensureNode(id, 'switch', sw.pci_bus_id || 'PCIe switch', {host});
    ensureEdge(`contains:${host}:${id}`, `host:${host}`, id, 'contains');
  }
  for (const gpu of topo.gpus || []) {
    const id = gpuIds.get(gpu.id);
    ensureNode(id, 'gpu', `GPU ${gpu.id}`, {host, pci: gpu.pci_bus_id});
    ensureEdge(`contains:${host}:${id}`, `host:${host}`, id, 'contains');
    if (Number.isInteger(gpu.numa_node)) ensureEdge(`affinity:${host}:gpu${gpu.id}:numa${gpu.numa_node}`, id, `cpu:${host}:${gpu.numa_node}`, 'affinity');
    for (const peer of gpu.nvlink_peers || []) {
      if (!gpuIds.has(peer.peer_gpu_id)) continue;
      const a = gpu.id < peer.peer_gpu_id ? gpuIds.get(gpu.id) : gpuIds.get(peer.peer_gpu_id);
      const b = gpu.id < peer.peer_gpu_id ? gpuIds.get(peer.peer_gpu_id) : gpuIds.get(gpu.id);
      ensureEdge(`nvlink:${host}:${a}:${b}`, a, b, 'nvlink', {bandwidth: peer.bandwidth_gbps});
    }
    for (const nic of gpu.network_devices || []) ensureEdge(`affinity:${host}:gpu${gpu.id}:${nic}`, id, `nic:${host}:${nic}`, 'affinity');
  }
  for (const nic of topo.network_devices || []) {
    const id = `nic:${host}:${nic.name}`;
    ensureNode(id, 'nic', nic.name, {host, pci: nic.pci_bus_id});
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
}
function processRank(ev) {
  const host = ev.hostname || 'host';
  ensureNode(`host:${host}`, 'host', host, {host});
  const rid = `rank:${ev.rank}`;
  ensureNode(rid, 'rank', `R${ev.rank}`, {host});
  ensureEdge(`contains:${host}:${rid}`, `host:${host}`, rid, 'contains');
  if (ev.gpu_pci_bus_id) {
    const gid = `gpu:${host}:${normPci(ev.gpu_pci_bus_id)}`;
    ensureNode(gid, 'gpu', `GPU ${ev.cuda_device}`, {host, pci: ev.gpu_pci_bus_id});
    ensureEdge(`rank_gpu:${ev.rank}`, rid, gid, 'rank_gpu');
  }
}
function classifyTransfer(ev) {
  const d = (ev.debug_string || '').toLowerCase();
  if (d.includes('cuda_ipc') || d.includes('cuda ipc')) return 'nvlink';
  if (d.includes('mlx5') || d.includes('/ib') || d.includes('rc_') || d.includes('dc_')) return 'transfer';
  return 'unknown';
}
function processTransfer(ev) {
  ensureNode(`rank:${ev.local_rank}`, 'rank', `R${ev.local_rank}`);
  ensureNode(`rank:${ev.peer_rank}`, 'rank', `R${ev.peer_rank}`);
  if (ev.direction !== 'send') return;
  const a = ev.local_rank, b = ev.peer_rank;
  const edge = ensureEdge(`transfer:${a}:${b}`, `rank:${a}`, `rank:${b}`, classifyTransfer(ev));
  edge.type = 'transfer';
  edge.bytes += ev.bytes || 0;
  edge.count += 1;
  state.totalBytes += ev.bytes || 0;
  state.recent.push({t: Date.now(), bytes: ev.bytes || 0});
}
function processEvent(ev) {
  state.eventCount++;
  if (ev.type === 'topology') processTopology(ev.topology);
  else if (ev.type === 'rank') processRank(ev);
  else if (ev.type === 'transfer') processTransfer(ev);
  eventCountEl.textContent = String(state.eventCount);
  totalBytesEl.textContent = fmtBytes(state.totalBytes);
}
function edgeClass(e) {
  return `edge ${e.type || ''} ${e.attrs?.confidence || ''}`;
}
function tick() {
  const nodes = Array.from(state.nodes.values());
  const edges = Array.from(state.edges.values());
  const byHost = new Map();
  for (const n of nodes) {
    const host = n.attrs.host || 'default';
    if (!byHost.has(host)) byHost.set(host, []);
    byHost.get(host).push(n);
  }
  const hostCenters = new Map();
  Array.from(byHost.keys()).forEach((host, i, arr) => {
    const angle = arr.length === 1 ? 0 : (Math.PI * 2 * i) / arr.length;
    hostCenters.set(host, {x: Math.cos(angle) * 280, y: Math.sin(angle) * 220});
  });
  for (let step = 0; step < 3; step++) {
    for (const e of edges) {
      const a = state.nodes.get(e.source), b = state.nodes.get(e.target);
      if (!a || !b) continue;
      const dx = b.x - a.x, dy = b.y - a.y;
      const dist = Math.max(1, Math.hypot(dx, dy));
      const target = linkLength[e.type] || 120;
      const f = (dist - target) * 0.0025;
      const fx = dx / dist * f, fy = dy / dist * f;
      if (!a.pinned) { a.vx += fx; a.vy += fy; }
      if (!b.pinned) { b.vx -= fx; b.vy -= fy; }
    }
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i], b = nodes[j];
        const dx = b.x - a.x, dy = b.y - a.y;
        const d2 = Math.max(90, dx * dx + dy * dy);
        const f = 80 / d2;
        const dist = Math.sqrt(d2);
        const fx = dx / dist * f, fy = dy / dist * f;
        if (!a.pinned) { a.vx -= fx; a.vy -= fy; }
        if (!b.pinned) { b.vx += fx; b.vy += fy; }
      }
    }
    for (const n of nodes) {
      const host = n.attrs.host || 'default';
      const c = hostCenters.get(host) || {x: 0, y: 0};
      const typePull = n.type === 'host' ? 0.035 : 0.008;
      if (!n.pinned) {
        n.vx += (c.x - n.x) * typePull;
        n.vy += (c.y - n.y) * typePull;
        n.x += n.vx;
        n.y += n.vy;
        n.vx *= 0.78;
        n.vy *= 0.78;
      }
    }
  }
}
function render() {
  tick();
  const now = Date.now();
  state.recent = state.recent.filter(x => now - x.t < 5000);
  const windowBytes = state.recent.reduce((a, x) => a + x.bytes, 0);
  windowRateEl.textContent = fmtBytes(windowBytes / 5) + '/s';
  viewport.setAttribute('transform', `translate(${state.view.x} ${state.view.y}) scale(${state.view.k})`);
  edgesLayer.replaceChildren();
  labelsLayer.replaceChildren();
  nodesLayer.replaceChildren();
  if (state.nodes.size === 0) {
    const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    t.setAttribute('class', 'empty');
    t.setAttribute('x', 24);
    t.setAttribute('y', 44);
    t.textContent = 'Waiting for topology and transfer events';
    nodesLayer.appendChild(t);
  }
  for (const e of state.edges.values()) {
    const a = state.nodes.get(e.source), b = state.nodes.get(e.target);
    if (!a || !b) continue;
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('class', edgeClass(e));
    line.setAttribute('x1', a.x);
    line.setAttribute('y1', a.y);
    line.setAttribute('x2', b.x);
    line.setAttribute('y2', b.y);
    if (e.type === 'transfer') line.setAttribute('stroke-width', String(1.5 + Math.min(6, Math.log2(Math.max(1, e.bytes)) / 5)));
    edgesLayer.appendChild(line);
    if (e.type === 'transfer' && e.bytes > 0) {
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('class', 'label');
      label.setAttribute('x', (a.x + b.x) / 2 + 4);
      label.setAttribute('y', (a.y + b.y) / 2 - 4);
      label.textContent = fmtBytes(e.bytes);
      labelsLayer.appendChild(label);
    }
  }
  for (const n of state.nodes.values()) {
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.setAttribute('class', `node ${n.type || 'unknown'}`);
    g.setAttribute('transform', `translate(${n.x} ${n.y})`);
    g.dataset.id = n.id;
    const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    c.setAttribute('r', typeRadius[n.type] || typeRadius.unknown);
    const txt = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    txt.setAttribute('x', (typeRadius[n.type] || typeRadius.unknown) + 5);
    txt.setAttribute('y', 4);
    txt.textContent = n.label || n.id;
    g.append(c, txt);
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
}, {passive: false});
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
    std::ostringstream ss;
    ss << "{\"type\":\"transfer\",\"local_rank\":" << event.local_rank
       << ",\"peer_rank\":" << event.peer_rank << ",\"tag\":" << event.tag
       << ",\"direction\":" << json_string(direction_string(event.direction))
       << ",\"bytes\":" << event.bytes << ",\"start_time_us\":" << event.start_time_us
       << ",\"end_time_us\":" << event.end_time_us
       << ",\"duration_seconds\":" << event.duration_seconds
       << ",\"memory_type\":" << json_string(event.memory_type)
       << ",\"debug_string\":" << json_string(event.debug_string) << "}";
    publish_raw(ss.str());
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
