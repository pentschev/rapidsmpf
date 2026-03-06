/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <variant>

#include <rapidsmpf/topology/json_io.hpp>

namespace rapidsmpf::topology {

namespace {

// ---------------------------------------------------------------------------
// Minimal JSON value representation for parsing
// ---------------------------------------------------------------------------

class json_value;
using json_object = std::map<std::string, json_value>;
using json_array = std::vector<json_value>;

class json_value {
  public:
    using variant_t =
        std::variant<std::monostate, bool, double, std::string, json_array, json_object>;

    json_value() = default;

    explicit json_value(bool v) : data_{v} {}

    explicit json_value(double v) : data_{v} {}

    explicit json_value(std::string v) : data_{std::move(v)} {}

    explicit json_value(json_array v) : data_{std::move(v)} {}

    explicit json_value(json_object v) : data_{std::move(v)} {}

    [[nodiscard]] bool is_null() const {
        return std::holds_alternative<std::monostate>(data_);
    }

    [[nodiscard]] bool is_string() const {
        return std::holds_alternative<std::string>(data_);
    }

    [[nodiscard]] bool is_number() const {
        return std::holds_alternative<double>(data_);
    }

    [[nodiscard]] bool is_array() const {
        return std::holds_alternative<json_array>(data_);
    }

    [[nodiscard]] bool is_object() const {
        return std::holds_alternative<json_object>(data_);
    }

    [[nodiscard]] std::string const& as_string() const {
        return std::get<std::string>(data_);
    }

    [[nodiscard]] double as_number() const {
        return std::get<double>(data_);
    }

    [[nodiscard]] int as_int() const {
        return static_cast<int>(std::get<double>(data_));
    }

    [[nodiscard]] unsigned int as_uint() const {
        return static_cast<unsigned int>(std::get<double>(data_));
    }

    [[nodiscard]] json_array const& as_array() const {
        return std::get<json_array>(data_);
    }

    [[nodiscard]] json_object const& as_object() const {
        return std::get<json_object>(data_);
    }

    [[nodiscard]] std::string get_string(
        std::string const& key, std::string const& def = ""
    ) const {
        if (!is_object())
            return def;
        auto const& obj = as_object();
        auto it = obj.find(key);
        if (it == obj.end() || !it->second.is_string())
            return def;
        return it->second.as_string();
    }

    [[nodiscard]] double get_number(std::string const& key, double def = 0) const {
        if (!is_object())
            return def;
        auto const& obj = as_object();
        auto it = obj.find(key);
        if (it == obj.end() || !it->second.is_number())
            return def;
        return it->second.as_number();
    }

    [[nodiscard]] int get_int(std::string const& key, int def = 0) const {
        return static_cast<int>(get_number(key, def));
    }

    [[nodiscard]] unsigned int get_uint(
        std::string const& key, unsigned int def = 0
    ) const {
        return static_cast<unsigned int>(get_number(key, def));
    }

    [[nodiscard]] json_value const* find(std::string const& key) const {
        if (!is_object())
            return nullptr;
        auto const& obj = as_object();
        auto it = obj.find(key);
        if (it == obj.end())
            return nullptr;
        return &it->second;
    }

  private:
    variant_t data_;
};

// ---------------------------------------------------------------------------
// Recursive descent JSON parser
// ---------------------------------------------------------------------------

class json_parser {
  public:
    explicit json_parser(std::string_view input) : input_{input}, pos_{0} {}

    json_value parse() {
        skip_ws();
        auto val = parse_value();
        skip_ws();
        if (pos_ < input_.size()) {
            throw std::runtime_error(
                "JSON: trailing content at position " + std::to_string(pos_)
            );
        }
        return val;
    }

  private:
    std::string_view input_;
    std::size_t pos_;

    [[nodiscard]] char peek() const {
        if (pos_ >= input_.size())
            throw std::runtime_error("JSON: unexpected end of input");
        return input_[pos_];
    }

    char advance() {
        char c = peek();
        ++pos_;
        return c;
    }

    void expect(char c) {
        char got = advance();
        if (got != c) {
            throw std::runtime_error(
                std::string("JSON: expected '") + c + "' but got '" + got + "' at "
                + std::to_string(pos_ - 1)
            );
        }
    }

    void skip_ws() {
        while (pos_ < input_.size()
               && (input_[pos_] == ' ' || input_[pos_] == '\t' || input_[pos_] == '\n'
                   || input_[pos_] == '\r'))
        {
            ++pos_;
        }
    }

    json_value parse_value() {
        skip_ws();
        char c = peek();
        if (c == '"')
            return json_value{parse_string()};
        if (c == '{')
            return json_value{parse_object()};
        if (c == '[')
            return json_value{parse_array()};
        if (c == 't' || c == 'f')
            return json_value{parse_bool()};
        if (c == 'n') {
            parse_null();
            return json_value{};
        }
        return json_value{parse_number()};
    }

    std::string parse_string() {
        expect('"');
        std::string result;
        while (true) {
            char c = advance();
            if (c == '"')
                break;
            if (c == '\\') {
                char esc = advance();
                switch (esc) {
                case '"':
                    result += '"';
                    break;
                case '\\':
                    result += '\\';
                    break;
                case '/':
                    result += '/';
                    break;
                case 'b':
                    result += '\b';
                    break;
                case 'f':
                    result += '\f';
                    break;
                case 'n':
                    result += '\n';
                    break;
                case 'r':
                    result += '\r';
                    break;
                case 't':
                    result += '\t';
                    break;
                case 'u':
                    // Skip 4 hex digits (basic handling)
                    for (int i = 0; i < 4; ++i)
                        advance();
                    result += '?';
                    break;
                default:
                    result += esc;
                    break;
                }
            } else {
                result += c;
            }
        }
        return result;
    }

    double parse_number() {
        std::size_t start = pos_;
        if (peek() == '-')
            ++pos_;
        while (pos_ < input_.size()
               && std::isdigit(static_cast<unsigned char>(input_[pos_])))
            ++pos_;
        if (pos_ < input_.size() && input_[pos_] == '.') {
            ++pos_;
            while (pos_ < input_.size()
                   && std::isdigit(static_cast<unsigned char>(input_[pos_])))
                ++pos_;
        }
        if (pos_ < input_.size() && (input_[pos_] == 'e' || input_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < input_.size() && (input_[pos_] == '+' || input_[pos_] == '-'))
                ++pos_;
            while (pos_ < input_.size()
                   && std::isdigit(static_cast<unsigned char>(input_[pos_])))
                ++pos_;
        }
        std::string num_str{input_.substr(start, pos_ - start)};
        return std::stod(num_str);
    }

    bool parse_bool() {
        if (input_.substr(pos_, 4) == "true") {
            pos_ += 4;
            return true;
        }
        if (input_.substr(pos_, 5) == "false") {
            pos_ += 5;
            return false;
        }
        throw std::runtime_error("JSON: invalid boolean at " + std::to_string(pos_));
    }

    void parse_null() {
        if (input_.substr(pos_, 4) != "null") {
            throw std::runtime_error("JSON: invalid null at " + std::to_string(pos_));
        }
        pos_ += 4;
    }

    json_array parse_array() {
        expect('[');
        json_array arr;
        skip_ws();
        if (peek() == ']') {
            ++pos_;
            return arr;
        }
        while (true) {
            arr.push_back(parse_value());
            skip_ws();
            if (peek() == ']') {
                ++pos_;
                return arr;
            }
            expect(',');
        }
    }

    json_object parse_object() {
        expect('{');
        json_object obj;
        skip_ws();
        if (peek() == '}') {
            ++pos_;
            return obj;
        }
        while (true) {
            skip_ws();
            std::string key = parse_string();
            skip_ws();
            expect(':');
            obj[std::move(key)] = parse_value();
            skip_ws();
            if (peek() == '}') {
                ++pos_;
                return obj;
            }
            expect(',');
        }
    }
};

// ---------------------------------------------------------------------------
// JSON serialization helpers
// ---------------------------------------------------------------------------

std::string escape_json(std::string const& s) {
    std::string out;
    out.reserve(s.size());
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
            out += c;
        }
    }
    return out;
}

class json_writer {
  public:
    explicit json_writer(int indent) : indent_{indent} {}

    [[nodiscard]] std::string const& str() const {
        return buf_;
    }

    void begin_object() {
        buf_ += '{';
        ++depth_;
        first_ = true;
    }

    void end_object() {
        --depth_;
        if (!first_)
            newline();
        buf_ += '}';
        first_ = false;
    }

    void begin_array() {
        buf_ += '[';
        ++depth_;
        first_ = true;
    }

    void end_array() {
        --depth_;
        if (!first_)
            newline();
        buf_ += ']';
        first_ = false;
    }

    void key(std::string const& k) {
        comma();
        newline();
        buf_ += '"';
        buf_ += escape_json(k);
        buf_ += "\": ";
    }

    void value_string(std::string const& v) {
        buf_ += '"';
        buf_ += escape_json(v);
        buf_ += '"';
        first_ = false;
    }

    void value_int(int v) {
        buf_ += std::to_string(v);
        first_ = false;
    }

    void value_uint(unsigned int v) {
        buf_ += std::to_string(v);
        first_ = false;
    }

    void value_double(double v) {
        std::ostringstream oss;
        oss << v;
        buf_ += oss.str();
        first_ = false;
    }

    void array_element_sep() {
        comma();
        if (indent_ > 0) {
            newline();
        } else {
            buf_ += ' ';
        }
    }

  private:
    int indent_;
    int depth_{0};
    bool first_{true};
    std::string buf_;

    void comma() {
        if (!first_)
            buf_ += ',';
    }

    void newline() {
        if (indent_ <= 0)
            return;
        buf_ += '\n';
        for (int i = 0; i < depth_ * indent_; ++i)
            buf_ += ' ';
    }
};

// ---------------------------------------------------------------------------
// Deserialization helpers
// ---------------------------------------------------------------------------

std::vector<int> parse_int_array(json_value const& val) {
    std::vector<int> result;
    if (!val.is_array())
        return result;
    for (auto const& elem : val.as_array()) {
        if (elem.is_number())
            result.push_back(elem.as_int());
    }
    return result;
}

std::vector<std::string> parse_string_array(json_value const& val) {
    std::vector<std::string> result;
    if (!val.is_array())
        return result;
    for (auto const& elem : val.as_array()) {
        if (elem.is_string())
            result.push_back(elem.as_string());
    }
    return result;
}

pcie_info parse_pcie(json_value const& val) {
    pcie_info info;
    if (!val.is_object())
        return info;
    info.generation = val.get_int("generation");
    info.width = val.get_int("width");
    info.bandwidth_gbps = val.get_number("bandwidth_gbps");
    return info;
}

nvlink_connection parse_nvlink(json_value const& val) {
    nvlink_connection conn;
    if (!val.is_object())
        return conn;
    conn.peer_gpu_id = val.get_uint("peer_gpu_id");
    conn.link_count = val.get_int("link_count");
    conn.nvlink_version = val.get_int("nvlink_version");
    conn.bandwidth_gbps = val.get_number("bandwidth_gbps");
    return conn;
}

cpu_topology_info parse_cpu(json_value const& val) {
    cpu_topology_info cpu;
    if (!val.is_object())
        return cpu;
    cpu.numa_node = val.get_int("numa_node", -1);
    cpu.model_name = val.get_string("model_name");
    cpu.core_count = val.get_int("core_count");
    cpu.cpu_affinity_list = val.get_string("cpu_affinity_list");
    return cpu;
}

gpu_topology_info parse_gpu(json_value const& val) {
    gpu_topology_info gpu;
    if (!val.is_object())
        return gpu;

    gpu.id = val.get_uint("id");
    gpu.name = val.get_string("name");
    gpu.pci_bus_id = val.get_string("pci_bus_id");
    gpu.uuid = val.get_string("uuid");
    gpu.numa_node = val.get_int("numa_node", -1);

    // cpu_affinity can be either a nested object (cuCascade format) or a flat string
    auto const* aff = val.find("cpu_affinity");
    if (aff != nullptr && aff->is_object()) {
        gpu.cpu_affinity_list = aff->get_string("cpulist");
        auto const* cores = aff->find("cores");
        if (cores != nullptr)
            gpu.cpu_cores = parse_int_array(*cores);
    } else {
        gpu.cpu_affinity_list = val.get_string("cpu_affinity_list");
        auto const* cores = val.find("cpu_cores");
        if (cores != nullptr)
            gpu.cpu_cores = parse_int_array(*cores);
    }

    auto const* mb = val.find("memory_binding");
    if (mb != nullptr)
        gpu.memory_binding = parse_int_array(*mb);

    auto const* nd = val.find("network_devices");
    if (nd != nullptr)
        gpu.network_devices = parse_string_array(*nd);

    auto const* pci = val.find("pcie");
    if (pci != nullptr)
        gpu.pcie = parse_pcie(*pci);

    auto const* nvl = val.find("nvlink_peers");
    if (nvl != nullptr && nvl->is_array()) {
        for (auto const& elem : nvl->as_array()) {
            gpu.nvlink_peers.push_back(parse_nvlink(elem));
        }
    }

    return gpu;
}

network_device_info parse_network_device(json_value const& val) {
    network_device_info dev;
    if (!val.is_object())
        return dev;
    dev.name = val.get_string("name");
    dev.numa_node = val.get_int("numa_node", -1);
    dev.pci_bus_id = val.get_string("pci_bus_id");
    dev.model_name = val.get_string("model_name");
    dev.bandwidth_gbps = val.get_number("bandwidth_gbps");

    auto const* pci = val.find("pcie");
    if (pci != nullptr)
        dev.pcie = parse_pcie(*pci);

    return dev;
}

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------

void write_pcie(json_writer& w, pcie_info const& info) {
    w.begin_object();
    w.key("generation");
    w.value_int(info.generation);
    w.key("width");
    w.value_int(info.width);
    w.key("bandwidth_gbps");
    w.value_double(info.bandwidth_gbps);
    w.end_object();
}

void write_int_array(json_writer& w, std::vector<int> const& vec) {
    w.begin_array();
    for (std::size_t i = 0; i < vec.size(); ++i) {
        if (i > 0)
            w.array_element_sep();
        w.value_int(vec[i]);
    }
    w.end_array();
}

void write_string_array(json_writer& w, std::vector<std::string> const& vec) {
    w.begin_array();
    for (std::size_t i = 0; i < vec.size(); ++i) {
        if (i > 0)
            w.array_element_sep();
        w.value_string(vec[i]);
    }
    w.end_array();
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::string to_json(system_topology const& topo, int indent) {
    json_writer w{indent};
    w.begin_object();

    w.key("system");
    w.begin_object();
    w.key("hostname");
    w.value_string(topo.hostname);
    w.key("num_gpus");
    w.value_uint(topo.num_gpus);
    w.key("num_numa_nodes");
    w.value_int(topo.num_numa_nodes);
    w.key("num_network_devices");
    w.value_int(topo.num_network_devices);
    w.end_object();

    w.key("cpus");
    w.begin_array();
    for (std::size_t i = 0; i < topo.cpus.size(); ++i) {
        if (i > 0)
            w.array_element_sep();
        auto const& cpu = topo.cpus[i];
        w.begin_object();
        w.key("numa_node");
        w.value_int(cpu.numa_node);
        w.key("model_name");
        w.value_string(cpu.model_name);
        w.key("core_count");
        w.value_int(cpu.core_count);
        w.key("cpu_affinity_list");
        w.value_string(cpu.cpu_affinity_list);
        w.end_object();
    }
    w.end_array();

    w.key("gpus");
    w.begin_array();
    for (std::size_t i = 0; i < topo.gpus.size(); ++i) {
        if (i > 0)
            w.array_element_sep();
        auto const& gpu = topo.gpus[i];
        w.begin_object();
        w.key("id");
        w.value_uint(gpu.id);
        w.key("name");
        w.value_string(gpu.name);
        w.key("pci_bus_id");
        w.value_string(gpu.pci_bus_id);
        w.key("uuid");
        w.value_string(gpu.uuid);
        w.key("numa_node");
        w.value_int(gpu.numa_node);
        w.key("cpu_affinity");
        w.begin_object();
        w.key("cpulist");
        w.value_string(gpu.cpu_affinity_list);
        w.key("cores");
        write_int_array(w, gpu.cpu_cores);
        w.end_object();
        w.key("memory_binding");
        write_int_array(w, gpu.memory_binding);
        w.key("network_devices");
        write_string_array(w, gpu.network_devices);
        w.key("pcie");
        write_pcie(w, gpu.pcie);
        w.key("nvlink_peers");
        w.begin_array();
        for (std::size_t j = 0; j < gpu.nvlink_peers.size(); ++j) {
            if (j > 0)
                w.array_element_sep();
            auto const& nvl = gpu.nvlink_peers[j];
            w.begin_object();
            w.key("peer_gpu_id");
            w.value_uint(nvl.peer_gpu_id);
            w.key("link_count");
            w.value_int(nvl.link_count);
            w.key("nvlink_version");
            w.value_int(nvl.nvlink_version);
            w.key("bandwidth_gbps");
            w.value_double(nvl.bandwidth_gbps);
            w.end_object();
        }
        w.end_array();
        w.end_object();
    }
    w.end_array();

    w.key("network_devices");
    w.begin_array();
    for (std::size_t i = 0; i < topo.network_devices.size(); ++i) {
        if (i > 0)
            w.array_element_sep();
        auto const& dev = topo.network_devices[i];
        w.begin_object();
        w.key("name");
        w.value_string(dev.name);
        w.key("numa_node");
        w.value_int(dev.numa_node);
        w.key("pci_bus_id");
        w.value_string(dev.pci_bus_id);
        w.key("model_name");
        w.value_string(dev.model_name);
        w.key("bandwidth_gbps");
        w.value_double(dev.bandwidth_gbps);
        w.key("pcie");
        write_pcie(w, dev.pcie);
        w.end_object();
    }
    w.end_array();

    w.end_object();
    if (indent > 0) {
        return w.str() + "\n";
    }
    return w.str();
}

system_topology from_json(std::string const& json_str) {
    json_parser parser{json_str};
    json_value root = parser.parse();

    if (!root.is_object()) {
        throw std::runtime_error("JSON: top-level value must be an object");
    }

    system_topology topo;

    auto const* sys = root.find("system");
    if (sys != nullptr && sys->is_object()) {
        topo.hostname = sys->get_string("hostname");
        topo.num_gpus = sys->get_uint("num_gpus");
        topo.num_numa_nodes = sys->get_int("num_numa_nodes");
        topo.num_network_devices = sys->get_int("num_network_devices");
    }

    auto const* cpus = root.find("cpus");
    if (cpus != nullptr && cpus->is_array()) {
        for (auto const& elem : cpus->as_array()) {
            topo.cpus.push_back(parse_cpu(elem));
        }
    }

    auto const* gpus = root.find("gpus");
    if (gpus != nullptr && gpus->is_array()) {
        for (auto const& elem : gpus->as_array()) {
            topo.gpus.push_back(parse_gpu(elem));
        }
    }

    auto const* nds = root.find("network_devices");
    if (nds != nullptr && nds->is_array()) {
        for (auto const& elem : nds->as_array()) {
            topo.network_devices.push_back(parse_network_device(elem));
        }
    }

    return topo;
}

system_topology from_json_file(std::string const& path) {
    std::ifstream file{path};
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open JSON file: " + path);
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    return from_json(ss.str());
}

}  // namespace rapidsmpf::topology
