# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool as bool_t
from libcpp.string cimport string


cdef class TopologyViz:
    """Python wrapper around the C++ ``rapidsmpf::topology::topology_viz`` class.

    Provides direct access to topology discovery, JSON I/O, and enrichment
    without spawning subprocesses.

    Examples
    --------
    >>> viz = TopologyViz()
    >>> viz.discover()
    >>> topo = viz.to_dict()
    >>> viz.to_json()  # enriched JSON string
    """

    def discover(self) -> bool:
        """Discover the full enriched topology of the local system.

        Runs cuCascade base discovery followed by bandwidth and naming
        enrichment (PCIe, NVLink, NIC speed, CPU model names).

        Returns
        -------
        bool
            ``True`` on success, ``False`` if base discovery fails.
            Partial enrichment failures are not fatal.
        """
        cdef bool_t ok
        with nogil:
            ok = self._handle.discover()
        return ok

    def load_json(self, str json_str not None) -> bool:
        """Load topology from a JSON string.

        Accepts both the enriched format and the original cuCascade
        ``topology_discovery`` format.

        Parameters
        ----------
        json_str
            A UTF-8 JSON string.

        Returns
        -------
        bool
            ``True`` on success, ``False`` if parsing fails.
        """
        cdef string s = json_str.encode("utf-8")
        cdef bool_t ok
        with nogil:
            ok = self._handle.load_json(s)
        return ok

    def load_json_file(self, str path not None) -> bool:
        """Load topology from a JSON file.

        Parameters
        ----------
        path
            Filesystem path to a JSON file.

        Returns
        -------
        bool
            ``True`` on success, ``False`` if the file cannot be opened
            or parsed.
        """
        cdef string p = path.encode("utf-8")
        cdef bool_t ok
        with nogil:
            ok = self._handle.load_json_file(p)
        return ok

    def enrich(self) -> bool:
        """Re-enrich an already-loaded topology with live system data.

        Fills in zero / empty bandwidth and naming fields by querying
        sysfs and NVML.

        Returns
        -------
        bool
            ``True`` if enrichment ran, ``False`` if no topology is loaded.
        """
        cdef bool_t ok
        with nogil:
            ok = self._handle.enrich()
        return ok

    @property
    def is_ready(self) -> bool:
        """Whether a topology has been loaded or discovered."""
        return self._handle.is_ready()

    def to_json(self, int indent = 2) -> str:
        """Serialize the stored topology to a JSON string.

        Parameters
        ----------
        indent
            Spaces per indentation level (default 2).  Use 0 for compact
            output.

        Returns
        -------
        str
            A UTF-8 JSON string.

        Raises
        ------
        RuntimeError
            If no topology has been loaded or discovered.
        """
        cdef string s
        with nogil:
            s = self._handle.to_json(indent)
        return s.decode("utf-8")

    def to_dict(self) -> dict:
        """Return the stored topology as a Python dictionary.

        Serializes to JSON internally and parses back to a dict.  This
        is the bridge between the C++ data structures and the Python
        rendering API.

        Returns
        -------
        dict
            Nested dictionary matching the ``system_topology`` JSON schema.

        Raises
        ------
        RuntimeError
            If no topology has been loaded or discovered.
        """
        import json
        return json.loads(self.to_json(indent=0))
