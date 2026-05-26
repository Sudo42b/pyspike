#!/usr/bin/env python3
"""pyspike_rpc_server.py — ggml-rpc protocol server backed by pyspike.

Implements ggml RPC proto v4.0.0 over TCP. Control messages (ALLOC/FREE/
SET/GET tensor, buffer accounting, HELLO handshake, device queries) are
fully wired here. GRAPH_COMPUTE is currently a stub that ACKs with success
and leaves output buffers untouched — real per-op dispatch to pyspike ELFs
lands in M2.

Wire reference: github.com/ggml-org/ggml src/ggml-rpc/ggml-rpc.cpp
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import socketserver
import struct
import sys
import threading
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import wire_protocol as wp  # noqa: E402
import op_registry as opr   # noqa: E402
import ddr_backend as ddr   # noqa: E402
import ggml_compute as gc   # noqa: E402

log = logging.getLogger("pyspike-rpc")


# ---------------------------------------------------------------------------
# Device memory model
# ---------------------------------------------------------------------------

DEFAULT_ALIGNMENT = 64
DEFAULT_MAX_SIZE = 1 << 32          # 4 GiB per buffer
DEFAULT_DEVICE_MEMORY = 8 << 30     # 8 GiB virtual device memory


@dataclass
class Buffer:
    """A device buffer. `data` is the host-side bytearray that shadows it."""
    remote_ptr: int
    size: int
    data: bytearray


class DeviceMemory:
    """Tracks per-device buffers. `remote_ptr` values are opaque handles —
    we just hand out monotonically increasing integers. The client never
    dereferences them on its side; it only sends them back to us.
    """

    def __init__(self, total: int = DEFAULT_DEVICE_MEMORY):
        self.total = total
        self.allocated = 0
        self.buffers: dict[int, Buffer] = {}
        self._lock = threading.Lock()
        self._next_ptr = 0x10_0000   # start above reserved low area

    def alloc(self, size: int) -> Buffer:
        size = max(size, 1)
        with self._lock:
            ptr = self._next_ptr
            self._next_ptr += max(size, DEFAULT_ALIGNMENT)
            # 8-byte align the next ptr to keep things tidy
            self._next_ptr = (self._next_ptr + 7) & ~7
            buf = Buffer(remote_ptr=ptr, size=size, data=bytearray(size))
            self.buffers[ptr] = buf
            self.allocated += size
        log.debug("alloc ptr=0x%x size=%d", ptr, size)
        return buf

    def free(self, ptr: int) -> None:
        with self._lock:
            buf = self.buffers.pop(ptr, None)
        if buf is None:
            log.warning("free of unknown ptr 0x%x", ptr)
            return
        self.allocated -= buf.size
        log.debug("free ptr=0x%x size=%d", ptr, buf.size)

    def get(self, ptr: int) -> Buffer:
        with self._lock:
            buf = self.buffers.get(ptr)
        if buf is None:
            raise KeyError(f"unknown remote_ptr 0x{ptr:x}")
        return buf

    def buffer_for_data_addr(self, data_addr: int) -> tuple[Buffer, int]:
        """Resolve a tensor `data` field to (buffer, offset_within_buffer).

        The client sets tensor.data = buffer.remote_ptr + offset. We linearly
        scan because the buffer table is small (<1k entries typical).
        """
        with self._lock:
            for ptr, buf in self.buffers.items():
                if ptr <= data_addr < ptr + buf.size:
                    return buf, data_addr - ptr
        raise KeyError(f"data addr 0x{data_addr:x} not within any buffer")

    def free_bytes(self) -> int:
        return max(self.total - self.allocated, 0)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class RpcSession:
    """Per-connection handler. ggml-rpc keeps one socket per remote backend
    instance for the lifetime of that backend, so each thread owns its own
    device-tensor scratch state (e.g. SET_TENSOR_HASH cache).
    """

    def __init__(self, server: "PyspikeRpcServer", conn: socket.socket, addr):
        self.server = server
        self.conn = conn
        self.addr = addr
        self.hash_cache: dict[int, bytes] = {}   # M2: persistent hash cache

    # --- dispatch table ----------------------------------------------------

    def serve(self) -> None:
        try:
            self._expect_hello()
            while True:
                cmd, payload = wp.recv_request(self.conn)
                self._dispatch(cmd, payload)
        except ConnectionError as e:
            log.info("[%s] client closed (%s)", self.addr, e)
        except Exception:
            log.exception("[%s] handler error", self.addr)
        finally:
            self.conn.close()

    def _expect_hello(self) -> None:
        cmd, payload = wp.recv_request(self.conn)
        if cmd != wp.RpcCmd.HELLO:
            raise ConnectionError(f"first cmd must be HELLO, got {cmd!r}")
        _client_caps = wp.unpack_hello_req(payload)
        wp.send_response(self.conn, wp.pack_hello_rsp())
        log.info("[%s] HELLO ok (proto v%d.%d.%d)", self.addr,
                 wp.RPC_PROTO_MAJOR_VERSION, wp.RPC_PROTO_MINOR_VERSION,
                 wp.RPC_PROTO_PATCH_VERSION)

    _HANDLERS = {}  # populated below

    def _dispatch(self, cmd: wp.RpcCmd, payload: bytes) -> None:
        handler = self._HANDLERS.get(cmd)
        if handler is None:
            log.warning("[%s] unhandled cmd %s (len=%d)", self.addr, cmd, len(payload))
            # one-way commands must not write a response; ignoring is safe
            if cmd not in wp.ONE_WAY:
                wp.send_response(self.conn, b"")
            return
        handler(self, payload)

    # --- handlers ----------------------------------------------------------

    def _h_device_count(self, _payload: bytes) -> None:
        wp.send_response(self.conn, wp.pack_device_count_rsp(len(self.server.devices)))

    def _h_get_device_memory(self, payload: bytes) -> None:
        device = wp.unpack_get_device_memory_req(payload)
        dev = self.server.devices[device]
        wp.send_response(self.conn, wp.pack_get_device_memory_rsp(
            dev.free_bytes(), dev.total))

    def _h_get_alignment(self, _payload: bytes) -> None:
        wp.send_response(self.conn, wp.pack_get_alignment_rsp(DEFAULT_ALIGNMENT))

    def _h_get_max_size(self, _payload: bytes) -> None:
        wp.send_response(self.conn, wp.pack_get_max_size_rsp(DEFAULT_MAX_SIZE))

    def _h_alloc_buffer(self, payload: bytes) -> None:
        device, size = wp.unpack_alloc_buffer_req(payload)
        dev = self.server.devices[device]
        buf = dev.alloc(size)
        wp.send_response(self.conn, wp.pack_alloc_buffer_rsp(buf.remote_ptr, buf.size))

    def _h_free_buffer(self, payload: bytes) -> None:
        ptr = wp.unpack_free_buffer_req(payload)
        self.server.devices[0].free(ptr)
        wp.send_response(self.conn, b"")

    def _h_buffer_get_base(self, payload: bytes) -> None:
        ptr = wp.unpack_buffer_get_base_req(payload)
        buf = self.server.devices[0].get(ptr)
        # The client uses base_ptr to compute tensor.data offsets — we return
        # the same remote_ptr so tensor.data == buffer + offset works out.
        wp.send_response(self.conn, wp.pack_buffer_get_base_rsp(buf.remote_ptr))

    def _h_buffer_clear(self, payload: bytes) -> None:
        ptr, value = wp.unpack_buffer_clear_req(payload)
        buf = self.server.devices[0].get(ptr)
        for i in range(len(buf.data)):
            buf.data[i] = value
        wp.send_response(self.conn, b"")

    def _h_set_tensor(self, payload: bytes) -> None:
        # One-way: no response.
        tensor, offset, data = wp.unpack_set_tensor_msg(payload)
        buf, base_off = self.server.devices[0].buffer_for_data_addr(tensor.data)
        start = base_off + offset
        end = start + len(data)
        if end > len(buf.data):
            log.error("[%s] SET_TENSOR overflow: end=%d > buf.size=%d",
                      self.addr, end, len(buf.data))
            return
        buf.data[start:end] = data
        log.debug("[%s] SET_TENSOR id=0x%x off=%d bytes=%d",
                  self.addr, tensor.id, offset, len(data))

    def _h_set_tensor_hash(self, payload: bytes) -> None:
        tensor, offset, hash_ = wp.unpack_set_tensor_hash_req(payload)
        data = self.hash_cache.get(hash_)
        if data is None:
            wp.send_response(self.conn, wp.pack_set_tensor_hash_rsp(0))  # miss
            return
        buf, base_off = self.server.devices[0].buffer_for_data_addr(tensor.data)
        start = base_off + offset
        buf.data[start:start + len(data)] = data
        wp.send_response(self.conn, wp.pack_set_tensor_hash_rsp(1))      # hit

    def _h_get_tensor(self, payload: bytes) -> None:
        tensor, offset, size = wp.unpack_get_tensor_req(payload)
        buf, base_off = self.server.devices[0].buffer_for_data_addr(tensor.data)
        start = base_off + offset
        end = start + size
        if end > len(buf.data):
            log.error("[%s] GET_TENSOR overflow: end=%d > buf.size=%d",
                      self.addr, end, len(buf.data))
            wp.send_response(self.conn, b"")
            return
        wp.send_response(self.conn, bytes(buf.data[start:end]))

    def _h_copy_tensor(self, payload: bytes) -> None:
        src, dst = wp.unpack_copy_tensor_req(payload)
        src_buf, src_off = self.server.devices[0].buffer_for_data_addr(src.data)
        dst_buf, dst_off = self.server.devices[0].buffer_for_data_addr(dst.data)
        # Copy min(src_size, dst_size_remaining) — server has full nb/ne so we
        # could compute exact bytes but client guarantees layouts match.
        nbytes = min(len(src_buf.data) - src_off, len(dst_buf.data) - dst_off)
        dst_buf.data[dst_off:dst_off + nbytes] = src_buf.data[src_off:src_off + nbytes]
        wp.send_response(self.conn, wp.pack_copy_tensor_rsp(1))

    def _h_init_tensor(self, _payload: bytes) -> None:
        # No-op for our backend — we initialise lazily on SET_TENSOR.
        wp.send_response(self.conn, b"")

    def _h_get_alloc_size(self, payload: bytes) -> None:
        _device, tensor, _srcs = wp.unpack_get_alloc_size_req(payload)
        # ggml stores tensors contiguously; size = nb[3] * ne[3] if available
        # else nb[ndim-1] * ne[ndim-1]. We fall back to product of ne * dtype size.
        # For now, conservative estimate = nb[max_dim] * ne[max_dim].
        nbytes = 0
        for d in range(wp.GGML_MAX_DIMS - 1, -1, -1):
            if tensor.ne[d] > 0 and tensor.nb[d] > 0:
                nbytes = tensor.ne[d] * tensor.nb[d]
                break
        nbytes = max(nbytes, 1)
        # Align to DEFAULT_ALIGNMENT
        nbytes = (nbytes + DEFAULT_ALIGNMENT - 1) & ~(DEFAULT_ALIGNMENT - 1)
        wp.send_response(self.conn, wp.pack_get_alloc_size_rsp(nbytes))

    def _h_graph_compute(self, payload: bytes) -> None:
        """M2 dispatcher: walk the graph in topo order (the `nodes` list IS
        topo order), compute each node via NumPy backend, write results back
        to device buffers. View-only ops (RESHAPE/VIEW/PERMUTE/...) are no-ops
        because the dst tensor already aliases the source buffer.

        One-way: no response. Failures are logged but don't kill the session.
        """
        device, node_ids, tensors = wp.unpack_graph_compute_msg(payload)
        by_id = {t.id: t for t in tensors}
        dev = self.server.devices[device]

        ok = 0
        nop = 0
        unsupported = 0
        errored = 0

        for node_id in node_ids:
            node = by_id.get(node_id)
            if node is None:
                log.warning("[%s] graph node id=0x%x missing from tensor table",
                            self.addr, node_id)
                errored += 1
                continue

            try:
                srcs = self._resolve_srcs(node, by_id, dev)
                result = gc.compute(node, srcs)
                if result is None:
                    nop += 1
                    continue
                self._write_result(node, result, dev)
                ok += 1
            except NotImplementedError as e:
                log.warning("[%s] node 0x%x op=%d unsupported: %s",
                            self.addr, node_id, node.op, e)
                unsupported += 1
            except Exception:
                log.exception("[%s] node 0x%x op=%d failed",
                              self.addr, node_id, node.op)
                errored += 1

        log.info("[%s] GRAPH_COMPUTE device=%d nodes=%d "
                 "ok=%d nop=%d unsupported=%d err=%d",
                 self.addr, device, len(node_ids),
                 ok, nop, unsupported, errored)

    def _resolve_srcs(self, node: wp.RpcTensor, by_id: dict, dev) -> list:
        """Build numpy views for each non-zero src tensor of `node`."""
        import numpy as np
        srcs = []
        for src_id in node.src:
            if src_id == 0:
                break
            src_t = by_id.get(src_id)
            if src_t is None:
                raise KeyError(f"src id 0x{src_id:x} not in tensor table")
            src_buf, src_off = dev.buffer_for_data_addr(src_t.data)
            srcs.append(ddr.view_tensor(src_t, src_buf.data, src_off))
        return srcs

    def _write_result(self, node: wp.RpcTensor, result, dev) -> None:
        """Place compute output into the destination tensor's DDR slot."""
        dst_buf, dst_off = dev.buffer_for_data_addr(node.data)
        ddr.write_tensor(node, dst_buf.data, dst_off, result)

    def _h_graph_recompute(self, _payload: bytes) -> None:
        # One-way — same stub as GRAPH_COMPUTE for M1.
        log.debug("[%s] GRAPH_RECOMPUTE (stub)", self.addr)


# Build the dispatch table on the class (after methods are defined).
RpcSession._HANDLERS = {
    wp.RpcCmd.DEVICE_COUNT:      RpcSession._h_device_count,
    wp.RpcCmd.GET_DEVICE_MEMORY: RpcSession._h_get_device_memory,
    wp.RpcCmd.GET_ALIGNMENT:     RpcSession._h_get_alignment,
    wp.RpcCmd.GET_MAX_SIZE:      RpcSession._h_get_max_size,
    wp.RpcCmd.ALLOC_BUFFER:      RpcSession._h_alloc_buffer,
    wp.RpcCmd.FREE_BUFFER:       RpcSession._h_free_buffer,
    wp.RpcCmd.BUFFER_GET_BASE:   RpcSession._h_buffer_get_base,
    wp.RpcCmd.BUFFER_CLEAR:      RpcSession._h_buffer_clear,
    wp.RpcCmd.SET_TENSOR:        RpcSession._h_set_tensor,
    wp.RpcCmd.SET_TENSOR_HASH:   RpcSession._h_set_tensor_hash,
    wp.RpcCmd.GET_TENSOR:        RpcSession._h_get_tensor,
    wp.RpcCmd.COPY_TENSOR:       RpcSession._h_copy_tensor,
    wp.RpcCmd.INIT_TENSOR:       RpcSession._h_init_tensor,
    wp.RpcCmd.GET_ALLOC_SIZE:    RpcSession._h_get_alloc_size,
    wp.RpcCmd.GRAPH_COMPUTE:     RpcSession._h_graph_compute,
    wp.RpcCmd.GRAPH_RECOMPUTE:   RpcSession._h_graph_recompute,
}


# ---------------------------------------------------------------------------

class _ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        RpcSession(self.server.pyspike_server, self.request, self.client_address).serve()


class PyspikeRpcServer:
    def __init__(self, host: str, port: int,
                 device_memory: int = DEFAULT_DEVICE_MEMORY,
                 n_devices: int = 1):
        self.host = host
        self.port = port
        self.devices = [DeviceMemory(device_memory) for _ in range(n_devices)]
        self._tcp: _ThreadedTCPServer | None = None

    def serve_forever(self) -> None:
        self._tcp = _ThreadedTCPServer((self.host, self.port), _Handler)
        self._tcp.pyspike_server = self
        log.info("listening on %s:%d (devices=%d, mem=%.1f GiB each)",
                 self.host, self.port, len(self.devices),
                 self.devices[0].total / (1 << 30))
        self._tcp.serve_forever()

    def shutdown(self) -> None:
        if self._tcp is not None:
            self._tcp.shutdown()
            self._tcp.server_close()


# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=50052)
    p.add_argument("--device-memory", type=int, default=DEFAULT_DEVICE_MEMORY,
                   help="Bytes of advertised device memory (default 8 GiB)")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)-5s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    srv = PyspikeRpcServer(args.host, args.port, device_memory=args.device_memory)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log.info("interrupted, shutting down")
        srv.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
