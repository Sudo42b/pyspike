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
import pyspike_runner as psr  # noqa: E402

# Opt-in: when set, GRAPH_COMPUTE tries to route compatible nodes through
# pyspike .elf execution; otherwise everything falls back to NumPy.
USE_PYSPIKE = os.environ.get("PYSPIKE_RPC_BACKEND", "").lower() in ("spike", "pyspike", "1", "true")

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
                # Opt-in pyspike route first; falls through to NumPy on miss.
                if USE_PYSPIKE and self._try_pyspike(node, by_id, dev):
                    ok += 1
                    continue
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

    # sub-op id → (kernel name for log, pyspike runner fn). Extend by adding
    # a firmware template under firmware_templates/ + a runner in pyspike_runner.
    # All entries below resolve to a single shared unary_intrin1.c.tpl skeleton
    # with the op's intrinsic call swapped in (see psr._UNARY_INTRIN1_CALLS).
    _UNARY_PYSPIKE_KERNELS = {
        opr.GGML_UNARY_OP_ABS:      ("ABS",      psr.SUPPORTED_PYSPIKE_OPS["unary_abs_fp16"]),
        opr.GGML_UNARY_OP_NEG:      ("NEG",      psr.SUPPORTED_PYSPIKE_OPS["unary_neg_fp16"]),
        opr.GGML_UNARY_OP_SGN:      ("SGN",      psr.SUPPORTED_PYSPIKE_OPS["unary_sgn_fp16"]),
        opr.GGML_UNARY_OP_STEP:     ("STEP",     psr.SUPPORTED_PYSPIKE_OPS["unary_step_fp16"]),
        opr.GGML_UNARY_OP_RELU:     ("RELU",     psr.SUPPORTED_PYSPIKE_OPS["unary_relu_fp16"]),
        opr.GGML_UNARY_OP_SIGMOID:  ("SIGMOID",  psr.SUPPORTED_PYSPIKE_OPS["unary_sigmoid_fp16"]),
        opr.GGML_UNARY_OP_TANH:     ("TANH",     psr.SUPPORTED_PYSPIKE_OPS["unary_tanh_fp16"]),
        opr.GGML_UNARY_OP_GELU:     ("GELU",     psr.SUPPORTED_PYSPIKE_OPS["unary_gelu_fp16"]),
        opr.GGML_UNARY_OP_GELU_ERF: ("GELU_ERF", psr.SUPPORTED_PYSPIKE_OPS["unary_gelu_erf_fp16"]),
        opr.GGML_UNARY_OP_EXP:      ("EXP",      psr.SUPPORTED_PYSPIKE_OPS["unary_exp_fp16"]),
        opr.GGML_UNARY_OP_SILU:     ("SILU",     psr.SUPPORTED_PYSPIKE_OPS["unary_silu_fp16"]),
        opr.GGML_UNARY_OP_FLOOR:    ("FLOOR",    psr.SUPPORTED_PYSPIKE_OPS["unary_floor_fp16"]),
        opr.GGML_UNARY_OP_TRUNC:    ("TRUNC",    psr.SUPPORTED_PYSPIKE_OPS["unary_trunc_fp16"]),
    }
    # main op (not under GGML_OP_UNARY) → pyspike runner fn. 'kind' is 'unary'
    # for 1-input runners or 'binary' for 2-input runners; the dispatcher
    # below resolves srcs accordingly.
    _MAIN_OP_PYSPIKE_KERNELS = {
        opr.GGML_OP_SQRT: ("SQRT", "unary",  psr.SUPPORTED_PYSPIKE_OPS["unary_sqrt_fp16"]),
        opr.GGML_OP_ADD:  ("ADD",  "binary", psr.SUPPORTED_PYSPIKE_OPS["binary_add_fp16"]),
        opr.GGML_OP_SUB:  ("SUB",  "binary", psr.SUPPORTED_PYSPIKE_OPS["binary_sub_fp16"]),
        opr.GGML_OP_MUL:  ("MUL",  "binary", psr.SUPPORTED_PYSPIKE_OPS["binary_mul_fp16"]),
        opr.GGML_OP_DIV:  ("DIV",  "binary", psr.SUPPORTED_PYSPIKE_OPS["binary_div_fp16"]),
        opr.GGML_OP_ACC:  ("ACC",  "binary", psr.SUPPORTED_PYSPIKE_OPS["binary_acc_fp16"]),
    }

    # main op (no GGML_OP_UNARY sub-id) → custom dispatcher methods that need
    # shape + op_params handling that doesn't fit the elementwise pattern.
    # Order matters in `_try_pyspike` — checked before the generic table.
    _SIMPLE_UNARY_DISPATCH = {
        opr.GGML_OP_SQR,
        opr.GGML_OP_SUM_ROWS,
        opr.GGML_OP_GROUP_NORM,
        opr.GGML_OP_NORM,
        opr.GGML_OP_SCALE,
        opr.GGML_OP_CLAMP,
        opr.GGML_OP_MEAN,
        opr.GGML_OP_SUM,
        opr.GGML_OP_TRI,
    }

    # GGML_OP_UNARY sub-op ids that route through the shape-parameterised
    # simple-unary path instead of `_UNARY_PYSPIKE_KERNELS`. These need
    # WIDTH/HEIGHT from the source tensor's ne rather than the unary_intrin1
    # 8-wide row pattern.
    _SIMPLE_UNARY_SUBOP_KERNELS = {
        opr.GGML_UNARY_OP_CEIL:  ("CEIL",  "ceil_fp16"),
        opr.GGML_UNARY_OP_EXPM1: ("EXPM1", "expm1_fp16"),
    }

    def _try_simple_unary_pyspike(self, node: wp.RpcTensor, by_id: dict, dev) -> bool:
        """Dispatch SQR/SUM_ROWS/GROUP_NORM/NORM/SCALE through the shape-
        parameterised template runners. Each kernel takes the input tensor's
        innermost dim as WIDTH and packs all remaining elements into HEIGHT.
        Returns False on any guard miss → caller falls back to NumPy.
        """
        import numpy as np
        if node.type != ddr.GGML_TYPE_F16:
            return False
        src_id = node.src[0]
        if src_id == 0:
            return False
        src_t = by_id.get(src_id)
        if src_t is None or src_t.type != ddr.GGML_TYPE_F16:
            return False
        width = src_t.ne[0]
        height = src_t.ne[1] * src_t.ne[2] * src_t.ne[3]
        total = width * height
        if width <= 0 or total <= 0:
            return False

        sb, so = dev.buffer_for_data_addr(src_t.data)
        nbytes = ddr.tensor_nbytes(src_t)
        src_bytes = bytes(sb.data[so:so + nbytes])

        op = node.op
        try:
            if op == opr.GGML_OP_SQR:
                kernel_name = "SQR"
                runner = psr.SUPPORTED_PYSPIKE_OPS["sqr_fp16"]
                out_bytes = runner(src_bytes, width)
            elif op == opr.GGML_OP_SUM_ROWS:
                kernel_name = "SUM_ROWS"
                runner = psr.SUPPORTED_PYSPIKE_OPS["sum_rows_fp16"]
                out_bytes = runner(src_bytes, width)
            elif op == opr.GGML_OP_NORM:
                eps = np.frombuffer(
                    np.array(node.op_params[0], dtype=np.int32).tobytes(),
                    dtype=np.float32)[0]
                kernel_name = "NORM"
                runner = psr.SUPPORTED_PYSPIKE_OPS["norm_fp16"]
                out_bytes = runner(src_bytes, width, float(eps))
            elif op == opr.GGML_OP_GROUP_NORM:
                n_groups = int(node.op_params[0])
                if n_groups != 1:
                    return False  # firmware kernel only handles num_groups=1
                eps = np.frombuffer(
                    np.array(node.op_params[1], dtype=np.int32).tobytes(),
                    dtype=np.float32)[0]
                kernel_name = "GROUP_NORM"
                runner = psr.SUPPORTED_PYSPIKE_OPS["group_norm_fp16"]
                out_bytes = runner(src_bytes, width, float(eps))
            elif op == opr.GGML_OP_SCALE:
                scale = np.frombuffer(
                    np.array(node.op_params[0], dtype=np.int32).tobytes(),
                    dtype=np.float32)[0]
                bias = np.frombuffer(
                    np.array(node.op_params[1], dtype=np.int32).tobytes(),
                    dtype=np.float32)[0]
                if bias != 0.0:
                    return False  # NumPy fallback applies the bias term
                # SCALE firmware loops over 16-elem SVR chunks distributed
                # across 16 SPUs — under ~256 elements some SPUs would skip
                # the final store_cr and hang the credit wait.
                if total < 256:
                    return False
                kernel_name = "SCALE"
                runner = psr.SUPPORTED_PYSPIKE_OPS["scale_fp16"]
                out_bytes = runner(src_bytes, width, float(scale))
            elif op == opr.GGML_OP_CLAMP:
                min_v = np.frombuffer(
                    np.array(node.op_params[0], dtype=np.int32).tobytes(),
                    dtype=np.float32)[0]
                max_v = np.frombuffer(
                    np.array(node.op_params[1], dtype=np.int32).tobytes(),
                    dtype=np.float32)[0]
                kernel_name = "CLAMP"
                runner = psr.SUPPORTED_PYSPIKE_OPS["clamp_fp16"]
                out_bytes = runner(src_bytes, width,
                                   float(min_v), float(max_v))
            elif op == opr.GGML_OP_MEAN:
                # firmware splits HEIGHT evenly across 16 SPUs — needs % 16 == 0
                if height % 16 != 0:
                    return False
                kernel_name = "MEAN"
                runner = psr.SUPPORTED_PYSPIKE_OPS["mean_fp16"]
                out_bytes = runner(src_bytes, width)
            elif op == opr.GGML_OP_SUM:
                kernel_name = "SUM"
                runner = psr.SUPPORTED_PYSPIKE_OPS["sum_fp16"]
                out_bytes = runner(src_bytes, width)
            elif op == opr.GGML_OP_TRI:
                tri_type = int(node.op_params[0])
                if tri_type < 0 or tri_type > 3:
                    return False
                kernel_name = "TRI"
                runner = psr.SUPPORTED_PYSPIKE_OPS["tri_fp16"]
                out_bytes = runner(src_bytes, width, tri_type)
            else:
                return False
            log.info("[%s] node 0x%x %s pyspike route: W=%d H=%d (%d fp16)",
                     self.addr, node.id, kernel_name, width, height, total)
        except Exception as e:
            log.warning("[%s] simple-unary pyspike failed (%s) → NumPy fallback",
                        self.addr, e)
            return False

        dst_buf, dst_off = dev.buffer_for_data_addr(node.data)
        dst_buf.data[dst_off:dst_off + len(out_bytes)] = out_bytes
        return True

    def _try_pad_pyspike(self, node: wp.RpcTensor, by_id: dict, dev) -> bool:
        """PAD: append zeros along right (ne[0]) and bottom (ne[1]). Kernel
        does not support channel/batch padding nor left/top — those go to
        NumPy fallback."""
        if node.type != ddr.GGML_TYPE_F16:
            return False
        src_id = node.src[0]
        if src_id == 0:
            return False
        src_t = by_id.get(src_id)
        if src_t is None or src_t.type != ddr.GGML_TYPE_F16:
            return False
        # ggml PAD op_params: [pad_after_dim0, pad_after_dim1, pad_after_dim2, pad_after_dim3]
        pad_w = int(node.op_params[0])
        pad_h = int(node.op_params[1])
        pad_c = int(node.op_params[2])
        pad_b = int(node.op_params[3])
        if pad_c != 0 or pad_b != 0:
            return False
        if src_t.ne[2] != 1 or src_t.ne[3] != 1:
            return False
        src_cols = src_t.ne[0]
        src_rows = src_t.ne[1]
        if src_cols <= 0 or src_rows <= 0:
            return False

        sb, so = dev.buffer_for_data_addr(src_t.data)
        nbytes = ddr.tensor_nbytes(src_t)
        src_bytes = bytes(sb.data[so:so + nbytes])
        try:
            log.info("[%s] node 0x%x PAD pyspike route: src=(%d,%d) pad=(R=%d,B=%d)",
                     self.addr, node.id, src_rows, src_cols, pad_w, pad_h)
            out_bytes = psr.SUPPORTED_PYSPIKE_OPS["pad_fp16"](
                src_bytes, src_rows, src_cols, pad_w, pad_h)
        except Exception as e:
            log.warning("[%s] pad pyspike failed (%s) → NumPy fallback",
                        self.addr, e)
            return False
        dst_buf, dst_off = dev.buffer_for_data_addr(node.data)
        dst_buf.data[dst_off:dst_off + len(out_bytes)] = out_bytes
        return True

    def _try_concat_pyspike(self, node: wp.RpcTensor, by_id: dict, dev) -> bool:
        """CONCAT axis=0 (innermost dim) with equal src col counts. Other
        axes / unequal cols go to NumPy fallback."""
        if node.type != ddr.GGML_TYPE_F16:
            return False
        axis = int(node.op_params[0])
        if axis != 0:
            return False
        src0_id = node.src[0]
        src1_id = node.src[1]
        if src0_id == 0 or src1_id == 0:
            return False
        s0 = by_id.get(src0_id)
        s1 = by_id.get(src1_id)
        if s0 is None or s1 is None:
            return False
        if s0.type != ddr.GGML_TYPE_F16 or s1.type != ddr.GGML_TYPE_F16:
            return False
        if s0.ne[2] != 1 or s0.ne[3] != 1 or s1.ne[2] != 1 or s1.ne[3] != 1:
            return False
        # Kernel assumes equal SRC_COLS for both sides.
        if s0.ne[0] != s1.ne[0]:
            return False
        if s0.ne[1] != s1.ne[1]:
            return False
        cols = s0.ne[0]
        rows = s0.ne[1]
        if cols <= 0 or rows <= 0:
            return False

        b0, o0 = dev.buffer_for_data_addr(s0.data)
        b1, o1 = dev.buffer_for_data_addr(s1.data)
        n0 = ddr.tensor_nbytes(s0)
        n1 = ddr.tensor_nbytes(s1)
        src0_bytes = bytes(b0.data[o0:o0 + n0])
        src1_bytes = bytes(b1.data[o1:o1 + n1])
        try:
            log.info("[%s] node 0x%x CONCAT pyspike route: 2x(%d,%d) axis=0",
                     self.addr, node.id, rows, cols)
            out_bytes = psr.SUPPORTED_PYSPIKE_OPS["concat_fp16"](
                src0_bytes, src1_bytes, cols, cols, rows)
        except Exception as e:
            log.warning("[%s] concat pyspike failed (%s) → NumPy fallback",
                        self.addr, e)
            return False
        dst_buf, dst_off = dev.buffer_for_data_addr(node.data)
        dst_buf.data[dst_off:dst_off + len(out_bytes)] = out_bytes
        return True

    def _try_pool_2d_pyspike(self, node: wp.RpcTensor, by_id: dict, dev) -> bool:
        """POOL_2D AVG via __pool_a, MAX via __pool_m. Padding=0 only."""
        if node.type != ddr.GGML_TYPE_F16:
            return False
        pool_op = int(node.op_params[0])
        if pool_op not in (0, 1):                    # 0=MAX, 1=AVG
            return False
        k0, k1 = int(node.op_params[1]), int(node.op_params[2])
        s0, s1 = int(node.op_params[3]), int(node.op_params[4])
        p0, p1 = int(node.op_params[5]), int(node.op_params[6])
        if p0 != 0 or p1 != 0:
            return False
        src_id = node.src[0]
        if src_id == 0:
            return False
        src_t = by_id.get(src_id)
        if src_t is None or src_t.type != ddr.GGML_TYPE_F16:
            return False
        if src_t.ne[2] != 1 or src_t.ne[3] != 1:
            return False
        in_w, in_h = src_t.ne[0], src_t.ne[1]
        if in_w <= 0 or in_h <= 0:
            return False

        sb, so = dev.buffer_for_data_addr(src_t.data)
        nbytes = ddr.tensor_nbytes(src_t)
        src_bytes = bytes(sb.data[so:so + nbytes])
        runner_key = "pool_2d_avg_fp16" if pool_op == 1 else "pool_2d_max_fp16"
        kernel_name = "POOL_2D_AVG" if pool_op == 1 else "POOL_2D_MAX"
        try:
            log.info("[%s] node 0x%x %s pyspike route: in=(%d,%d) k=(%d,%d) s=(%d,%d)",
                     self.addr, node.id, kernel_name, in_h, in_w, k1, k0, s1, s0)
            out_bytes = psr.SUPPORTED_PYSPIKE_OPS[runner_key](
                src_bytes, in_h, in_w, k1, k0, s1, s0)
        except Exception as e:
            log.warning("[%s] pool_2d pyspike failed (%s) → NumPy fallback",
                        self.addr, e)
            return False
        dst_buf, dst_off = dev.buffer_for_data_addr(node.data)
        dst_buf.data[dst_off:dst_off + len(out_bytes)] = out_bytes
        return True

    def _try_conv_2d_pyspike(self, node: wp.RpcTensor, by_id: dict, dev) -> bool:
        """CONV_2D PoC: vendor kernel handles IC=1, OC=1, stride=1, pad=0,
        dilation=1 only. ggml CONV_2D op_params: [s0, s1, p0, p1, d0, d1].
        src[0] = kernel, src[1] = input. Anything else goes to NumPy.
        """
        if node.type != ddr.GGML_TYPE_F16:
            return False
        s0 = int(node.op_params[0]); s1 = int(node.op_params[1])
        p0 = int(node.op_params[2]); p1 = int(node.op_params[3])
        d0 = int(node.op_params[4]); d1 = int(node.op_params[5])
        if s0 != 1 or s1 != 1 or p0 != 0 or p1 != 0 or d0 != 1 or d1 != 1:
            return False
        kernel_id, input_id = node.src[0], node.src[1]
        if kernel_id == 0 or input_id == 0:
            return False
        kt = by_id.get(kernel_id)
        it = by_id.get(input_id)
        if kt is None or it is None:
            return False
        if kt.type != ddr.GGML_TYPE_F16 or it.type != ddr.GGML_TYPE_F16:
            return False
        # Vendor: IC=1, OC=1, single batch. ggml kernel layout (W, H, IC, OC),
        # input (W, H, IC, B). Trailing-1 dims required.
        if kt.ne[2] != 1 or kt.ne[3] != 1:
            return False
        if it.ne[2] != 1 or it.ne[3] != 1:
            return False
        k_w, k_h = kt.ne[0], kt.ne[1]
        in_w, in_h = it.ne[0], it.ne[1]
        if k_w <= 0 or k_h <= 0 or in_w < k_w or in_h < k_h:
            return False

        kb, ko = dev.buffer_for_data_addr(kt.data)
        ib, io = dev.buffer_for_data_addr(it.data)
        nk = ddr.tensor_nbytes(kt)
        ni = ddr.tensor_nbytes(it)
        kernel_bytes = bytes(kb.data[ko:ko + nk])
        input_bytes = bytes(ib.data[io:io + ni])
        try:
            log.info("[%s] node 0x%x CONV_2D pyspike route: in=(%d,%d) k=(%d,%d)",
                     self.addr, node.id, in_h, in_w, k_h, k_w)
            out_bytes = psr.SUPPORTED_PYSPIKE_OPS["conv_2d_fp16"](
                kernel_bytes, input_bytes, in_h, in_w, k_h, k_w)
        except Exception as e:
            log.warning("[%s] conv_2d pyspike failed (%s) → NumPy fallback",
                        self.addr, e)
            return False
        dst_buf, dst_off = dev.buffer_for_data_addr(node.data)
        dst_buf.data[dst_off:dst_off + len(out_bytes)] = out_bytes
        return True

    def _try_im2col_pyspike(self, node: wp.RpcTensor, by_id: dict, dev) -> bool:
        """IM2COL 2D: vendor kernel supports single-channel inputs with equal
        strides, zero padding, dilation=1. Anything richer goes to NumPy.
        ggml IM2COL op_params: [s0, s1, p0, p1, d0, d1, is_2D].
        src[0] = kernel (shape only), src[1] = input.
        """
        if node.type != ddr.GGML_TYPE_F16:
            return False
        # ggml IM2COL is_2D flag
        if int(node.op_params[6]) != 1:
            return False
        s0 = int(node.op_params[0])
        s1 = int(node.op_params[1])
        p0 = int(node.op_params[2])
        p1 = int(node.op_params[3])
        d0 = int(node.op_params[4])
        d1 = int(node.op_params[5])
        if s0 != s1 or s0 <= 0:
            return False
        if p0 != 0 or p1 != 0:
            return False
        if d0 != 1 or d1 != 1:
            return False
        kernel_id = node.src[0]
        input_id = node.src[1]
        if kernel_id == 0 or input_id == 0:
            return False
        kernel_t = by_id.get(kernel_id)
        input_t = by_id.get(input_id)
        if kernel_t is None or input_t is None:
            return False
        if input_t.type != ddr.GGML_TYPE_F16:
            return False
        # Vendor kernel: single channel only (kernel ne[2]=1).
        if kernel_t.ne[2] != 1 or kernel_t.ne[3] != 1:
            return False
        if input_t.ne[2] != 1 or input_t.ne[3] != 1:
            return False
        in_w, in_h = input_t.ne[0], input_t.ne[1]
        k_w, k_h = kernel_t.ne[0], kernel_t.ne[1]
        if in_w <= 0 or in_h <= 0 or k_w <= 0 or k_h <= 0:
            return False

        ib, io = dev.buffer_for_data_addr(input_t.data)
        n_in = ddr.tensor_nbytes(input_t)
        input_bytes = bytes(ib.data[io:io + n_in])
        try:
            log.info("[%s] node 0x%x IM2COL pyspike route: in=(%d,%d) k=(%d,%d) s=%d",
                     self.addr, node.id, in_h, in_w, k_h, k_w, s0)
            out_bytes = psr.SUPPORTED_PYSPIKE_OPS["im2col_fp16"](
                input_bytes, in_h, in_w, k_h, k_w, s0)
        except Exception as e:
            log.warning("[%s] im2col pyspike failed (%s) → NumPy fallback",
                        self.addr, e)
            return False
        dst_buf, dst_off = dev.buffer_for_data_addr(node.data)
        dst_buf.data[dst_off:dst_off + len(out_bytes)] = out_bytes
        return True

    def _try_upscale_pyspike(self, node: wp.RpcTensor, by_id: dict, dev) -> bool:
        """UPSCALE (nearest, integer factor) → reuse the REPEAT pyspike kernel.
        ggml semantics: dst[b, c, h, w] = src[b, c, h/sh, w/sw]. With mode=0
        (nearest) and integer scaling, this matches REPEAT exactly along the
        first two dims (REP0=sw, REP1=sh, REP2=REP3=1).
        """
        if node.type != ddr.GGML_TYPE_F16:
            return False
        mode = int(node.op_params[0]) & 0xFF
        if mode != 0:                                # nearest only
            return False
        src_id = node.src[0]
        if src_id == 0:
            return False
        src_t = by_id.get(src_id)
        if src_t is None or src_t.type != ddr.GGML_TYPE_F16:
            return False
        src_ne = tuple(src_t.ne)
        dst_ne = tuple(node.ne)
        if any(s <= 0 or d <= 0 for s, d in zip(src_ne, dst_ne)):
            return False
        # Only spatial upscaling (channel/batch identical).
        if dst_ne[2] != src_ne[2] or dst_ne[3] != src_ne[3]:
            return False
        if dst_ne[0] % src_ne[0] != 0 or dst_ne[1] % src_ne[1] != 0:
            return False
        src_total = src_ne[0] * src_ne[1] * src_ne[2] * src_ne[3] * 2
        dst_total = dst_ne[0] * dst_ne[1] * dst_ne[2] * dst_ne[3] * 2
        if src_total > 0x80000 or dst_total > 0x80000:
            return False

        sb, so = dev.buffer_for_data_addr(src_t.data)
        src_bytes = bytes(sb.data[so:so + src_total])
        try:
            log.info("[%s] node 0x%x UPSCALE pyspike route (via REPEAT): src=%s dst=%s",
                     self.addr, node.id, src_ne, dst_ne)
            out_bytes = psr.SUPPORTED_PYSPIKE_OPS["repeat_fp16"](
                src_bytes, src_ne, dst_ne)
        except Exception as e:
            log.warning("[%s] upscale pyspike failed (%s) → NumPy fallback",
                        self.addr, e)
            return False
        dst_buf, dst_off = dev.buffer_for_data_addr(node.data)
        dst_buf.data[dst_off:dst_off + len(out_bytes)] = out_bytes
        return True

    def _try_repeat_pyspike(self, node: wp.RpcTensor, by_id: dict, dev) -> bool:
        """REPEAT: dst.ne / src.ne tile factors along each of 4 dims. Both
        src and dst must fit in their 512KB L2 pool. ggml op_params is not
        used — the broadcast shape comes from node.ne."""
        if node.type != ddr.GGML_TYPE_F16:
            return False
        src_id = node.src[0]
        if src_id == 0:
            return False
        src_t = by_id.get(src_id)
        if src_t is None or src_t.type != ddr.GGML_TYPE_F16:
            return False
        src_ne = tuple(src_t.ne)
        dst_ne = tuple(node.ne)
        if any(s <= 0 or d <= 0 for s, d in zip(src_ne, dst_ne)):
            return False
        if any(d % s != 0 for s, d in zip(src_ne, dst_ne)):
            return False
        src_total = src_ne[0] * src_ne[1] * src_ne[2] * src_ne[3] * 2
        dst_total = dst_ne[0] * dst_ne[1] * dst_ne[2] * dst_ne[3] * 2
        if src_total > 0x80000 or dst_total > 0x80000:
            return False  # host-side tiling not implemented yet

        sb, so = dev.buffer_for_data_addr(src_t.data)
        src_bytes = bytes(sb.data[so:so + src_total])
        try:
            log.info("[%s] node 0x%x REPEAT pyspike route: src_ne=%s dst_ne=%s",
                     self.addr, node.id, src_ne, dst_ne)
            out_bytes = psr.SUPPORTED_PYSPIKE_OPS["repeat_fp16"](
                src_bytes, src_ne, dst_ne)
        except Exception as e:
            log.warning("[%s] repeat pyspike failed (%s) → NumPy fallback",
                        self.addr, e)
            return False
        dst_buf, dst_off = dev.buffer_for_data_addr(node.data)
        dst_buf.data[dst_off:dst_off + len(out_bytes)] = out_bytes
        return True

    def _try_arange_pyspike(self, node: wp.RpcTensor, dev) -> bool:
        """ARANGE has no input tensor — output length is in node.ne[0], and
        op_params[0..2] carry (start, stop, step) as float32-in-int32 bits.
        Firmware only handles start=0, step=1, so we route just that case
        (which is the dominant ggml use)."""
        import numpy as np
        if node.type != ddr.GGML_TYPE_F16:
            return False
        if node.ne[1] != 1 or node.ne[2] != 1 or node.ne[3] != 1:
            return False
        n = node.ne[0]
        if n <= 0 or n % 8 != 0:
            return False
        start = float(np.frombuffer(
            np.array(node.op_params[0], dtype=np.int32).tobytes(),
            dtype=np.float32)[0])
        step = float(np.frombuffer(
            np.array(node.op_params[2], dtype=np.int32).tobytes(),
            dtype=np.float32)[0])
        if start != 0.0 or step != 1.0:
            return False
        try:
            log.info("[%s] node 0x%x ARANGE pyspike route: N=%d",
                     self.addr, node.id, n)
            out_bytes = psr.SUPPORTED_PYSPIKE_OPS["arange_fp16"](n)
        except Exception as e:
            log.warning("[%s] arange pyspike failed (%s) → NumPy fallback",
                        self.addr, e)
            return False
        dst_buf, dst_off = dev.buffer_for_data_addr(node.data)
        dst_buf.data[dst_off:dst_off + len(out_bytes)] = out_bytes
        return True

    def _try_simple_unary_subop_pyspike(self, node: wp.RpcTensor, by_id: dict, dev) -> bool:
        """GGML_OP_UNARY sub-ops (CEIL, EXPM1, ...) that need W/H from the
        source tensor's ne instead of the generic 8-wide unary_intrin1 row
        pattern. Returns False on guard miss → caller falls back to NumPy.
        """
        if node.type != ddr.GGML_TYPE_F16:
            return False
        sub_op = node.op_params[0]
        entry = self._SIMPLE_UNARY_SUBOP_KERNELS.get(sub_op)
        if entry is None:
            return False
        src_id = node.src[0]
        if src_id == 0:
            return False
        src_t = by_id.get(src_id)
        if src_t is None or src_t.type != ddr.GGML_TYPE_F16:
            return False
        width = src_t.ne[0]
        height = src_t.ne[1] * src_t.ne[2] * src_t.ne[3]
        if width <= 0 or width * height <= 0:
            return False

        sb, so = dev.buffer_for_data_addr(src_t.data)
        nbytes = ddr.tensor_nbytes(src_t)
        src_bytes = bytes(sb.data[so:so + nbytes])

        kernel_name, runner_key = entry
        runner = psr.SUPPORTED_PYSPIKE_OPS[runner_key]
        try:
            log.info("[%s] node 0x%x %s pyspike route: W=%d H=%d (%d fp16)",
                     self.addr, node.id, kernel_name, width, height,
                     width * height)
            out_bytes = runner(src_bytes, width)
        except Exception as e:
            log.warning("[%s] simple-unary-subop pyspike failed (%s) → NumPy fallback",
                        self.addr, e)
            return False

        dst_buf, dst_off = dev.buffer_for_data_addr(node.data)
        dst_buf.data[dst_off:dst_off + len(out_bytes)] = out_bytes
        return True

    def _try_mul_mat_pyspike(self, node: wp.RpcTensor, by_id: dict, dev) -> bool:
        """MUL_MAT pyspike route — 2D contiguous fp16 only.
        ggml shape: src0 ne=(K,M), src1 ne=(K,N); dst is (M,N).
        Higher-rank (batched) shapes and non-fp16 fall back to NumPy.
        """
        if node.type != ddr.GGML_TYPE_F16:
            return False
        src0_id, src1_id = node.src[0], node.src[1]
        if src0_id == 0 or src1_id == 0:
            return False
        s0 = by_id.get(src0_id); s1 = by_id.get(src1_id)
        if s0 is None or s1 is None:
            return False
        if s0.type != ddr.GGML_TYPE_F16 or s1.type != ddr.GGML_TYPE_F16:
            return False
        # 2D only (batched matmul not supported by this kernel)
        if s0.ne[2] != 1 or s0.ne[3] != 1 or s1.ne[2] != 1 or s1.ne[3] != 1:
            return False
        K, M = s0.ne[0], s0.ne[1]
        if s1.ne[0] != K:
            return False
        N = s1.ne[1]
        # Each row must be bus-word aligned (kernel assumes WIDTH=8 fp16 cols).
        if K % 8 != 0:
            return False

        b0, o0 = dev.buffer_for_data_addr(s0.data)
        b1, o1 = dev.buffer_for_data_addr(s1.data)
        nbytes0 = ddr.tensor_nbytes(s0); nbytes1 = ddr.tensor_nbytes(s1)
        src0_bytes = bytes(b0.data[o0:o0 + nbytes0])
        src1_bytes = bytes(b1.data[o1:o1 + nbytes1])

        # tiled wrapper handles arbitrary M/N via host-side splits (firmware
        # call stays bounded to MUL_MAT_TILE_M × tile_n). K is firmware-tiled.
        runner = psr.SUPPORTED_PYSPIKE_OPS["mul_mat_tiled_fp16"]
        try:
            log.info("[%s] node 0x%x MUL_MAT pyspike route: M=%d K=%d N=%d "
                     "(tile=%dx%d)",
                     self.addr, node.id, M, K, N,
                     psr.MUL_MAT_TILE_M, psr.MUL_MAT_TILE_N)
            out_bytes = runner(src0_bytes, src1_bytes, M=M, K=K, N=N)
        except Exception as e:
            log.warning("[%s] mul_mat pyspike failed (%s) → NumPy fallback",
                        self.addr, e)
            return False
        dst_buf, dst_off = dev.buffer_for_data_addr(node.data)
        dst_buf.data[dst_off:dst_off + len(out_bytes)] = out_bytes
        return True

    def _try_pyspike(self, node: wp.RpcTensor, by_id: dict, dev) -> bool:
        """If `node` matches a pattern we have a pyspike kernel for, run it
        and write the result. Returns True on success, False to signal NumPy
        fallback. Any pyspike error is logged and also returns False.
        """
        if node.op == opr.GGML_OP_MUL_MAT:
            return self._try_mul_mat_pyspike(node, by_id, dev)
        if node.op == opr.GGML_OP_ARANGE:
            return self._try_arange_pyspike(node, dev)
        if node.op == opr.GGML_OP_REPEAT:
            return self._try_repeat_pyspike(node, by_id, dev)
        if node.op == opr.GGML_OP_UPSCALE:
            return self._try_upscale_pyspike(node, by_id, dev)
        if node.op == opr.GGML_OP_PAD:
            return self._try_pad_pyspike(node, by_id, dev)
        if node.op == opr.GGML_OP_CONCAT:
            return self._try_concat_pyspike(node, by_id, dev)
        if node.op == opr.GGML_OP_POOL_2D:
            return self._try_pool_2d_pyspike(node, by_id, dev)
        if node.op == opr.GGML_OP_IM2COL:
            return self._try_im2col_pyspike(node, by_id, dev)
        if node.op == opr.GGML_OP_CONV_2D:
            return self._try_conv_2d_pyspike(node, by_id, dev)
        if node.op in self._SIMPLE_UNARY_DISPATCH:
            return self._try_simple_unary_pyspike(node, by_id, dev)
        if (node.op == opr.GGML_OP_UNARY
                and node.op_params[0] in self._SIMPLE_UNARY_SUBOP_KERNELS):
            return self._try_simple_unary_subop_pyspike(node, by_id, dev)
        if node.op == opr.GGML_OP_UNARY:
            kernel = self._UNARY_PYSPIKE_KERNELS.get(node.op_params[0])
            kind = "unary"
            runner_entry = kernel  # 2-tuple (name, runner)
        else:
            entry = self._MAIN_OP_PYSPIKE_KERNELS.get(node.op)
            if entry is None:
                return False
            kernel_name, kind, runner = entry
            runner_entry = (kernel_name, runner)
            kernel = entry
        if kernel is None:
            return False
        if node.type != ddr.GGML_TYPE_F16:
            return False

        # Resolve source tensor bytes (skip numpy conversion).
        needed = 2 if kind == "binary" else 1
        src_bytes = []
        nbytes = 0
        for i in range(needed):
            src_id = node.src[i]
            if src_id == 0:
                return False
            src_t = by_id.get(src_id)
            if src_t is None or src_t.type != ddr.GGML_TYPE_F16:
                return False
            sb, so = dev.buffer_for_data_addr(src_t.data)
            n = ddr.tensor_nbytes(src_t)
            if n == 0 or n % psr.ROW_BYTES != 0:
                return False
            if i == 0:
                nbytes = n
            elif n != nbytes:
                return False  # binary inputs must match
            src_bytes.append(bytes(sb.data[so:so + n]))

        kernel_name, runner = runner_entry
        try:
            log.info("[%s] node 0x%x %s pyspike route: %d bytes (%d fp16) kind=%s",
                     self.addr, node.id, kernel_name, nbytes, nbytes // 2, kind)
            out_bytes = runner(*src_bytes)
        except Exception as e:
            log.warning("[%s] pyspike route failed (%s) → NumPy fallback", self.addr, e)
            return False

        dst_buf, dst_off = dev.buffer_for_data_addr(node.data)
        dst_buf.data[dst_off:dst_off + len(out_bytes)] = out_bytes
        return True

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
