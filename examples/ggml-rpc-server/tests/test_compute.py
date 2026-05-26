"""M2 end-to-end: SET_TENSOR a known input, run GRAPH_COMPUTE for various
ops via the NumPy backend, GET_TENSOR the result, verify against numpy ref.

Run: uv run --no-sync python3 examples/ggml-rpc-server/tests/test_compute.py
"""
from __future__ import annotations

import os
import socket
import struct
import sys
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import wire_protocol as wp           # noqa: E402
import op_registry as opr            # noqa: E402
import ddr_backend as ddr            # noqa: E402
from pyspike_rpc_server import _ThreadedTCPServer, _Handler, PyspikeRpcServer  # noqa: E402


# ----- transport helpers -----
def _recv(sock, n):
    out = b""
    while len(out) < n:
        chunk = sock.recv(n - len(out))
        if not chunk:
            raise ConnectionError("peer closed")
        out += chunk
    return out


def send_cmd(sock, cmd, payload):
    sock.sendall(bytes([cmd]) + struct.pack("<Q", len(payload)) + payload)
    if cmd in wp.ONE_WAY:
        return None
    (size,) = struct.unpack("<Q", _recv(sock, 8))
    return _recv(sock, size) if size else b""


def hello(s):
    rsp = send_cmd(s, wp.RpcCmd.HELLO, b"\x00" * wp.RPC_CONN_CAPS_SIZE)
    assert rsp[0] == wp.RPC_PROTO_MAJOR_VERSION


def alloc(s, size):
    rsp = send_cmd(s, wp.RpcCmd.ALLOC_BUFFER, struct.pack("<IQ", 0, size))
    ptr, _sz = struct.unpack("<QQ", rsp)
    return ptr


def set_tensor(s, t: wp.RpcTensor, offset: int, data: bytes):
    send_cmd(s, wp.RpcCmd.SET_TENSOR, t.pack() + struct.pack("<Q", offset) + data)


def get_tensor(s, t: wp.RpcTensor, offset: int, size: int) -> bytes:
    return send_cmd(s, wp.RpcCmd.GET_TENSOR, t.pack() + struct.pack("<QQ", offset, size))


def graph_compute(s, nodes: list[int], tensors: list[wp.RpcTensor]):
    payload = (
        struct.pack("<II", 0, len(nodes))
        + struct.pack(f"<{len(nodes)}Q", *nodes)
        + struct.pack("<I", len(tensors))
        + b"".join(t.pack() for t in tensors)
    )
    send_cmd(s, wp.RpcCmd.GRAPH_COMPUTE, payload)


# ----- helpers to build tensors with proper ne/nb -----
def make_tensor(tid: int, dtype: int, shape: tuple[int, ...], data_addr: int,
                buffer: int = 0, op: int = 0, op_params: tuple = (),
                src: tuple = (), name: bytes = b"") -> wp.RpcTensor:
    """Build a contiguous tensor. `shape` is ggml-order (ne[0] = inner)."""
    elem_size = ddr.ggml_dtype(dtype).itemsize
    ne = list(shape) + [1] * (4 - len(shape))
    nb = [elem_size]
    for d in range(1, 4):
        nb.append(nb[-1] * ne[d - 1])
    op_params_padded = tuple(op_params) + (0,) * (16 - len(op_params))
    src_padded = tuple(src) + (0,) * (10 - len(src))
    return wp.RpcTensor(
        id=tid, type=dtype, buffer=buffer,
        ne=tuple(ne), nb=tuple(nb),
        op=op, op_params=op_params_padded,
        src=src_padded, data=data_addr, name=name,
    )


# ----- main test -----
def run_case(s, base_ptr: int, case_name: str,
             input_arrays: list[np.ndarray], node_op, node_op_params,
             expected: np.ndarray, dtype=ddr.GGML_TYPE_F16) -> bool:
    """Layout: src tensors back-to-back, then dst. Each gets a fresh id."""
    elem = ddr.ggml_dtype(dtype).itemsize
    cur_addr = base_ptr
    tensors = []
    next_id = 1

    src_ids = []
    for arr in input_arrays:
        arr16 = arr.astype(ddr.ggml_dtype(dtype), copy=False)
        nbytes = arr16.size * elem
        t = make_tensor(next_id, dtype, arr16.shape[::-1], cur_addr,
                        name=f"src{next_id}".encode())
        tensors.append(t)
        src_ids.append(next_id)
        # write data
        set_tensor(s, t, 0, arr16.tobytes())
        cur_addr += ((nbytes + 63) // 64) * 64
        next_id += 1

    dst_t = make_tensor(next_id, dtype, expected.shape[::-1], cur_addr,
                        op=node_op, op_params=node_op_params,
                        src=tuple(src_ids),
                        name=f"dst_{case_name}".encode())
    tensors.append(dst_t)
    dst_size = expected.size * elem
    cur_addr += ((dst_size + 63) // 64) * 64

    graph_compute(s, [dst_t.id], tensors)

    raw = get_tensor(s, dst_t, 0, dst_size)
    got = np.frombuffer(raw, dtype=ddr.ggml_dtype(dtype)).reshape(expected.shape)
    ok = np.allclose(got.astype(np.float32), expected.astype(np.float32),
                     atol=5e-3, rtol=5e-3)
    status = "PASS" if ok else "FAIL"
    diff = np.max(np.abs(got.astype(np.float32) - expected.astype(np.float32)))
    print(f"  {case_name:<18} {status}  max_abs_diff={diff:.4g}")
    if not ok:
        print(f"    got     ={got.flatten()[:8]}")
        print(f"    expected={expected.flatten()[:8]}")
    return ok, cur_addr


def main() -> int:
    srv = PyspikeRpcServer("127.0.0.1", 0)
    tcp = _ThreadedTCPServer(("127.0.0.1", 0), _Handler)
    tcp.pyspike_server = srv
    srv._tcp = tcp
    port = tcp.server_address[1]
    threading.Thread(target=tcp.serve_forever, daemon=True).start()
    print(f"[compute] server on 127.0.0.1:{port}")

    rng = np.random.default_rng(42)
    failed = 0

    try:
        s = socket.create_connection(("127.0.0.1", port))
        hello(s)
        base = alloc(s, 1 << 20)   # 1 MiB scratch

        cur = base

        # --- ABS (unary) ---
        x = rng.standard_normal(16).astype(np.float32)
        ok, cur = run_case(s, cur, "UNARY_ABS", [x],
                           opr.GGML_OP_UNARY, (opr.GGML_UNARY_OP_ABS,) + (0,) * 15,
                           np.abs(x))
        failed += not ok

        # --- NEG ---
        ok, cur = run_case(s, cur, "UNARY_NEG", [x],
                           opr.GGML_OP_UNARY, (opr.GGML_UNARY_OP_NEG,) + (0,) * 15,
                           -x)
        failed += not ok

        # --- SILU ---
        silu_ref = x * (1.0 / (1.0 + np.exp(-x)))
        ok, cur = run_case(s, cur, "UNARY_SILU", [x],
                           opr.GGML_OP_UNARY, (opr.GGML_UNARY_OP_SILU,) + (0,) * 15,
                           silu_ref)
        failed += not ok

        # --- ADD (binary) ---
        a = rng.standard_normal(16).astype(np.float32)
        b = rng.standard_normal(16).astype(np.float32)
        ok, cur = run_case(s, cur, "BINARY_ADD", [a, b],
                           opr.GGML_OP_ADD, (0,) * 16, a + b)
        failed += not ok

        # --- MUL ---
        ok, cur = run_case(s, cur, "BINARY_MUL", [a, b],
                           opr.GGML_OP_MUL, (0,) * 16, a * b)
        failed += not ok

        # --- SOFT_MAX (no mask, no scale) ---
        logits = rng.standard_normal(16).astype(np.float32) * 2.0
        sm = np.exp(logits - logits.max())
        sm /= sm.sum()
        ok, cur = run_case(s, cur, "SOFT_MAX", [logits],
                           opr.GGML_OP_SOFT_MAX, (0,) * 16, sm)
        failed += not ok

        # --- RMS_NORM (eps = 1e-5 packed as fp32 bits in op_params[0]) ---
        eps = 1e-5
        eps_i32 = int.from_bytes(np.float32(eps).tobytes(), "little", signed=True)
        rms_ref = x / np.sqrt((x * x).mean() + eps)
        ok, cur = run_case(s, cur, "RMS_NORM", [x],
                           opr.GGML_OP_RMS_NORM, (eps_i32,) + (0,) * 15,
                           rms_ref)
        failed += not ok

        # --- SCALE (scale=0.5, bias=0) ---
        sc = 0.5
        sc_i32 = int.from_bytes(np.float32(sc).tobytes(), "little", signed=True)
        ok, cur = run_case(s, cur, "SCALE",   [x],
                           opr.GGML_OP_SCALE, (sc_i32, 0) + (0,) * 14, x * sc)
        failed += not ok

        # --- MUL_MAT: dst[i,j] = sum_k src0[k,i] * src1[k,j]
        # ggml: src0 = K x M (weights), src1 = K x N, dst = M x N
        # numpy reversed: src0.shape = (M, K), src1.shape = (N, K)
        # result: dst = src1 @ src0.T  shape (N, M) which reversed = (M, N) in ggml
        K, M, N = 8, 4, 3
        w = rng.standard_normal((M, K)).astype(np.float32)    # (M, K) numpy
        a2 = rng.standard_normal((N, K)).astype(np.float32)   # (N, K) numpy
        ref_mm = a2 @ w.T                                      # (N, M)
        # For run_case we send shapes as numpy-row-major; make_tensor reverses to ggml ne
        ok, cur = run_case(s, cur, "MUL_MAT",  [w, a2],
                           opr.GGML_OP_MUL_MAT, (0,) * 16, ref_mm)
        failed += not ok

        s.close()
    finally:
        tcp.shutdown(); tcp.server_close()

    print(f"\n[compute] {'PASS' if not failed else f'FAIL ({failed})'}")
    return failed


if __name__ == "__main__":
    sys.exit(main())
