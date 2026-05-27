"""Localhost ABS-unary test against the running server.

Sends a UNARY+ABS fp16 graph (N=16, contiguous) and verifies the response.
The server's `_h_graph_compute` decides between NumPy fallback (default) and
pyspike .elf route (when PYSPIKE_RPC_BACKEND=spike is set in the server env).
Either way the *result* should be `abs(input)`; only the server log distinguishes
which path ran.

Usage:
    python3 test_abs_pyspike.py [HOST [PORT [N]]]
"""
from __future__ import annotations

import os
import socket
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import wire_protocol as wp           # noqa: E402
import op_registry as opr            # noqa: E402
import ddr_backend as ddr            # noqa: E402


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


def make_tensor(tid, dtype, shape, data_addr, op=0, op_params=(), src=()):
    elem = ddr.ggml_dtype(dtype).itemsize
    ne = list(shape) + [1] * (4 - len(shape))
    nb = [elem]
    for d in range(1, 4):
        nb.append(nb[-1] * ne[d - 1])
    return wp.RpcTensor(
        id=tid, type=dtype, ne=tuple(ne), nb=tuple(nb),
        op=op, op_params=tuple(op_params) + (0,) * (16 - len(op_params)),
        src=tuple(src) + (0,) * (10 - len(src)),
        data=data_addr,
    )


def main():
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 50052
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    if n % 8 != 0:
        print(f"N must be divisible by 8, got {n}")
        return 1

    s = socket.create_connection((host, port), timeout=600)
    send_cmd(s, wp.RpcCmd.HELLO, b"\x00" * wp.RPC_CONN_CAPS_SIZE)
    print(f"connected to {host}:{port}, N={n} (fp16 ABS)")

    remote_ptr, _ = struct.unpack("<QQ", send_cmd(s, wp.RpcCmd.ALLOC_BUFFER,
                                                   struct.pack("<IQ", 0, 1 << 20)))

    # input tensor (id=1)
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(n).astype(np.float32) * 3.0).astype(np.float16)
    src_t = make_tensor(1, ddr.GGML_TYPE_F16, (n,), remote_ptr)
    send_cmd(s, wp.RpcCmd.SET_TENSOR,
             src_t.pack() + struct.pack("<Q", 0) + x.tobytes())

    # output tensor (id=2) — UNARY+ABS over src[0]=1
    dst_addr = remote_ptr + ((n * 2 + 63) // 64) * 64
    dst_t = make_tensor(2, ddr.GGML_TYPE_F16, (n,), dst_addr,
                        op=opr.GGML_OP_UNARY,
                        op_params=(opr.GGML_UNARY_OP_ABS,),
                        src=(1,))

    graph = (
        struct.pack("<II", 0, 1)              # device=0, n_nodes=1
        + struct.pack("<Q", dst_t.id)          # nodes[0] = dst id
        + struct.pack("<I", 2)                 # n_tensors=2
        + src_t.pack() + dst_t.pack()
    )
    print("sending GRAPH_COMPUTE (timeout 600s for first spike build)...")
    send_cmd(s, wp.RpcCmd.GRAPH_COMPUTE, graph)

    rsp = send_cmd(s, wp.RpcCmd.GET_TENSOR, dst_t.pack() + struct.pack("<QQ", 0, n * 2))
    got = np.frombuffer(rsp, dtype=np.float16)
    ref = np.abs(x)

    diff = np.max(np.abs(got.astype(np.float32) - ref.astype(np.float32)))
    print(f"input    : {x[:8]}")
    print(f"got      : {got[:8]}")
    print(f"expected : {ref[:8]}")
    print(f"max diff : {diff:.4g}  →  {'PASS' if diff < 5e-3 else 'FAIL'}")

    send_cmd(s, wp.RpcCmd.FREE_BUFFER, struct.pack("<Q", remote_ptr))
    s.close()
    return 0 if diff < 5e-3 else 1


if __name__ == "__main__":
    sys.exit(main())
