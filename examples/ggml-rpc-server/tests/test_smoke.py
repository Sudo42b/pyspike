"""Smoke test: spin up the RPC server in-process and walk the full control
sequence a real ggml-rpc client uses on connect — HELLO → DEVICE_COUNT →
GET_ALIGNMENT → GET_MAX_SIZE → ALLOC_BUFFER → BUFFER_GET_BASE → SET_TENSOR
→ GET_TENSOR (roundtrip) → COPY_TENSOR → FREE_BUFFER. No pyspike needed.

Run: uv run --no-sync python3 examples/ggml-rpc-server/tests/test_smoke.py
"""
from __future__ import annotations

import os
import socket
import struct
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import wire_protocol as wp           # noqa: E402
import op_registry as opr            # noqa: E402
from pyspike_rpc_server import _ThreadedTCPServer, _Handler, PyspikeRpcServer  # noqa: E402


def _recv(sock, n):
    out = b""
    while len(out) < n:
        chunk = sock.recv(n - len(out))
        if not chunk:
            raise ConnectionError("peer closed")
        out += chunk
    return out


def send_cmd(sock, cmd: wp.RpcCmd, payload: bytes):
    sock.sendall(bytes([cmd]) + struct.pack("<Q", len(payload)) + payload)
    if cmd in wp.ONE_WAY:
        return None
    (size,) = struct.unpack("<Q", _recv(sock, 8))
    return _recv(sock, size) if size else b""


def main() -> int:
    srv = PyspikeRpcServer("127.0.0.1", 0)
    tcp = _ThreadedTCPServer(("127.0.0.1", 0), _Handler)
    tcp.pyspike_server = srv
    srv._tcp = tcp
    port = tcp.server_address[1]
    threading.Thread(target=tcp.serve_forever, daemon=True).start()
    print(f"[smoke] server bound on 127.0.0.1:{port}")

    failed = 0
    try:
        s = socket.create_connection(("127.0.0.1", port))

        # HELLO
        rsp = send_cmd(s, wp.RpcCmd.HELLO, b"\x00" * wp.RPC_CONN_CAPS_SIZE)
        major, minor, patch, _ = struct.unpack_from("<BBBB", rsp, 0)
        assert major == 4, f"major={major}"
        print(f"[smoke] HELLO ok — proto v{major}.{minor}.{patch}")

        # DEVICE_COUNT
        rsp = send_cmd(s, wp.RpcCmd.DEVICE_COUNT, b"")
        n_devices = struct.unpack("<I", rsp)[0]
        assert n_devices == 1
        print(f"[smoke] DEVICE_COUNT = {n_devices}")

        # GET_ALIGNMENT / GET_MAX_SIZE / GET_DEVICE_MEMORY
        alignment = struct.unpack("<Q", send_cmd(s, wp.RpcCmd.GET_ALIGNMENT, b""))[0]
        max_size = struct.unpack("<Q", send_cmd(s, wp.RpcCmd.GET_MAX_SIZE, b""))[0]
        free, total = struct.unpack("<QQ",
            send_cmd(s, wp.RpcCmd.GET_DEVICE_MEMORY, struct.pack("<I", 0)))
        assert alignment == 64
        assert max_size == 1 << 32
        print(f"[smoke] align={alignment} max={max_size:#x} mem={free/(1<<30):.1f}G/{total/(1<<30):.1f}G")

        # ALLOC_BUFFER
        remote_ptr, remote_size = struct.unpack("<QQ",
            send_cmd(s, wp.RpcCmd.ALLOC_BUFFER, struct.pack("<IQ", 0, 65536)))
        assert remote_size >= 65536
        print(f"[smoke] ALLOC_BUFFER ptr=0x{remote_ptr:x} size={remote_size}")

        # BUFFER_GET_BASE
        base_ptr = struct.unpack("<Q",
            send_cmd(s, wp.RpcCmd.BUFFER_GET_BASE, struct.pack("<Q", remote_ptr)))[0]
        assert base_ptr == remote_ptr
        print(f"[smoke] BUFFER_GET_BASE = 0x{base_ptr:x}")

        # BUFFER_CLEAR
        send_cmd(s, wp.RpcCmd.BUFFER_CLEAR, struct.pack("<QB", remote_ptr, 0xAB))
        print("[smoke] BUFFER_CLEAR ok")

        # SET_TENSOR / GET_TENSOR roundtrip
        t = wp.RpcTensor(
            id=0xdead, type=1, buffer=remote_ptr,
            ne=(16, 1, 1, 1), nb=(2, 32, 32, 32),
            op=opr.GGML_OP_ADD, data=base_ptr, name=b"smoke",
        )
        send_cmd(s, wp.RpcCmd.SET_TENSOR, t.pack() + struct.pack("<Q", 0) + bytes(range(32)))
        rsp = send_cmd(s, wp.RpcCmd.GET_TENSOR, t.pack() + struct.pack("<QQ", 0, 32))
        assert rsp == bytes(range(32)), f"roundtrip mismatch: {rsp.hex()}"
        print("[smoke] SET/GET_TENSOR roundtrip OK")

        # COPY_TENSOR src→dst
        dst_ptr = struct.unpack("<QQ",
            send_cmd(s, wp.RpcCmd.ALLOC_BUFFER, struct.pack("<IQ", 0, 65536)))[0]
        dst = wp.RpcTensor(
            id=0xbeef, type=1, buffer=dst_ptr,
            ne=(16, 1, 1, 1), nb=(2, 32, 32, 32),
            op=0, data=dst_ptr, name=b"dst",
        )
        rsp = send_cmd(s, wp.RpcCmd.COPY_TENSOR, t.pack() + dst.pack())
        assert rsp[0] == 1
        rsp = send_cmd(s, wp.RpcCmd.GET_TENSOR, dst.pack() + struct.pack("<QQ", 0, 32))
        assert rsp == bytes(range(32)), "COPY_TENSOR mismatch"
        print("[smoke] COPY_TENSOR + GET OK")

        # GRAPH_COMPUTE stub (one-way)
        graph = (
            struct.pack("<II", 0, 1)
            + struct.pack("<Q", t.id)
            + struct.pack("<I", 1)
            + t.pack()
        )
        send_cmd(s, wp.RpcCmd.GRAPH_COMPUTE, graph)
        print("[smoke] GRAPH_COMPUTE stub accepted")

        # FREE_BUFFER both
        send_cmd(s, wp.RpcCmd.FREE_BUFFER, struct.pack("<Q", remote_ptr))
        send_cmd(s, wp.RpcCmd.FREE_BUFFER, struct.pack("<Q", dst_ptr))
        print("[smoke] FREE_BUFFER x2 ok")

        s.close()
    except Exception as e:
        print(f"[smoke] FAIL: {e!r}")
        import traceback
        traceback.print_exc()
        failed = 1
    finally:
        tcp.shutdown()
        tcp.server_close()

    print("\n[smoke]", "PASS" if not failed else "FAIL")
    return failed


if __name__ == "__main__":
    sys.exit(main())
