"""Same as test_smoke.py but talks to a remote server.

Usage: python3 test_remote_smoke.py [HOST [PORT]]
       (default: 100.103.189.26 50052)
"""
from __future__ import annotations

import os
import socket
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import wire_protocol as wp           # noqa: E402
import op_registry as opr            # noqa: E402


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


def main() -> int:
    host = sys.argv[1] if len(sys.argv) > 1 else "100.103.189.26"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 50052
    print(f"[remote-smoke] connecting to {host}:{port}")

    s = socket.create_connection((host, port), timeout=5)
    try:
        # HELLO
        rsp = send_cmd(s, wp.RpcCmd.HELLO, b"\x00" * wp.RPC_CONN_CAPS_SIZE)
        major, minor, patch, _ = struct.unpack_from("<BBBB", rsp, 0)
        print(f"  HELLO ok — proto v{major}.{minor}.{patch}")

        # DEVICE_COUNT / GET_DEVICE_MEMORY
        n_devices = struct.unpack("<I",
            send_cmd(s, wp.RpcCmd.DEVICE_COUNT, b""))[0]
        free, total = struct.unpack("<QQ",
            send_cmd(s, wp.RpcCmd.GET_DEVICE_MEMORY, struct.pack("<I", 0)))
        print(f"  devices={n_devices} mem={free/(1<<30):.1f}G/{total/(1<<30):.1f}G")

        # ALLOC, SET/GET, FREE roundtrip
        remote_ptr, _ = struct.unpack("<QQ",
            send_cmd(s, wp.RpcCmd.ALLOC_BUFFER, struct.pack("<IQ", 0, 65536)))
        base = struct.unpack("<Q",
            send_cmd(s, wp.RpcCmd.BUFFER_GET_BASE, struct.pack("<Q", remote_ptr)))[0]
        t = wp.RpcTensor(
            id=0xcafe, type=1, buffer=remote_ptr,
            ne=(16, 1, 1, 1), nb=(2, 32, 32, 32),
            op=opr.GGML_OP_UNARY, op_params=(opr.GGML_UNARY_OP_ABS,) + (0,) * 15,
            data=base, name=b"remote",
        )
        # Write 64 bytes
        payload_in = bytes(range(64))
        send_cmd(s, wp.RpcCmd.SET_TENSOR, t.pack() + struct.pack("<Q", 0) + payload_in)
        rsp = send_cmd(s, wp.RpcCmd.GET_TENSOR, t.pack() + struct.pack("<QQ", 0, 64))
        assert rsp == payload_in, f"roundtrip mismatch (got {rsp[:8].hex()})"
        print(f"  SET/GET roundtrip OK — {len(rsp)} bytes intact")

        # GRAPH_COMPUTE stub (one-way) — should be acked by server log but no rsp
        graph = (
            struct.pack("<II", 0, 1)
            + struct.pack("<Q", t.id)
            + struct.pack("<I", 1)
            + t.pack()
        )
        send_cmd(s, wp.RpcCmd.GRAPH_COMPUTE, graph)
        print("  GRAPH_COMPUTE accepted (M1 stub — see server log for op resolution)")

        send_cmd(s, wp.RpcCmd.FREE_BUFFER, struct.pack("<Q", remote_ptr))
        print("  FREE_BUFFER ok")

        print("\n[remote-smoke] PASS")
        return 0
    except Exception as e:
        print(f"\n[remote-smoke] FAIL: {e!r}")
        import traceback; traceback.print_exc()
        return 1
    finally:
        s.close()


if __name__ == "__main__":
    sys.exit(main())
