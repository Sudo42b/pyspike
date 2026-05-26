"""ggml-rpc wire protocol — Python reimplementation of ggml/src/ggml-rpc.

Mirrors the C++ packed structs from upstream ggml (proto v4.0.0). All structs
are little-endian and 1-byte packed unless noted. Socket framing:

    request : cmd_byte (u8) | input_size (u64) | input_data[input_size]
    response: output_size (u64) | output_data[output_size]

One-way commands (SET_TENSOR, GRAPH_COMPUTE, GRAPH_RECOMPUTE) skip the response.
"""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass, field
from enum import IntEnum

GGML_MAX_DIMS = 4
GGML_MAX_SRC = 10
GGML_MAX_OP_PARAMS = 64
GGML_MAX_NAME = 64
RPC_CONN_CAPS_SIZE = 24
MAX_CHUNK_SIZE = 1024 * 1024 * 1024

RPC_PROTO_MAJOR_VERSION = 4
RPC_PROTO_MINOR_VERSION = 0
RPC_PROTO_PATCH_VERSION = 0

HASH_THRESHOLD = 10 * 1024 * 1024


class RpcCmd(IntEnum):
    ALLOC_BUFFER = 0
    GET_ALIGNMENT = 1
    GET_MAX_SIZE = 2
    BUFFER_GET_BASE = 3
    FREE_BUFFER = 4
    BUFFER_CLEAR = 5
    SET_TENSOR = 6
    SET_TENSOR_HASH = 7
    GET_TENSOR = 8
    COPY_TENSOR = 9
    GRAPH_COMPUTE = 10
    GET_DEVICE_MEMORY = 11
    INIT_TENSOR = 12
    GET_ALLOC_SIZE = 13
    HELLO = 14
    DEVICE_COUNT = 15
    GRAPH_RECOMPUTE = 16


ONE_WAY = frozenset({RpcCmd.SET_TENSOR, RpcCmd.GRAPH_COMPUTE, RpcCmd.GRAPH_RECOMPUTE})


_RPC_TENSOR_FMT = (
    "<"
    "Q"                                       # id
    "I"                                       # type
    "Q"                                       # buffer
    f"{GGML_MAX_DIMS}I"                       # ne[4]
    f"{GGML_MAX_DIMS}I"                       # nb[4]
    "I"                                       # op
    f"{GGML_MAX_OP_PARAMS // 4}i"             # op_params[16]
    "i"                                       # flags
    f"{GGML_MAX_SRC}Q"                        # src[10]
    "Q"                                       # view_src
    "Q"                                       # view_offs
    "Q"                                       # data
    f"{GGML_MAX_NAME}s"                       # name
    "4s"                                      # padding
)
RPC_TENSOR_SIZE = struct.calcsize(_RPC_TENSOR_FMT)
assert RPC_TENSOR_SIZE == 296, f"rpc_tensor expected 296, got {RPC_TENSOR_SIZE}"
assert RPC_TENSOR_SIZE % 8 == 0, "rpc_tensor must be 8-byte aligned"


@dataclass
class RpcTensor:
    id: int = 0
    type: int = 0
    buffer: int = 0
    ne: tuple = (0, 0, 0, 0)
    nb: tuple = (0, 0, 0, 0)
    op: int = 0
    op_params: tuple = field(default_factory=lambda: (0,) * (GGML_MAX_OP_PARAMS // 4))
    flags: int = 0
    src: tuple = field(default_factory=lambda: (0,) * GGML_MAX_SRC)
    view_src: int = 0
    view_offs: int = 0
    data: int = 0
    name: bytes = b""

    def pack(self) -> bytes:
        name = self.name[:GGML_MAX_NAME].ljust(GGML_MAX_NAME, b"\x00")
        return struct.pack(
            _RPC_TENSOR_FMT,
            self.id, self.type, self.buffer,
            *self.ne, *self.nb,
            self.op,
            *self.op_params,
            self.flags,
            *self.src,
            self.view_src, self.view_offs, self.data,
            name, b"\x00\x00\x00\x00",
        )

    @classmethod
    def unpack(cls, buf: bytes, offset: int = 0) -> "RpcTensor":
        fields = struct.unpack_from(_RPC_TENSOR_FMT, buf, offset)
        i = 0
        tid = fields[i]; i += 1
        ttype = fields[i]; i += 1
        buffer = fields[i]; i += 1
        ne = fields[i:i + GGML_MAX_DIMS]; i += GGML_MAX_DIMS
        nb = fields[i:i + GGML_MAX_DIMS]; i += GGML_MAX_DIMS
        op = fields[i]; i += 1
        op_params = fields[i:i + GGML_MAX_OP_PARAMS // 4]; i += GGML_MAX_OP_PARAMS // 4
        flags = fields[i]; i += 1
        src = fields[i:i + GGML_MAX_SRC]; i += GGML_MAX_SRC
        view_src = fields[i]; i += 1
        view_offs = fields[i]; i += 1
        data = fields[i]; i += 1
        name = fields[i].rstrip(b"\x00")
        return cls(
            id=tid, type=ttype, buffer=buffer,
            ne=tuple(ne), nb=tuple(nb), op=op,
            op_params=tuple(op_params), flags=flags,
            src=tuple(src), view_src=view_src, view_offs=view_offs,
            data=data, name=name,
        )


# --- Fixed-size message structs (request / response) ---

_FMT_HELLO_REQ = f"<{RPC_CONN_CAPS_SIZE}s"
_FMT_HELLO_RSP = f"<BBBB{RPC_CONN_CAPS_SIZE}s"

_FMT_ALLOC_BUFFER_REQ = "<IQ"
_FMT_ALLOC_BUFFER_RSP = "<QQ"

_FMT_BUFFER_GET_BASE_REQ = "<Q"
_FMT_BUFFER_GET_BASE_RSP = "<Q"

_FMT_FREE_BUFFER_REQ = "<Q"
_FMT_BUFFER_CLEAR_REQ = "<QB"

_FMT_GET_TENSOR_REQ = f"<{RPC_TENSOR_SIZE}sQQ"
_FMT_SET_TENSOR_HEADER = f"<{RPC_TENSOR_SIZE}sQ"  # raw bytes follow
_FMT_SET_TENSOR_HASH_REQ = f"<{RPC_TENSOR_SIZE}sQQ"
_FMT_SET_TENSOR_HASH_RSP = "<B"

_FMT_COPY_TENSOR_REQ = f"<{RPC_TENSOR_SIZE}s{RPC_TENSOR_SIZE}s"
_FMT_COPY_TENSOR_RSP = "<B"

_FMT_INIT_TENSOR_REQ = f"<{RPC_TENSOR_SIZE}s"

_FMT_GET_ALLOC_SIZE_REQ = f"<I{RPC_TENSOR_SIZE}s{GGML_MAX_SRC * RPC_TENSOR_SIZE}s"
_FMT_GET_ALLOC_SIZE_RSP = "<Q"

_FMT_GET_DEVICE_MEMORY_REQ = "<I"
_FMT_GET_DEVICE_MEMORY_RSP = "<QQ"

_FMT_DEVICE_COUNT_RSP = "<I"
_FMT_GET_ALIGNMENT_RSP = "<Q"
_FMT_GET_MAX_SIZE_RSP = "<Q"
_FMT_GRAPH_COMPUTE_RSP = "<B"


def pack_hello_rsp(conn_caps: bytes = b"\x00" * RPC_CONN_CAPS_SIZE) -> bytes:
    return struct.pack(
        _FMT_HELLO_RSP,
        RPC_PROTO_MAJOR_VERSION,
        RPC_PROTO_MINOR_VERSION,
        RPC_PROTO_PATCH_VERSION,
        0,
        conn_caps[:RPC_CONN_CAPS_SIZE].ljust(RPC_CONN_CAPS_SIZE, b"\x00"),
    )


def unpack_hello_req(buf: bytes) -> bytes:
    (conn_caps,) = struct.unpack(_FMT_HELLO_REQ, buf)
    return conn_caps


def unpack_alloc_buffer_req(buf: bytes) -> tuple[int, int]:
    return struct.unpack(_FMT_ALLOC_BUFFER_REQ, buf)


def pack_alloc_buffer_rsp(remote_ptr: int, remote_size: int) -> bytes:
    return struct.pack(_FMT_ALLOC_BUFFER_RSP, remote_ptr, remote_size)


def unpack_free_buffer_req(buf: bytes) -> int:
    return struct.unpack(_FMT_FREE_BUFFER_REQ, buf)[0]


def unpack_buffer_get_base_req(buf: bytes) -> int:
    return struct.unpack(_FMT_BUFFER_GET_BASE_REQ, buf)[0]


def pack_buffer_get_base_rsp(base_ptr: int) -> bytes:
    return struct.pack(_FMT_BUFFER_GET_BASE_RSP, base_ptr)


def unpack_buffer_clear_req(buf: bytes) -> tuple[int, int]:
    return struct.unpack(_FMT_BUFFER_CLEAR_REQ, buf)


def unpack_set_tensor_msg(buf: bytes) -> tuple[RpcTensor, int, bytes]:
    """SET_TENSOR is rpc_tensor || offset:u64 || raw_bytes[...]"""
    tensor_bytes, offset = struct.unpack_from(_FMT_SET_TENSOR_HEADER, buf, 0)
    header_size = struct.calcsize(_FMT_SET_TENSOR_HEADER)
    return RpcTensor.unpack(tensor_bytes), offset, buf[header_size:]


def unpack_set_tensor_hash_req(buf: bytes) -> tuple[RpcTensor, int, int]:
    tensor_bytes, offset, hash_ = struct.unpack(_FMT_SET_TENSOR_HASH_REQ, buf)
    return RpcTensor.unpack(tensor_bytes), offset, hash_


def pack_set_tensor_hash_rsp(result: int) -> bytes:
    return struct.pack(_FMT_SET_TENSOR_HASH_RSP, result & 0xFF)


def unpack_get_tensor_req(buf: bytes) -> tuple[RpcTensor, int, int]:
    tensor_bytes, offset, size = struct.unpack(_FMT_GET_TENSOR_REQ, buf)
    return RpcTensor.unpack(tensor_bytes), offset, size


def unpack_copy_tensor_req(buf: bytes) -> tuple[RpcTensor, RpcTensor]:
    src_bytes, dst_bytes = struct.unpack(_FMT_COPY_TENSOR_REQ, buf)
    return RpcTensor.unpack(src_bytes), RpcTensor.unpack(dst_bytes)


def pack_copy_tensor_rsp(result: int) -> bytes:
    return struct.pack(_FMT_COPY_TENSOR_RSP, result & 0xFF)


def unpack_init_tensor_req(buf: bytes) -> RpcTensor:
    (tensor_bytes,) = struct.unpack(_FMT_INIT_TENSOR_REQ, buf)
    return RpcTensor.unpack(tensor_bytes)


def unpack_get_alloc_size_req(buf: bytes) -> tuple[int, RpcTensor, list]:
    device, tensor_bytes, srcs_bytes = struct.unpack(_FMT_GET_ALLOC_SIZE_REQ, buf)
    tensor = RpcTensor.unpack(tensor_bytes)
    srcs = [RpcTensor.unpack(srcs_bytes, i * RPC_TENSOR_SIZE) for i in range(GGML_MAX_SRC)]
    return device, tensor, srcs


def pack_get_alloc_size_rsp(alloc_size: int) -> bytes:
    return struct.pack(_FMT_GET_ALLOC_SIZE_RSP, alloc_size)


def unpack_get_device_memory_req(buf: bytes) -> int:
    return struct.unpack(_FMT_GET_DEVICE_MEMORY_REQ, buf)[0]


def pack_get_device_memory_rsp(free_mem: int, total_mem: int) -> bytes:
    return struct.pack(_FMT_GET_DEVICE_MEMORY_RSP, free_mem, total_mem)


def pack_device_count_rsp(n: int) -> bytes:
    return struct.pack(_FMT_DEVICE_COUNT_RSP, n)


def pack_get_alignment_rsp(alignment: int) -> bytes:
    return struct.pack(_FMT_GET_ALIGNMENT_RSP, alignment)


def pack_get_max_size_rsp(max_size: int) -> bytes:
    return struct.pack(_FMT_GET_MAX_SIZE_RSP, max_size)


def pack_graph_compute_rsp(result: int) -> bytes:
    return struct.pack(_FMT_GRAPH_COMPUTE_RSP, result & 0xFF)


def unpack_graph_compute_msg(buf: bytes) -> tuple[int, list[int], list[RpcTensor]]:
    """GRAPH_COMPUTE input: device:u32 | n_nodes:u32 | nodes[n]:u64 |
    n_tensors:u32 | tensors[n_tensors] (each RPC_TENSOR_SIZE bytes)
    """
    device, n_nodes = struct.unpack_from("<II", buf, 0)
    pos = 8
    nodes = list(struct.unpack_from(f"<{n_nodes}Q", buf, pos))
    pos += 8 * n_nodes
    (n_tensors,) = struct.unpack_from("<I", buf, pos)
    pos += 4
    tensors = [RpcTensor.unpack(buf, pos + i * RPC_TENSOR_SIZE) for i in range(n_tensors)]
    return device, nodes, tensors


# --- Socket framing ---

def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(min(remaining, 65536))
        if not chunk:
            raise ConnectionError(f"peer closed; expected {n} bytes, got {n - remaining}")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_request(sock: socket.socket) -> tuple[RpcCmd, bytes]:
    cmd_byte = _recv_exact(sock, 1)
    cmd = RpcCmd(cmd_byte[0])
    (size,) = struct.unpack("<Q", _recv_exact(sock, 8))
    payload = _recv_exact(sock, size) if size else b""
    return cmd, payload


def send_response(sock: socket.socket, payload: bytes) -> None:
    sock.sendall(struct.pack("<Q", len(payload)) + payload)
