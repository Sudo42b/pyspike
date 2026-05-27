#!/usr/bin/env python3
"""Convert test/<OP>/n1s16/n1s16_<op>.c kernels into firmware_templates/<op>.c.tpl
ready for `pyspike_runner._render_template`.

The transformation is intentionally shallow: parametrise shape-related
`#define`s and any op-specific scalars (eps, etc.) into `{{KEY}}` placeholders,
keep everything else verbatim. Each op has its own manifest entry — the few
ops whose .c renames a define (e.g. NORM's `FP16_EPS_1E_NEG_5`) carry extra
regex replacements there.

Usage:
    python tools/generate_templates.py              # process every op in OP_MANIFEST
    python tools/generate_templates.py --op sqr     # one op
    python tools/generate_templates.py --diff       # show diff against existing .tpl
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TEST_DIR = REPO_ROOT / "test"
OUT_DIR = Path(__file__).resolve().parents[1] / "firmware_templates"


@dataclass
class OpManifest:
    source_dir: str           # test/<source_dir>/n1s16/n1s16_<source_stem>.c
    source_stem: str          # bare filename without n1s16_ prefix
    placeholders: list[str]   # #define names to turn into {{NAME}} placeholders
    extra_replacements: list[tuple[str, str]] = field(default_factory=list)
    template_name: str | None = None    # default: f"unary_{source_stem}"
    header_comment: str | None = None   # 1-line summary in the generated header

    @property
    def out_path(self) -> Path:
        name = self.template_name or f"unary_{self.source_stem}"
        return OUT_DIR / f"{name}.c.tpl"

    @property
    def src_path(self) -> Path:
        return TEST_DIR / self.source_dir / "n1s16" / f"n1s16_{self.source_stem}.c"


# Shape #defines parametrised by the runner. Edit by adding an entry below.
OP_MANIFEST: dict[str, OpManifest] = {
    "sqr": OpManifest(
        source_dir="SQR",
        source_stem="sqr",
        placeholders=["WIDTH", "HEIGHT"],
        header_comment="element-wise FP16 square (dst = src * src)",
    ),
    "sum_rows": OpManifest(
        source_dir="SUM_ROWS",
        source_stem="sum_rows",
        placeholders=["WIDTH", "HEIGHT"],
        header_comment="row-wise FP16 SUM_ROWS (dst[row] = sum(row))",
    ),
    "group_norm": OpManifest(
        source_dir="GROUP_NORM",
        source_stem="group_norm",
        placeholders=["WIDTH", "HEIGHT"],
        # GROUP_NORM kernel hard-codes eps as FP16 bits — host runner supplies
        # the actual eps (from ggml op_params[1]) and rewires the macro.
        extra_replacements=[
            (r"#define\s+FP16_EPS_1E_5\s+0x[0-9A-Fa-f]+",
             "#define FP16_EPS            {{EPS_FP16}}"),
            (r"\bFP16_EPS_1E_5\b", "FP16_EPS"),
        ],
        header_comment="GROUP_NORM (num_groups=1) over a 2D FP16 tensor",
    ),
    "norm": OpManifest(
        source_dir="NORM",
        source_stem="norm",
        placeholders=["WIDTH", "HEIGHT"],
        extra_replacements=[
            (r"#define\s+FP16_EPS_1E_NEG_5\s+0x[0-9A-Fa-f]+",
             "#define FP16_EPS             {{EPS_FP16}}"),
            (r"\bFP16_EPS_1E_NEG_5\b", "FP16_EPS"),
        ],
        header_comment="per-row layer NORM",
    ),
    "scale": OpManifest(
        source_dir="SCALE",
        source_stem="scale",
        placeholders=["WIDTH", "HEIGHT"],
        header_comment="element-wise scale (dst = src * scale); scale at DDR BASE_DDR_B",
    ),
    "ceil": OpManifest(
        source_dir="CEIL",
        source_stem="ceil",
        placeholders=["WIDTH", "HEIGHT"],
        header_comment="element-wise ceil = -floor(-x); per-row tile",
    ),
    "expm1": OpManifest(
        source_dir="EXPM1",
        source_stem="expm1",
        placeholders=["WIDTH", "HEIGHT"],
        header_comment="exp(x) - 1; per-row, requires HEIGHT % 16 == 0",
    ),
    "clamp": OpManifest(
        source_dir="CLAMP",
        source_stem="clamp",
        placeholders=["WIDTH", "HEIGHT"],
        header_comment="clamp(x, min, max); min/max as 2 FP16 at DDR BASE_DDR_B",
    ),
    "mean": OpManifest(
        source_dir="MEAN",
        source_stem="mean",
        placeholders=["WIDTH", "HEIGHT"],
        # MEAN kernel hard-codes 1/WIDTH as `FP16_INV8 = 0x2C00`; runner supplies
        # the actual inverse for the current WIDTH and rewires the macro name.
        extra_replacements=[
            (r"#define\s+FP16_INV8\s+0x[0-9A-Fa-f]+[^\n]*",
             "#define FP16_INV_W           {{INV_W_FP16}}"),
            (r"\bFP16_INV8\b", "FP16_INV_W"),
        ],
        header_comment="row-wise mean = sum/WIDTH; HEIGHT % 16 == 0",
    ),
    "sum": OpManifest(
        source_dir="SUM",
        source_stem="sum",
        placeholders=["WIDTH", "HEIGHT"],
        header_comment="total reduction; output = 32 bytes (1 fp16 sum + 15 pad)",
    ),
    "arange": OpManifest(
        source_dir="ARANGE",
        source_stem="arange",
        placeholders=["ROWS", "COLS"],
        header_comment="dst[i] = i (start=0, step=1 only); N = ROWS*COLS fp16",
    ),
    "tri": OpManifest(
        source_dir="TRI",
        source_stem="tri",
        # TRI hardcodes default width/height/tri_type as locals in main();
        # the kernel still reads OP_PARAMS DDR via magic, but if the host
        # never writes that section the defaults — which we substitute — win.
        placeholders=[],
        extra_replacements=[
            (r"uint32_t\s+width\s*=\s*8u;",
             "uint32_t width = {{WIDTH}}u;"),
            (r"uint32_t\s+height\s*=\s*8u;",
             "uint32_t height = {{HEIGHT}}u;"),
            (r"uint32_t\s+tri_type\s*=\s*TRI_UPPER_DIAG;",
             "uint32_t tri_type = {{TRI_TYPE}}u;"),
        ],
        header_comment="triangular mask; W/H/tri_type dynamic, copy+fill only",
    ),
    "pad": OpManifest(
        source_dir="PAD",
        source_stem="pad",
        placeholders=["SRC_ROWS", "SRC_COLS", "PAD_RIGHT", "PAD_BOTTOM"],
        header_comment="pad with zeros along right/bottom; CHANNELS=1",
    ),
    "concat": OpManifest(
        source_dir="CONCAT",
        source_stem="concat",
        placeholders=["SRC_COLS", "ROWS"],
        header_comment="concat axis=0: dst row = [src0 row | src1 row]; 2 inputs",
    ),
    "im2col": OpManifest(
        source_dir="IM2COL",
        source_stem="im2col",
        placeholders=["IN_W", "IN_H", "K_W", "K_H", "STRIDE"],
        header_comment="im2col 2D: input → (OH*OW, KH*KW) patches; stride/IC=1 only",
    ),
    "pool_2d_avg": OpManifest(
        source_dir="POOL_2D",
        source_stem="pool_2d",
        placeholders=["IN_H", "IN_W", "OUT_H", "OUT_W", "K_H", "K_W", "S_H", "S_W"],
        # FP16_QUARTER (= 1/(KH*KW) fp16) is hard-coded for k=2x2; rename to
        # FP16_INV_K and let the runner inject the actual reciprocal bits.
        extra_replacements=[
            (r"#define\s+FP16_QUARTER\s+0x[0-9A-Fa-f]+[^\n]*",
             "#define FP16_INV_K          {{INV_K_FP16}}"),
            (r"\bFP16_QUARTER\b", "FP16_INV_K"),
        ],
        template_name="unary_pool_2d_avg",
        header_comment="average pool 2d; dynamic shape + inv(K_H*K_W) fp16 fill",
    ),
}


_DEFINE_RE = re.compile(
    r"(#define\s+{name}\s+)([^\s/]+)(.*)",
)


def _replace_define(src: str, name: str) -> str:
    """Swap `#define <name> <value>` to `#define <name> {{<name>}}`."""
    pat = re.compile(rf"(#define\s+{re.escape(name)}\s+)([^\s/]+)(.*)")
    if not pat.search(src):
        raise ValueError(f"no `#define {name} ...` found in source")
    return pat.sub(rf"\g<1>{{{{{name}}}}}\g<3>", src, count=1)


def _strip_ifndef_guard(src: str, name: str) -> str:
    """Remove `#ifndef NAME / #define NAME val / #endif` guards left around the
    bare `#define`. Some test/ kernels keep both — once we substitute a
    placeholder into the inner define, the guard is dead weight.
    Keep the guard if the inner define isn't the one we substituted.
    """
    guard_re = re.compile(
        rf"#ifndef\s+{re.escape(name)}\s*\n"
        rf"(#define\s+{re.escape(name)}\s+[^\n]+\n)"
        rf"#endif\s*\n",
        re.MULTILINE,
    )
    return guard_re.sub(r"\g<1>", src)


def _generate(op_key: str) -> str:
    m = OP_MANIFEST[op_key]
    if not m.src_path.is_file():
        raise FileNotFoundError(f"missing kernel source {m.src_path}")
    src = m.src_path.read_text()

    for name in m.placeholders:
        src = _strip_ifndef_guard(src, name)
        src = _replace_define(src, name)

    for pat, repl in m.extra_replacements:
        new_src = re.sub(pat, repl, src)
        if new_src == src:
            raise ValueError(
                f"extra_replacements pattern did not match for {op_key}: {pat!r}")
        src = new_src

    # Rewrite the leading n1s16-style header banner to declare this as generated
    # firmware. Tolerate kernels with or without the comment block.
    header = _make_header(op_key, m)
    src = re.sub(
        r"^//=+\n(?://[^\n]*\n)+//=+\n",
        header,
        src,
        count=1,
        flags=re.MULTILINE,
    )
    return src


def _make_header(op_key: str, m: OpManifest) -> str:
    name = m.template_name or f"unary_{m.source_stem}"
    blurb = m.header_comment or f"generated kernel for {op_key}"
    return (
        "//==================================================================\n"
        f"// {{{{OP_NAME}}}} (generated) — {blurb}\n"
        f"// Source: test/{m.source_dir}/n1s16/n1s16_{m.source_stem}.c\n"
        f"// Template name: {name}.c.tpl\n"
        "//==================================================================\n"
    )


def _diff(op_key: str, generated: str) -> str:
    m = OP_MANIFEST[op_key]
    current = m.out_path.read_text() if m.out_path.is_file() else ""
    diff = difflib.unified_diff(
        current.splitlines(keepends=True),
        generated.splitlines(keepends=True),
        fromfile=str(m.out_path),
        tofile=f"<generated:{op_key}>",
    )
    return "".join(diff)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--op", help="generate a single op by manifest key")
    p.add_argument("--diff", action="store_true",
                   help="print diff against existing .tpl instead of writing")
    p.add_argument("--list", action="store_true",
                   help="list manifest keys")
    args = p.parse_args(argv)

    if args.list:
        for k, m in OP_MANIFEST.items():
            print(f"{k:20s} -> {m.out_path.relative_to(REPO_ROOT)}")
        return 0

    keys = [args.op] if args.op else list(OP_MANIFEST)
    rc = 0
    for k in keys:
        if k not in OP_MANIFEST:
            print(f"error: unknown op '{k}' (see --list)", file=sys.stderr)
            return 2
        try:
            generated = _generate(k)
        except Exception as e:
            print(f"{k}: FAIL ({e})", file=sys.stderr)
            rc = 1
            continue
        if args.diff:
            d = _diff(k, generated)
            if d:
                print(d)
                print(f"--- {k}: differs ---")
            else:
                print(f"{k}: matches existing .tpl")
            continue
        out = OP_MANIFEST[k].out_path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(generated)
        print(f"{k}: wrote {out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
