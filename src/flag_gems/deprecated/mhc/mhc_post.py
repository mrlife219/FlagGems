# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Triton implementation of mHC Post operator (optimized v3).

Computes:
    out[n, i, h] = post_layer_mix[n, i] * x[n, h]
                 + sum_j(comb_res_mix[n, j, i] * residual[n, j, h])

Key optimizations (v3):
- 2D grid = (N, cdiv(H, BLOCK_H)): high program count for latency hiding.
- @triton.autotune over BLOCK_H / num_warps / num_stages.
- Contiguous layout: stride math removed, enabling LDG.128.
- All 4 accumulators computed then stored (better ILP).
- BLOCK_H chosen to evenly divide H when possible (256 divides all targets).
- TLE-DSA fast path (opt-in, FLAGGEMS_ASCEND_TLE_MHC_POST=1, 910B 2.89x,
  bandwidth-saturated): see the TLE block below the imports.
"""

import logging
import os

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TLE-DSA fast path (opt-in via env, default OFF = byte-identical baseline).
#
# Evidence (910B standalone, 2026-11-07, N=16384 H=3584 bf16, rep=100 +
# loop=200 cross-checked, perf/bench/mhc_post_tle_candidate.py):
#   prod autotune 3.250 ms / 325.2 GB/s -> TLE bh=2048 1.123 ms / 941.3 GB/s
#   (2.89x, bandwidth-saturated at the RmsNorm ~940 GB/s ceiling);
#   err vs torch ref identical to prod (0.0625, bf16 rounding level).
#
# Kernel design: bulk GMEM->UB MTE2 copies for all 5 input streams
# (tle.dsa.copy), fp32 tensor math in UB (tle.dsa.to_tensor), 4 masked
# stores. bf16 dsa elementwise ops (tle.dsa.add/mul/...) are NOT used:
# CANN 8.5.0 HIVMTraits whitelist rejects BF16 (AGENTS 3.4).
#
# Known constraints (probe v1/v2, 2026-11-07):
#   - tle.dsa.copy size MUST be clamped: sz = tl.minimum(BLOCK_H, H - pid_h*BLOCK_H);
#     without the clamp the first tile copies H elements into a BLOCK_H-sized UB
#     buffer -> overflow (err 24 vs 0.03 fixed).
#   - bh=4096 exceeds the 192KB UB after multi-buffer amplification (needs
#     ~208KB) -> BLOCK_H is FIXED at 2048, deliberately NOT autotuned.
#   - capability probe is by import, NOT triton version number (flagtree 3.2
#     backports TLE while upstream gates it behind 3.6).
# ---------------------------------------------------------------------------
_tle = None
if os.environ.get("FLAGGEMS_ASCEND_TLE_MHC_POST", "0") == "1":
    try:
        import triton.experimental.tle.language as _tle
    except BaseException:  # noqa: BLE001  (module missing in this triton build)
        _tle = None
    else:
        if not (hasattr(_tle, "dsa") and hasattr(_tle.dsa, "ascend")):
            _tle = None

_TLE_MHC_POST_BLOCK_H = 2048

# Config-lottery guard (2026-11-12): upstream @triton.autotune keys on ["H"]
# only and N never enters the kernel signature — a cold-start process
# benchmarks the 12 configs ONCE on its first (usually small) shape and locks
# that config for every N at this H. Three independent production windows
# matched forced-pin reproductions to <=0.6%: 256/4/1 (0853 run: 8044.5/
# 6035.2/2009.2 us on 16384/12288/4096 = 2.47x-slow), 512/4/2 (11-05 run:
# 4691 us), 128-family (11-12 ascending sweep: 14.5 ms, 4.4x artifact).
# 1024/4/1 is the measured optimum for every prefill shape (graph k 3258/
# 2447/826 us, matches the per-N fresh-process autotuned pick exactly) and
# still beats the tuner's own pick at N=8 (41.9 vs 57.4 us).
_MHC_POST_PIN_MIN_N = 4096
_MHC_POST_PIN_CONFIG = triton.Config(
    {"BLOCK_H": 1024}, num_warps=4, num_stages=1
)


@triton.autotune(
    configs=[
        # Small BLOCK_H: many programs, good for latency hiding
        triton.Config({"BLOCK_H": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 256}, num_warps=8, num_stages=2),
        # Medium BLOCK_H
        triton.Config({"BLOCK_H": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 512}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 512}, num_warps=8, num_stages=2),
        # Large BLOCK_H
        triton.Config({"BLOCK_H": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 1024}, num_warps=8, num_stages=1),
    ],
    key=["H"],
)
@triton.jit
def mhc_post_kernel_hc_mult_4(
    a_ptr,  # comb_res_mix : (N, 4, 4), float32 — a[n, j, i]
    b_ptr,  # residual     : (N, 4, H), bfloat16
    c_ptr,  # post_layer_mix: (N, 4),   float32
    d_ptr,  # x            : (N, H),    bfloat16
    out_ptr,  # output       : (N, 4, H), bfloat16
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """
    Grid: (N, cdiv(H, BLOCK_H)).
    Each program handles one token × one h-tile × all 4 hc streams.
    """
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_off = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_off < H

    # ── pointer bases (contiguous layout) ──
    a_base = pid_n * 16  # (N, 4, 4) → stride_n = 16
    c_base = pid_n * 4  # (N, 4)    → stride_n = 4
    b_base = pid_n * 4 * H  # (N, 4, H) → stride_n = 4*H
    d_base = pid_n * H  # (N, H)    → stride_n = H
    out_base = pid_n * 4 * H  # (N, 4, H) → stride_n = 4*H

    # ── load 20 scalars (L1 cached across h-tiles) ──
    c0 = tl.load(c_ptr + c_base + 0).to(tl.float32)
    c1 = tl.load(c_ptr + c_base + 1).to(tl.float32)
    c2 = tl.load(c_ptr + c_base + 2).to(tl.float32)
    c3 = tl.load(c_ptr + c_base + 3).to(tl.float32)

    a00 = tl.load(a_ptr + a_base + 0).to(tl.float32)
    a01 = tl.load(a_ptr + a_base + 1).to(tl.float32)
    a02 = tl.load(a_ptr + a_base + 2).to(tl.float32)
    a03 = tl.load(a_ptr + a_base + 3).to(tl.float32)
    a10 = tl.load(a_ptr + a_base + 4).to(tl.float32)
    a11 = tl.load(a_ptr + a_base + 5).to(tl.float32)
    a12 = tl.load(a_ptr + a_base + 6).to(tl.float32)
    a13 = tl.load(a_ptr + a_base + 7).to(tl.float32)
    a20 = tl.load(a_ptr + a_base + 8).to(tl.float32)
    a21 = tl.load(a_ptr + a_base + 9).to(tl.float32)
    a22 = tl.load(a_ptr + a_base + 10).to(tl.float32)
    a23 = tl.load(a_ptr + a_base + 11).to(tl.float32)
    a30 = tl.load(a_ptr + a_base + 12).to(tl.float32)
    a31 = tl.load(a_ptr + a_base + 13).to(tl.float32)
    a32 = tl.load(a_ptr + a_base + 14).to(tl.float32)
    a33 = tl.load(a_ptr + a_base + 15).to(tl.float32)

    # ── load vectors (bf16 → f32) ──
    d_vals = tl.load(d_ptr + d_base + h_off, mask=h_mask, other=0.0).to(tl.float32)
    b0 = tl.load(b_ptr + b_base + 0 * H + h_off, mask=h_mask, other=0.0).to(tl.float32)
    b1 = tl.load(b_ptr + b_base + 1 * H + h_off, mask=h_mask, other=0.0).to(tl.float32)
    b2 = tl.load(b_ptr + b_base + 2 * H + h_off, mask=h_mask, other=0.0).to(tl.float32)
    b3 = tl.load(b_ptr + b_base + 3 * H + h_off, mask=h_mask, other=0.0).to(tl.float32)

    # ── compute all 4 output streams ──
    acc0 = c0 * d_vals + a00 * b0 + a10 * b1 + a20 * b2 + a30 * b3
    acc1 = c1 * d_vals + a01 * b0 + a11 * b1 + a21 * b2 + a31 * b3
    acc2 = c2 * d_vals + a02 * b0 + a12 * b1 + a22 * b2 + a32 * b3
    acc3 = c3 * d_vals + a03 * b0 + a13 * b1 + a23 * b2 + a33 * b3

    # ── store all 4 outputs ──
    tl.store(out_ptr + out_base + 0 * H + h_off, acc0.to(tl.bfloat16), mask=h_mask)
    tl.store(out_ptr + out_base + 1 * H + h_off, acc1.to(tl.bfloat16), mask=h_mask)
    tl.store(out_ptr + out_base + 2 * H + h_off, acc2.to(tl.bfloat16), mask=h_mask)
    tl.store(out_ptr + out_base + 3 * H + h_off, acc3.to(tl.bfloat16), mask=h_mask)


@triton.jit
def mhc_post_kernel_hc_mult_4_tle(
    a_ptr,  # comb_res_mix : (N, 4, 4), float32 — a[n, j, i]
    b_ptr,  # residual     : (N, 4, H), bfloat16
    c_ptr,  # post_layer_mix: (N, 4),   float32
    d_ptr,  # x            : (N, H),    bfloat16
    out_ptr,  # output       : (N, 4, H), bfloat16
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """TLE-DSA variant of the hc_mult=4 kernel.

    Bulk GMEM->UB MTE2 staging for all 5 input streams (tle.dsa.copy), fp32
    tensor math in UB (tle.dsa.to_tensor), 4 masked stores.
    Grid: (N, cdiv(H, BLOCK_H)); BLOCK_H is fixed by the wrapper (UB ceiling).
    """
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs < H
    # MANDATORY clamp: without it the first tile copies H elements into a
    # BLOCK_H-sized UB buffer -> overflow (probe v1: err 24 vs 0.03 when fixed).
    sz = tl.minimum(BLOCK_H, H - pid_h * BLOCK_H)

    # ── pointer bases (contiguous layout, same contract as hc_mult_4) ──
    a_base = pid_n * 16  # (N, 4, 4) → stride_n = 16
    c_base = pid_n * 4  # (N, 4)    → stride_n = 4
    b_base = pid_n * 4 * H  # (N, 4, H) → stride_n = 4*H
    d_base = pid_n * H  # (N, H)    → stride_n = H
    out_base = pid_n * 4 * H  # (N, 4, H) → stride_n = 4*H

    # ── 20 scalar coefficients (L1 cached across h-tiles) ──
    c0 = tl.load(c_ptr + c_base + 0).to(tl.float32)
    c1 = tl.load(c_ptr + c_base + 1).to(tl.float32)
    c2 = tl.load(c_ptr + c_base + 2).to(tl.float32)
    c3 = tl.load(c_ptr + c_base + 3).to(tl.float32)

    a00 = tl.load(a_ptr + a_base + 0).to(tl.float32)
    a01 = tl.load(a_ptr + a_base + 1).to(tl.float32)
    a02 = tl.load(a_ptr + a_base + 2).to(tl.float32)
    a03 = tl.load(a_ptr + a_base + 3).to(tl.float32)
    a10 = tl.load(a_ptr + a_base + 4).to(tl.float32)
    a11 = tl.load(a_ptr + a_base + 5).to(tl.float32)
    a12 = tl.load(a_ptr + a_base + 6).to(tl.float32)
    a13 = tl.load(a_ptr + a_base + 7).to(tl.float32)
    a20 = tl.load(a_ptr + a_base + 8).to(tl.float32)
    a21 = tl.load(a_ptr + a_base + 9).to(tl.float32)
    a22 = tl.load(a_ptr + a_base + 10).to(tl.float32)
    a23 = tl.load(a_ptr + a_base + 11).to(tl.float32)
    a30 = tl.load(a_ptr + a_base + 12).to(tl.float32)
    a31 = tl.load(a_ptr + a_base + 13).to(tl.float32)
    a32 = tl.load(a_ptr + a_base + 14).to(tl.float32)
    a33 = tl.load(a_ptr + a_base + 15).to(tl.float32)

    # ── UB staging: 1 x stream + 4 residual streams, bulk MTE2 copies ──
    ty = d_ptr.dtype.element_ty
    d_ub = _tle.dsa.alloc([BLOCK_H], ty, _tle.dsa.ascend.UB)
    b0_ub = _tle.dsa.alloc([BLOCK_H], ty, _tle.dsa.ascend.UB)
    b1_ub = _tle.dsa.alloc([BLOCK_H], ty, _tle.dsa.ascend.UB)
    b2_ub = _tle.dsa.alloc([BLOCK_H], ty, _tle.dsa.ascend.UB)
    b3_ub = _tle.dsa.alloc([BLOCK_H], ty, _tle.dsa.ascend.UB)

    _tle.dsa.copy(d_ptr + d_base + offs, d_ub, [sz])
    _tle.dsa.copy(b_ptr + b_base + 0 * H + offs, b0_ub, [sz])
    _tle.dsa.copy(b_ptr + b_base + 1 * H + offs, b1_ub, [sz])
    _tle.dsa.copy(b_ptr + b_base + 2 * H + offs, b2_ub, [sz])
    _tle.dsa.copy(b_ptr + b_base + 3 * H + offs, b3_ub, [sz])

    # ── compute in UB as fp32 tensors (tensor math, NOT dsa elementwise) ──
    d_vals = _tle.dsa.to_tensor(d_ub).to(tl.float32)
    b0 = _tle.dsa.to_tensor(b0_ub).to(tl.float32)
    b1 = _tle.dsa.to_tensor(b1_ub).to(tl.float32)
    b2 = _tle.dsa.to_tensor(b2_ub).to(tl.float32)
    b3 = _tle.dsa.to_tensor(b3_ub).to(tl.float32)

    acc0 = c0 * d_vals + a00 * b0 + a10 * b1 + a20 * b2 + a30 * b3
    acc1 = c1 * d_vals + a01 * b0 + a11 * b1 + a21 * b2 + a31 * b3
    acc2 = c2 * d_vals + a02 * b0 + a12 * b1 + a22 * b2 + a32 * b3
    acc3 = c3 * d_vals + a03 * b0 + a13 * b1 + a23 * b2 + a33 * b3

    # ── store all 4 outputs ──
    ot = out_ptr.dtype.element_ty
    tl.store(out_ptr + out_base + 0 * H + offs, acc0.to(ot), mask=mask)
    tl.store(out_ptr + out_base + 1 * H + offs, acc1.to(ot), mask=mask)
    tl.store(out_ptr + out_base + 2 * H + offs, acc2.to(ot), mask=mask)
    tl.store(out_ptr + out_base + 3 * H + offs, acc3.to(ot), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 512}, num_warps=8, num_stages=1),
    ],
    key=["H", "HC"],
)
@triton.jit
def mhc_post_kernel_generic(
    a_ptr,  # comb_res_mix : (N, HC, HC), float32
    b_ptr,  # residual     : (N, HC, H), bfloat16
    c_ptr,  # post_layer_mix: (N, HC), float32
    d_ptr,  # x            : (N, H), bfloat16
    out_ptr,  # output      : (N, HC, H), bfloat16
    H: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Generic mHC post kernel for arbitrary HC.

    Grid: (N, HC, cdiv(H, BLOCK_H)).
    Each program handles one token × one output-stream(i) × one h-tile.
    """
    pid_n = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_h = tl.program_id(2)

    h_off = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_off < H

    a_base = pid_n * HC * HC
    b_base = pid_n * HC * H
    c_base = pid_n * HC
    d_base = pid_n * H
    out_base = pid_n * HC * H + pid_i * H

    d_vals = tl.load(d_ptr + d_base + h_off, mask=h_mask, other=0.0).to(tl.float32)
    c_i = tl.load(c_ptr + c_base + pid_i).to(tl.float32)

    acc = c_i * d_vals
    for j in tl.static_range(0, HC):
        a_ji = tl.load(a_ptr + a_base + j * HC + pid_i).to(tl.float32)
        b_j = tl.load(b_ptr + b_base + j * H + h_off, mask=h_mask, other=0.0).to(
            tl.float32
        )
        acc += a_ji * b_j

    tl.store(out_ptr + out_base + h_off, acc.to(tl.bfloat16), mask=h_mask)


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """
    mHC post-processing operator.

    Args:
        x: (N, H), bfloat16 — layer output
        residual: (N, hc_mult, H), bfloat16 — multi-head residual
        post_layer_mix: (N, hc_mult, 1), float32 — per-stream scale for x
        comb_res_mix: (N, hc_mult, hc_mult), float32 — combination matrix

    Returns:
        out: (N, hc_mult, H), bfloat16
    """
    logger.debug(
        "GEMS MHC_POST FORWARD, x=%s, residual=%s, post_layer_mix=%s, comb_res_mix=%s",
        x.shape,
        residual.shape,
        post_layer_mix.shape,
        comb_res_mix.shape,
    )

    N, hc, H = residual.shape
    assert x.shape == (N, H)
    assert post_layer_mix.shape in ((N, hc, 1), (N, hc))
    assert comb_res_mix.shape == (N, hc, hc)

    out = torch.empty_like(residual)

    c = post_layer_mix.squeeze(-1).contiguous()  # (N, hc)
    a = comb_res_mix.contiguous()  # (N, hc, hc)
    b = residual.contiguous()  # (N, hc, H)
    d = x.contiguous()  # (N, H)

    if hc == 4:
        if (
            _tle is not None
            and x.dtype == torch.bfloat16
            and residual.dtype == torch.bfloat16
        ):
            # TLE-DSA fast path. BLOCK_H fixed at 2048: bh=4096 exceeds the
            # 192KB UB after multi-buffer amplification (see module header).
            # Inputs are contiguous by construction (the .contiguous() calls
            # above); bf16-only guard keeps untested dtype classes off (AGENTS
            # 4.2 rule 5: conclusions do not extrapolate to untested classes).
            grid_tle = (N, triton.cdiv(H, _TLE_MHC_POST_BLOCK_H))
            mhc_post_kernel_hc_mult_4_tle[grid_tle](
                a,
                b,
                c,
                d,
                out,
                H=H,
                BLOCK_H=_TLE_MHC_POST_BLOCK_H,
            )
            return out

        if (
            N >= _MHC_POST_PIN_MIN_N
            and x.dtype == torch.bfloat16
            and residual.dtype == torch.bfloat16
        ):
            # Collapse the tuner to the single measured-optimal config. A
            # single-config autotuner skips BOTH benchmarking and the stale
            # shape-keyed cache (triton Autotuner.run gates on
            # len(self.configs) > 1), so the pinned config is used regardless
            # of what a previous small-shape call already locked in.
            if len(mhc_post_kernel_hc_mult_4.configs) != 1:
                mhc_post_kernel_hc_mult_4.configs = [_MHC_POST_PIN_CONFIG]

        def grid_specialized(META):
            return (N, triton.cdiv(H, META["BLOCK_H"]))

        mhc_post_kernel_hc_mult_4[grid_specialized](
            a,
            b,
            c,
            d,
            out,
            H=H,
        )
    else:

        def grid_generic(META):
            return (N, hc, triton.cdiv(H, META["BLOCK_H"]))

        mhc_post_kernel_generic[grid_generic](
            a,
            b,
            c,
            d,
            out,
            H=H,
            HC=hc,
        )
    return out


def mhc_post_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """PyTorch reference implementation."""
    y = x.unsqueeze(-2) * post_layer_mix + torch.bmm(comb_res_mix.mT, residual.float())
    return y.type_as(x)
