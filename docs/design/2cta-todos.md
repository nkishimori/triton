# 2-CTA autoWS: Open TODOs and Review Feedback

**PR:** #1120 (rebased as PR from `jananisriram/2cta-autows-rebase`)
**Phabricator:** D104758163
**Reviewers:** manman-ren, njriasan, pchen7e4

**NOTE:** When any TODO is resolved with a concrete implementation change, update
the main design doc (`docs/design/2cta-autoWS-sync.md`) to reflect the change.
The design doc should always match the current implementation so it remains useful
as a reference for future work.

---

## TODO 1: Consistent 2-CTA detection across the codebase

**Problem:** The codebase uses multiple fragmented conditions to detect 2-CTA mode:
- `getModuleTwoCTAs(op)` — checks `ttng.two-ctas` module attribute
- `getNumCTAs() == 2` — checks `ttg.num-ctas` module attribute
- `cluster-dim-x >= 2` — checks cluster dimensions
- `op.getTwoCtas()` — checks per-op `two_ctas` attribute (on TCGen5MMAOp)
- `tlx::tlxEnablePairedMMA(op)` — TLX-specific paired CTA check

With the `ctas_per_cga` approach, `num_ctas=1` is set automatically, so any code
checking `num_ctas == 2` will MISS our 2-CTA kernels. We need a single, authoritative
way to detect "this module is in 2-CTA mode."

**Nick's concern (line 74):** "I think we need to heavily guard against this case.
This seems like the prime circumstances for an upstream cherry-pick to break everything."

**Proposed solution:** Create a unified helper function:

```cpp
// In include/triton/Dialect/TritonNvidiaGPU/IR/Dialect.h or similar shared header
inline bool isModule2CTA(ModuleOp mod) {
    // Check both paths:
    // 1. PlanCTA path: num_ctas == 2
    // 2. ctas_per_cga path: cluster-dim-x >= 2 with num_ctas == 1
    if (getModuleTwoCTAs(mod))
        return true;
    // Check cluster dimensions
    if (auto attr = mod->getAttrOfType<IntegerAttr>("ttg.cluster-dim-x"))
        if (attr.getInt() >= 2)
            return true;
    return false;
}
```

**Action items:**
- [ ] Audit every 2-CTA check site (see list below) — future cleanup
- [ ] Create unified `isModule2CTA()` helper — future cleanup
- [ ] Replace fragmented checks with the helper where appropriate — future cleanup
- [x] Add compile-time validation: if `ctas_per_cga` is set, assert `num_ctas == 1`
      — already enforced in `CUDAOptions.__post_init__` (compiler.py:197)

### 2-CTA check sites (to audit)

The 2-CTA detection is **highly fragmented** across 6 different mechanisms:

| Mechanism | Where checked | What it means |
|---|---|---|
| `DotOp::getTwoCtas()` | AccelerateMatmul | User-driven 2-CTA from `ctas_per_cga` at TTIR level |
| `TCGen5MMAOp::getTwoCtas()` | CheckMatmulTwoCTAs, Insert2CTASync, Transform2CTALoads, MMAv5 lowering | Per-op flag at TTGIR level |
| `getModuleTwoCTAs()` / `AttrTwoCTAsName` | MMAv5 LLVM lowering (3 call sites) | Module-level bool, set by CheckMatmulTwoCTAs pass |
| `cluster-dim-{x,y,z} >= 2` | Insert2CTASync, Transform2CTALoads, compiler.py pipeline | Physical cluster configuration |
| `num_ctas == 1` | PlanCTA (skip), compiler.py (`ctas_per_cga` forces this) | Distinguishes PlanCTA path vs `ctas_per_cga` path |
| `ctas_per_cga` Python option | compiler.py pipeline decisions | User-facing API |

**Flow:** `ctas_per_cga` (Python) → sets `cluster_dims` + forces `num_ctas=1` →
`DotOp.two_ctas` (TTIR) → `TCGen5MMAOp.two_ctas` (TTGIR, via AccelerateMatmul) →
`CheckMatmulTwoCTAs` sets module `"ttng.two-ctas"` → `getModuleTwoCTAs()` in LLVM lowering.

#### All check sites

**Module attribute set:**
- `CheckMatmulTwoCTAs.cpp:64` — walks all MMA ops, sets `ttng.two-ctas` on module

**Module attribute read (`getModuleTwoCTAs()`):**
- `MMAv5.cpp:547` — `convertDot()`: gates `cta_group::2` PTX, B halving, cluster sync
- `MMAv5.cpp:667` — `convertScaledDot()`: same for scaled MMA
- `MMAv5.cpp:771` — `TCGen5CommitOpConversion`: gates `cta_group::2` on commit + CTA0 predicate

**Per-op `getTwoCtas()`:**
- `AccelerateMatmul.cpp:559` — on DotOp: skip `splitBOperand`, set `two_ctas` on MMA
- `AccelerateMatmul.cpp:607` — `mma.setTwoCtas(useTwoCTAs)` propagation
- `CheckMatmulTwoCTAs.cpp:52-54` — validates all MMA ops agree
- `Insert2CTASync.cpp:151,294` — collects MMAs needing cross-CTA sync
- `Transform2CTALoads.cpp:118` — collects non-async 2-CTA MMAs for B splitting
- `MMAv5.cpp:703` — doubles M dimension for scaled 2-CTA instruction descriptor

**Cluster dim checks:**
- `Insert2CTASync.cpp:131-141` — guard: early return if no cluster dim >= 2
- `Insert2CTASync.cpp:276-284` — same in `doInsert2CTASync()`
- `Transform2CTALoads.cpp:103-113` — guard: early return if no cluster
- `compiler.py:645-651` — enable: add `Transform2CTALoads` pass
- `compiler.py:703-706` — guard: skip upstream WS when cluster >= 2
- `compiler.py:729` — enable: add `Insert2CTASync` (Meta WS only)

**PTX-level (`cta_group::2`):**
- `MMAv5.cpp:241` — `createGen5MMA()`: `cta_group::2` on MMA instruction
- `MMAv5.cpp:272` — `createScaledGen5MMA()`: same for scaled
- `MMAv5.cpp:330` — `createMMACommit()`: `cta_group::2` on commit
- `MMAv5.cpp:309-321` — multicast mask with `broadcastBits |= 1`
- `MMAv5.cpp:400-422` — CTA sync + leader-only predicate
- `MMAv5.cpp:460-461` — assert no N-dimension repetition in 2-CTA
- `MMAv5.cpp:472-473` — halve B operand shape
- `MMAv5.cpp:582-584` — double `mmaSizeM` for instruction descriptor

**TMEM / exit barrier:**
- `NVGPUToLLVMPass.cpp` — TMEM `cta_group::2` when `cluster-dim-x >= 2`
- `ConvertWarpSpecializeToLLVM.cpp` — exit cluster barrier for `cluster-dim-x >= 2`

---

## TODO 2: Error if `two_ctas=True` without `ctas_per_cga`

**Nick (line 102):** "It feels awkward to allow setting `two_ctas=True` if you can't
actually set the two CTAs. We should make it clear that if `two_ctas=True` it's an
error unless you have `ctas_per_cga=(2, 1, 1)`."

**Action items:**
- [x] Add validation in AccelerateMatmul.cpp — DONE. Emits error when
      `two_ctas=True` but `cluster-dim-x < 2`. The check:
      ```cpp
      auto module = dotOp->getParentOfType<ModuleOp>();
      auto clusterDims = TritonGPUDialect::getClusterDims(module);
      if (clusterDims[0] < 2)
        return dotOp.emitError("two_ctas=True requires ctas_per_cga=(2,1,1)");
      ```
      Python-level validation is not feasible — `dot()` in core.py has no access
      to launch options (`ctas_per_cga`). `semantic.py` could check via
      `self.builder.options`, but AccelerateMatmul is more robust.
- [ ] Update design doc examples to use constexprs (Nick: "agents will learn from
      our examples so we should use explicit examples with constexprs")
- [ ] Document the exact supported arg combinations explicitly:
      - SUPPORTED: `two_ctas=True` + `ctas_per_cga=(2,1,1)` + `num_ctas=1`
      - ERROR: `two_ctas=True` without `ctas_per_cga`
      - FUTURE: `two_ctas=True` + `ctas_per_cga=(4,1,1)` or other even-X CGA
- [x] Confirm at least 1 test with `num_ctas=1` + `cluster-dim-x=2` (Nick, line 449):
      CONFIRMED — `transform_2cta_loads.mlir`, `insert_2cta_sync.mlir`, and
      `blackwell-triton-addmm-2cta_test.py` all use this configuration

---

## TODO 3: Gate `Transform2CTALoads` on `use_meta_ws`

**manman + nick (compiler.py:635):** "If 2cta support only works with Meta's autoWS,
maybe we should check `knobs.nvidia.use_meta_ws` here as well."

**Current code (compiler.py:634-636):**
```python
# 2-CTA: Split B descriptor loads before optimize_descriptor_encoding
# so the cloned half-width descriptor gets its encoding set properly.
if (capability // 10 >= 10 and opt.cluster_dims is not None and max(opt.cluster_dims) >= 2
        and opt.ctas_per_cga is not None):
    nvidia.passes.hopper.add_2cta_transform_loads(pm)
```

`Insert2CTASync` (line 729) is already gated on `use_meta_ws`. `Transform2CTALoads`
is not — so B loads get split even without Meta WS, but the cross-CTA sync never
gets inserted, likely causing incorrect results.

**Fix:** Add `knobs.nvidia.use_meta_ws` to the condition:
```python
if (capability // 10 >= 10 and opt.cluster_dims is not None and max(opt.cluster_dims) >= 2
        and opt.ctas_per_cga is not None and knobs.nvidia.use_meta_ws):
```

**Resolution:** `Transform2CTALoads` must NOT be gated on `use_meta_ws`. The
`ctas_per_cga` approach sets `num_ctas=1`, bypassing PlanCTA entirely.
`Transform2CTALoads` is the **only** B splitting path for `ctas_per_cga` kernels —
both WS and non-WS. Cross-CTA sync is handled separately:
- Meta WS: `Insert2CTASync` (gated on `use_meta_ws`)
- Non-WS: MMAv5.cpp's inline `ClusterArriveOp`

**Action items:**
- [x] Confirmed: `Transform2CTALoads` cannot be gated on `use_meta_ws`
- [ ] **Respond to manman/nick:** Explain that only `Insert2CTASync` should be
      gated, not `Transform2CTALoads`

---

## TODO 4: `useTwoCTAs` vs `TensorMemoryCTAMode` in AccelerateMatmul

**pchen (AccelerateMatmul.cpp:588):** "If our auto 2cta follows the approach of TLX,
maybe it's better to set `useTwoCTAs` as false here, as that has indications in
backend codegen which we're not depending on. We solved the same problem more
thoroughly by introducing `TensorMemoryCTAMode` instead, which differentiates LHS
and RHS."

**manman (AccelerateMatmul.cpp:588):** "I think it is okay to set useTwoCTAs True
for Meta's autoWS + 2cta. But want to confirm with @pchen7e4."

**Investigation findings:**

`useTwoCTAs=true` is **required** for autoWS. The passes are complementary:

| Concern | Transform2CTALoads | Insert2CTASync | MMAv5.cpp (`twoCTAs`) |
|---------|-------------------|----------------|----------------------|
| B splitting (IR) | Yes | No | Expects it (`bOperandShape /= 2`) |
| Cross-CTA sync | No | Yes (mbarrier) | Fallback (ClusterArrive, skipped in WS) |
| `cta_group::2` PTX | No | No | **Yes — required** |
| M doubling in descriptor | No | No | **Yes — required** |
| CTA0 predicate on MMA | No | No | **Yes — required** |
| Multicast mask on commit | No | No | **Yes — required** |

Setting `useTwoCTAs=false` would emit `cta_group::1` and break 2-CTA entirely.

`TensorMemoryCTAMode` is orthogonal — it only affects TMEM LinearLayout for the
m=64 hardware quirk. It has zero effect on MMAv5.cpp codegen. It is NOT a
replacement for `useTwoCTAs`. Both are needed for m=64 2-CTA; only `useTwoCTAs`
is needed for m=128 2-CTA.

**Team answer (Peng):** "I was specifically talking about only
TensorMemoryEncodingAttr. In our approach we should not construct it with
useTwoCTA as True as that changes how TensorMemory is modeled."

**Resolution:** Two separate things:
- `TCGen5MMAOp.two_ctas` (the op attribute) = **must be true** for `cta_group::2` PTX
- `TensorMemoryEncodingAttr.ctaMode` = **must stay DEFAULT** (not TwoCTA_LHS/RHS)

Current code is correct: `useTwoCTAs=true` on MMA op, `DEFAULT` ctaMode on TMEM.

**Action items:**
- [x] Confirmed with Peng: TMEM encoding stays DEFAULT, MMA op two_ctas stays true
- [ ] For m=64 support (future): will need `TensorMemoryCTAMode` TwoCTA_LHS/RHS

---

## TODO 5: TCGen5CommitOp and WS interaction

**Context (design doc limitation #2):** Multiple MMAs per loop iteration share a
single cross-CTA barrier, which could cause phase conflicts.

**manman (line 358):** "I wonder if we should check for this case and bail out with
an error message if this is not properly handled."

**pchen (line 358):** "mma op itself does not really connect with mbar. It's the
tcgen05 commit following each single (or a few) mma ops that really connects the
prior mma to mbar. If here we rely on MMA lowering which generates a commit
following each single mma, it could be problematic."

**Nick (line 358):** "We support generating the commit, but typically we only do
this when our channel communication is for several MMA ops (e.g. outside a loop)."

**Core question for the team:** Does autoWS's commit insertion strategy (in
WSCodePartition) align with 2-CTA's cross-CTA sync requirements? Specifically:
- AutoWS generates `TCGen5CommitOp` to signal gemm→epilogue completion. Does it
  generate one commit per MMA, or one commit for a batch of MMAs?
- If per-MMA: does `Insert2CTASync`'s single barrier correctly pair with each commit?
- If batched: does `Insert2CTASync`'s barrier need to account for the batch?
- For multiple MMAs per loop iteration (e.g., FA with 2 dots): should we detect
  this and bail out, or can we support it?

**Investigation findings:**
- WSCodePartition creates **one TCGen5CommitOp per channel group** (not per MMA).
  Multiple channels can share one barrier via `groupChannels()`. After creation,
  `fuseTcgen05CommitBarriers()` merges consecutive commits with compatible consumers.
- The commit is placed after the nested region containing the MMA, not inline.
- Insert2CTASync creates **completely separate barriers** from WS pipeline barriers.
  It does NOT reuse WS barriers. The cross-CTA barrier has `arriveCount=2`.

**Team answer (Manman):** Limit to 1 GEMM per loop first. FA support is future work.

**Action items:**
- [x] Document how autoWS handles TCGen5CommitOp creation — DONE
- [ ] Add bail-out guard for multi-MMA-per-iteration (FA case = future work)
- [ ] Verify commit op has correct `cta_group` in all WS paths

---

## TODO 6: B-empty barrier synchronization in WS

**Nick (line 181):** "I wonder about the potential implications on WS barriers.
Consider the b_empty barrier in AutoWS. In that setup we signal the B barrier as
soon as the MMA finishes. However since B is in the other CTA I'd like to ensure
we explicitly call out the synchronization requirements for how we can label B as done.
There may be a similar issue if A/B opt to share a barrier."

### The problem

In 1-CTA WS, the pipeline for B is:
1. Producer loads B into local SMEM → arrives on `b_full`
2. Consumer waits `b_full` → MMA reads B from SMEM
3. MMA finishes → consumer arrives on `b_empty` (producer can overwrite)

In 2-CTA mode, CTA 1 loads its half of B into **CTA 1's SMEM**. The MMA
(`cta_group::2`) reads A from CTA 0's SMEM and B from both CTAs' SMEM.

**Key questions:**
1. When the MMA finishes (signaled by `tcgen05.commit`), does that guarantee
   both CTAs' B SMEM has been fully consumed? If yes, a local `b_empty` arrive
   in the consumer (CTA 0) is sufficient — but CTA 1's producer also needs to
   know its B SMEM is free. How does CTA 1's producer learn this?
2. Are `b_full` / `b_empty` barriers local (per-CTA) or cluster-wide? If local,
   CTA 1's producer never sees CTA 0's `b_empty` arrive.
3. If A and B share a barrier (common WS optimization), the `b_empty` signal
   also gates A's next load. In 2-CTA, A is only in CTA 0 — does the shared
   barrier still work correctly when B is split across CTAs?

### Investigation findings

**Barrier architecture:** WS barriers are **per-CTA local SMEM barriers**. Each CTA
has its own barrier slots. Cross-CTA coordination is handled entirely by
`Insert2CTASync`'s separate barrier (arriveCount=2), which is independent from WS
pipeline barriers.

**B-empty signaling:** In WS, the B operand uses either a token (CreateTokenOp) or
an MMA inline barrier (consumerBarrier) for producer-acquire/consumer-release. These
are per-CTA — CTA 1's producer waits on CTA 1's local token/barrier. Each CTA
independently manages its own half of B through its own local WS pipeline.

**A/B shared barriers:** Yes, A and B can share barriers via `groupChannels()`. If
both loads feed the same MMA with the same taskIds, they merge into one CommChannel.
`BarrierExpectOp` sums expected bytes from both TMA loads. In 2-CTA mode, this
merging still happens — but since B is half-width per CTA (via Transform2CTALoads),
the expected byte count reflects the half-width load.

**Insert2CTASync is fully independent:** It creates its own `LocalAllocOp` +
`InitBarrierOp` with `arriveCount=2`. Both CTAs arrive on the leader's barrier via
`MapToRemoteBufferOp`, only the leader waits. This ensures both CTAs' B loads are
complete before the MMA. This is separate from the WS pipeline's B-full/B-empty
protocol.

### Remaining questions for the team

- [ ] **Confirm:** `tcgen05.commit.cta_group::2` guarantees both CTAs' SMEM inputs
      are consumed (so the per-CTA B-empty signaling is correct)?
- [ ] **Confirm:** When A/B share a barrier in 2-CTA mode, the half-width B
      `BarrierExpectOp` byte count is correct? (Each CTA loads half of B, so the
      expected bytes should reflect half-width, not full-width)
- [ ] Add explicit sync rules to the design doc (pending answers above)

---

## TODO 7: Host-side TMA support

**Nick (line 364):** "This shouldn't be correct. We can update host side TMA as well.
We need to support this case."

Currently limitation #4 says `Transform2CTALoads` requires B's descriptor from
`MakeTensorDescOp` (device-side TMA). Host-side TMA passes descriptors as function
arguments (`TensorDescType`), which `Transform2CTALoads` cannot trace back to a
`MakeTensorDescOp` to clone with half-width block shape.

**Ask team:** Should host-side TMA support be part of this PR or future work?
Nick's phrasing ("we need to support this case") suggests it should be supported,
but the implementation is non-trivial — host-side descriptors are created outside
the kernel, so splitting requires modifying the launcher or adding a runtime
descriptor-update mechanism.

**Investigation findings:**

When B comes from host-side TMA, the trace chain hits `ReinterpretTensorDescOp`
instead of `MakeTensorDescOp` and silently returns `failure()` — B splitting is
skipped with no error. The kernel would compile but produce incorrect results
(full B loaded, `cta_group::2` MMA expects half B).

Three approaches to fix:
1. **Runtime descriptor cloning** via `TensormapCreateOp` — emit PTX to create a
   half-width descriptor from the host descriptor at kernel start. Expensive (~30 PTX
   instructions) but works with existing kernel signatures.
2. **Two host descriptors** — host creates two half-width descriptors, kernel takes
   both as args. No runtime cost but changes kernel signature.
3. **Bail out** — detect host-side TMA + `two_ctas` and emit an error. Safest for now.

**Two candidate approaches:**

**Option 1: Dual descriptors from Python (simplest)**
- Host creates two half-width descriptors, kernel takes both as args
- No new compiler ops, zero runtime cost
- Requires Python runtime changes to detect 2-CTA and create the second descriptor
- Leaks compiler concern (B splitting) into the Python runtime

**Option 2: New `ReplaceTensorDescBoxDimOp` (cleanest)**
- New op: takes a `!tt.tensordesc`, returns new `!tt.tensordesc` with modified box_dim
- Lowers to existing `tensormap.replace.tile.box_dim` PTX + fencing (~50 cycles)
- Fits naturally in Transform2CTALoads as an else branch for host-side case
- ~70 lines total: ~20 in TritonOps.td, ~30 in TMALowering.cpp, ~15 in
  Transform2CTALoads.cpp, ~5 in TMAToLLVM.cpp

**Resolution (per Nick's feedback):** The infrastructure already exists. Data
Partitioning (WSDataPartition.cpp) already modifies host-side TMA descriptor
types in the IR. The runtime reads the final block shape from the IR type via
`getTensorDescMetadata()` and creates the CuTensorMap with the correct box_dim.

**Implemented:** Transform2CTALoads now handles host-side descriptors by updating
the function argument's `TensorDescType` to half-width block shape (same pattern
as Data Partitioning). No new ops or runtime changes needed.

**Action items:**
- [x] Implement host-side TMA support in Transform2CTALoads — DONE
- [x] Add a lit test for host-side TMA 2-CTA B splitting — DONE (passes)
- [x] Add E2E Python tests — DONE (non-WS: passes; WS: blocked by WS+2CTA bugs)
- [x] Fixed encoding preservation on `halfBlockType` (must carry `nvmma_shared`)
- [x] Fixed null pointer in cleanup (`makeDesc` null guard)

**WS + 2-CTA bug chain (pre-existing in Zhijing's PR):**
1. ~~`handleOperandD` "Unexpected Producer Found"~~ — caused by pointer stores not
   creating epilogue partition. Fixed by using descriptor stores in WS test kernels.
2. `Insert2CTASync` crashes with "inserting operands without operand storage" —
   the pass tries to modify ops inside WarpSpecializeOp partition regions which
   have restricted operand storage. Root cause: Insert2CTASync was designed for
   pre-WS IR but runs after WS in the Meta pipeline.
- [ ] **Verify:** The `CuTensorMap` created by the runtime may need an additional
      field or flag set for `cta_group::2` mode. Currently `cuTensorMapEncodeTiled`
      is called without any 2-CTA-specific parameters — the `.cta_group::2`
      qualifier is on the PTX load instruction, not the descriptor. Confirm this
      is sufficient, or if the descriptor itself needs a `cta_group` field.

---

## TODO 13: `.cta_group::2` on TMA loads to eliminate explicit cross-CTA sync

**Context (design doc limitation #6):** D96323995 added `.cta_group::2` support for
TMA loads (`AsyncTMACopyGlobalToLocalOp`), which routes barrier arrivals to the
leader CTA in hardware. Currently `Insert2CTASync` inserts explicit cross-CTA sync
(arrive remote, wait local) before each 2-CTA MMA. If B-operand TMA loads used
`.cta_group::2`, the hardware would handle cross-CTA barrier routing automatically,
potentially eliminating the explicit sync pass entirely.

**Nick (line 371):** "This seems safer. Should we just deploy this approach?"

**Investigation findings:**

`.cta_group::2` on TMA loads and `Insert2CTASync` serve **different purposes** but
could theoretically be unified:

- `.cta_group::2` on TMA: routes TMA completion arrivals to the leader CTA's barrier
  (hardware-managed). Used by TLX natively.
- `Insert2CTASync`: explicit "arrive remote, wait local" mbarrier protocol before MMA.
  Uses a separate dedicated barrier from the pipeline's TMA barriers.

To eliminate `Insert2CTASync`, you'd need:
1. Set `two_cta=true` on B-operand `AsyncTMACopyGlobalToLocalOp` (currently not done
   by `Transform2CTALoads`, which operates at `DescriptorLoadOp` level pre-lowering)
2. Set the pipeline barrier's `expect_tx` to cover BOTH CTAs' load bytes
3. Have the leader wait on the unified barrier before MMA

This is what TLX does natively. For non-TLX autoWS, `Insert2CTASync` is the simpler
approach since it doesn't require modifying the pipeline's barrier protocol.

**Team answer (Peng):** Regular TMA load approach is better tested. `.cta_group::2`
TMA loads could make things simpler but defer to follow-up.

**Resolution:** Keep `Insert2CTASync` for now. `.cta_group::2` TMA loads = future work.

**Action items:**
- [x] Decision: defer `.cta_group::2` TMA loads to follow-up

---

## TODO 8: Fix contradictory comments

**manman (Insert2CTASync.cpp:17):** "This comment seems to be contradictory with
comments in compiler.py."

The file-level comment says the pass runs "before pipelining/WS" but compiler.py
places it AFTER all WS passes.

**Action items:**
- [x] Fix the comment in Insert2CTASync.cpp to match actual placement — DONE
      (changed "BEFORE the WS pipeline" to "AFTER all WS-related passes")

---

## TODO 9: Documentation improvements

- [x] Update design doc: `two_ctas=True` should be compatible with any even-sized
      CGA-X in the future — DONE (added limitation #9)
- [x] Document even `num_tiles` constraint — DONE (added limitation #8). Whether
      the compiler should enforce it is still open.
- [ ] Confirm at least 1 test with `num_ctas=1` + `cluster-dim-x=2` (Nick, line 449)
- [x] Add MLIR_ENABLE_DUMP instructions for key passes in test file — DONE
- [x] Update examples to use constexprs — DONE (kernel params with tl.constexpr)
- [x] **Exit cluster barrier scope — DONE.** Changed condition in
      `ConvertWarpSpecializeToLLVM.cpp` from `tlxIsClustered(func)` to
      `tlxIsClustered(func) || getModuleTwoCTAs(func)`. This covers autoWS 2-CTA
      without broadening to all clustered kernels (avoids MultiCTAReduction risk).

---

## TODO 14: Verify Peng/Hongtao's TLX 2-CTA fixes work for autoWS

**manman (general comment):** "Peng and Hongtao recently fixed some correctness
issues related to TLX + 2cta, we will need to make sure the fixes work for
Triton + autoWS + 2cta. Since 2cta for FA is still ongoing, maybe we should
bail out for complex cases so compiler will not generate incorrect code."

**Investigation findings:**

Audit of shared codegen files for 2-CTA fixes:

| Bug | File | Fix | On main? | AutoWS impact |
|-----|------|-----|----------|---------------|
| #1 Deadlock: cluster barrier in WS | MMAv5.cpp:400-422 | Skip ClusterArrive in WS | Yes | Correct |
| #2 Wrong cta_group on commit | WSCodePartition.cpp | N/A — TCGen5CommitOp no longer has `two_ctas` attr; module attr handles it | Yes | Correct |
| #3 Missing cluster0 predicate on commit | MMAv5.cpp:771-777 | CTA0-only predicate | Yes | Correct |
| #4 TMEM cta_group::1 | NVGPUToLLVMPass.cpp | Uses `getModuleTwoCTAs()` | Yes | Correct |
| #5 Exit cluster barrier | ConvertWarpSpecializeToLLVM.cpp | TLX-only guard | **Partial** — autoWS guard is part of THIS PR, not yet on main |

**Key finding:** Bug #5 (exit cluster barrier) is the only fix that's NOT on main
independently. It's part of our PR. Nick flagged that it should only emit for 2-CTA
MMA, not any clustered kernel (see TODO 9).

**`TensorMemoryCTAMode`** (m=64 fix) is on main but NOT used by autoWS. AutoWS
always passes `DEFAULT`. This is a gap for m=64 2-CTA support.

**Action items:**
- [x] Audit whether TLX 2-CTA fixes are on main — DONE (all except bug #5)
- [ ] After our Triton 2-CTA implementation is modified and verified, re-verify
      that TLX fixes still work correctly with autoWS. This is a post-implementation
      validation step, not a pre-requisite.
- [ ] Add bail-out guards for complex/unsupported cases (scaled MMA, m=64 without
      proper TensorMemoryCTAMode)

---

## TODO 15: Explicit sync documentation for AutoWS + 2-CTA

**Nick (general comment):** "My biggest request is that we add more explicit
documentation about the rules governing synchronization, how they interact with
AutoWS. I don't believe this should create deadlock relative to AutoWS, but I
want us to be very clear about the expected behavior."

This is a meta-TODO that spans TODOs 5, 6, and 13. The design doc needs a new
section that documents:
- The complete barrier protocol for 2-CTA WS (which barriers are local vs cluster)
- How `Insert2CTASync`'s barriers interact with WS pipeline barriers
- Under what conditions deadlock can/cannot occur
- The `tcgen05.commit.cta_group::2` hardware guarantees relied upon

**Action items:**
- [ ] Add a "Synchronization Rules" section to the design doc covering all of the
      above, drawing from answers to TODOs 5, 6, and 13

---

## TODO 16: `clearLoopScheduleInfo` moved inside `if (!replaced)` (WSCodePartition.cpp)

**Nick (WSCodePartition.cpp:3415):** "Can you explain this change? This is a bug?
In general this edit seems scary."

**What changed:** On main, `builder.clearLoopScheduleInfo()` runs unconditionally
after the `if (!replaced)` block. Zhijing's original diff (D97387127) moved it
**inside** the `if (!replaced)` block, so it only runs when a new `TCGen5CommitOp`
is created. This is preserved in our rebase.

**Question:** Is this intentional? When `replaced=true`, the commit was handled by
a different path (reuse group). Does `clearLoopScheduleInfo` still need to run in
that case?

**Action items:**
- [ ] **Ask Zhijing:** Was moving `clearLoopScheduleInfo` inside `if (!replaced)`
      intentional? What breaks if it runs unconditionally (as on main)?
- [ ] If intentional: document why in a code comment
- [ ] If not: revert to main's behavior (`clearLoopScheduleInfo` outside the if)

---

## TODO 10: Buckify for CI

**CI failure:** `ci-shell-test-require_buckify_on_beta` — new `.cpp` files not in
buck build targets.

**Action:**
```bash
buck2 run fbcode//triton/tools/reactor:reactor -- buckify triton beta
```

---

## TODO 11: Guard upstream WS + 2-CTA

**Status: DONE** — Added guard in compiler.py to skip upstream WS when
`cluster_dims >= 2`. Only Meta WS supports 2-CTA.

---

## TODO 12: FA with mixed 2-CTA / non-2-CTA dots

**Nick (AccelerateMatmul.cpp:564):** "Is this a general concern for the kernel?
What is supposed to happen when in say FA we don't use 2-CTA on all dots."

**Nick (general):** "I'm concerned that based on some of these comments we can't
support alternating 2-CTA usage in a kernel which seems important for FA."

**Answer:** Mixing `cta_group::1` and `cta_group::2` within a kernel is
**impossible per hardware spec.** The PTX ISA states:

> "All tcgen05 instructions in a kernel **must** use the same `.cta_group` value."

This is already enforced by `CheckMatmulTwoCTAs` (`CheckMatmulTwoCTAs.cpp`), which
emits an error if any MMA ops disagree on `two_ctas`. So FA with selective 2-CTA
on only some dots is fundamentally unsupported — it must be all or nothing.

**Action items:**
- [x] Confirm hardware constraint: all tcgen05 ops must use same `cta_group` — CONFIRMED
- [x] Confirm compiler guard: `CheckMatmulTwoCTAs` already validates consistency — CONFIRMED
- [x] Document this constraint explicitly in the design doc — DONE (added "Hardware
      Constraint" section + "Supported Configurations" table + limitation #10)
- [x] Respond to Nick: mixed 2-CTA is impossible per hardware — documented in
      design doc (Supported Configurations table + Hardware Constraint section)
      dots to use 2-CTA or none
