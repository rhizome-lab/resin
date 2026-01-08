# Open Questions

Single source of truth for design decisions. Updated as we resolve questions.

## Resolution Key

- ✅ **Resolved** - Decision made, documented
- 🔶 **Leaning** - Direction chosen, may revisit
- ❓ **Open** - Needs investigation/decision

---

## Core Architecture

| Question | Status | Notes |
|----------|--------|-------|
| Type system for slots | 🔶 Leaning | Simpler than maki (opaque types). Generics maybe unnecessary |
| Unified graph container | ✅ Resolved | Yes. Single `Graph` type, typed slots (`Output<T>`/`Input<T>`), compile-time safety for Rust, runtime TypeId validation for loaded graphs. Value enum at execution. |
| Portable workflows | ✅ Resolved | Yes. JSON (human-readable) + optional binary. Versioned files, ops declare compatibility. External refs via IDs, optional asset embedding for full portability. |
| Parameter system | 🔶 Leaning | Yes, first-class across all domains |
| Modularity | 🔶 Leaning | Very modular, bevy philosophy |
| Bevy integration | 🔶 Leaning | Low priority. Separate adapter crate if needed. Must not affect core design - resin is standalone first |
| Evaluation strategy | 🔶 Leaning | Evaluator trait. Lazy default, others as needed. See [evaluation-strategy](./design/evaluation-strategy.md) |
| Time models | ✅ Resolved | EvalContext for time, explicit baking, numeric rates + explicit conversion, block = audio iteration, hybrid nodes = feedback edges, determinism = best effort + strict mode. See [time-models](./design/time-models.md) |

## Expression Language

See [expression-language](./design/expression-language.md) for full design.

| Question | Status | Notes |
|----------|--------|-------|
| Backend selection | ✅ Resolved | Cranelift JIT (CPU hot paths), WGSL (GPU), Interpreted (fallback) |
| Expression AST scope | ✅ Resolved | Math + conditionals + let bindings. No loops (use graph recurrence) |
| Per-domain or unified | ✅ Resolved | Unified Expr, domains bind different variables (position, uv, time, etc.) |
| Built-in functions | ✅ Resolved | WGSL built-ins as reference set. Plugin functions decompose or backend traits |
| Value representation | ✅ Resolved | Enum (not dyn trait). Vectors/matrices feature-gated. Square matrices only |
| Matrix operations | ✅ Resolved | `*` works on matrices (WGSL-style). Type inference dispatches |
| Crate structure | ✅ Resolved | core + macros + parse + wgsl + cranelift. Interpreter in core |
| Expr → WGSL codegen | 🔶 Leaning | String generation from AST. Decomposition-first for plugins |
| Expr → Cranelift codegen | 🔶 Leaning | IR generation, external calls for complex functions |
| Plugin function API | 🔶 Leaning | Core trait + backend extension traits in backend crates |
| Constant folding | 🔶 Leaning | Separate resin-expr-opt crate, AST → AST transform |

## Ops & Serialization

| Question | Status | Notes |
|----------|--------|-------|
| Ops as values | ✅ Resolved | Yes, ops are serializable structs. Derive macro for DynOp impl. See [ops-as-values](./design/ops-as-values.md) |
| Plugin op registration | ✅ Resolved | Core defines trait + serialization contract. Plugin *loading* is host's responsibility. Optional adapters (resin-wasm-plugins, etc.) for common cases. See [plugin-architecture](./design/plugin-architecture.md) |
| Graph evaluation caching | ✅ Resolved | Hash-based caching at node boundaries (salsa-style). Inputs unchanged → return cached output |
| External references | 🔶 Leaning | IDs + resolution context (ComfyUI pattern). Maybe support embedding small assets for reproducibility? |

## Meshes

| Question | Status | Notes |
|----------|--------|-------|
| Half-edge vs index-based | 🔶 Leaning | Half-edge internal, indexed on export. Accept memory cost for topology ops. See [mesh-representation](./design/mesh-representation.md) |
| Instancing | ✅ Resolved | Plugin crate (`resin-instances`), not core. Provides `Instances` type + scatter ops + merge to `Mesh`. |
| SDF integration | ✅ Resolved | SDF is `Field<Vec3, f32>`. SDF ops are field combinators. Meshing via explicit operators (MarchingCubes, etc.). Optional `Sdf` marker trait for type-level guarantees. |
| Fields for selection | ✅ Resolved | Yes. `Field<VertexData, bool>` for inline/lazy selection, `SelectionSet` for materialized/named/manual picks. Same Field vs materialized pattern. |

## Audio

| Question | Status | Notes |
|----------|--------|-------|
| Sample rate | ✅ Resolved | Runtime from EvalContext. Nodes query `ctx.sample_rate`. Lazy buffer init (one-time cost per rate). Same graph works at any host rate. |
| Block size | ✅ Resolved | Host-controlled, variable. Nodes handle any size. Block size from EvalContext. Feedback edges update per-block. |
| Modulation depth | ✅ Resolved | Hybrid. Node author decides which params are modulatable via `#[modulatable]`. Explicit mod inputs, not implicit on everything. Graph resolves per-block (not per-sample), zero cost when unconnected. |
| Polyphony model | ✅ Resolved | Per-graph cloning (Pd pattern). Plugin crate (`resin-poly`), not core. Voice allocator clones graph instances, mixes output. Matches instancing pattern. |
| Control vs audio rate | ✅ Resolved | No special types. Control rate = lower numeric rate. Explicit conversion nodes if needed (from mixed rates decision). |
| State management | ✅ Resolved | Recurrent graphs: feedback edges carry state, nodes stay pure. Delay granularity per-edge, mixed rates via explicit conversion. See [recurrent-graphs](./design/recurrent-graphs.md) |

## Textures

| Question | Status | Notes |
|----------|--------|-------|
| GPU vs CPU | ✅ Resolved | Abstract over both via burn/CubeCL. See [prior-art](./prior-art.md#burn--cubecl) |
| Tiling | ✅ Resolved | Explicit operators. Tiling isn't fundamental to most generators. `MakeSeamless`, `Tile`, etc. |
| Resolution/materialization | 🔶 Leaning | Separate Field (lazy) / Image (materialized). Resolution explicit at render(). No propagation. See [texture-materialization](./design/texture-materialization.md) |
| 3D textures | ✅ Resolved | Same nodes, Vec3 input. Vec4 for looping animation (time as 4th dim). Memory/preview are host concerns. |
| Texture vs field | ✅ Resolved | Unified via generic `Field<I, O>` trait. Same concept, different input types. Time handling → see Time models. |

## Vector 2D

| Question | Status | Notes |
|----------|--------|-------|
| Curve types | ✅ Resolved | All via traits. See [curve-types](./design/curve-types.md) |
| Precision f32/f64 | ✅ Resolved | Support both via generic `T: Float` |
| Winding rule | ✅ Resolved | Both, default non-zero. See [winding-rules](./design/winding-rules.md) |
| Vector networks | ✅ Resolved | Network internally, both APIs as equals. See [vector-networks](./design/vector-networks.md) |
| Text | 🔶 Leaning | Include outline extraction, exclude layout (harfbuzz territory) |
| Path ↔ rigging | ✅ Resolved | Rigging (bones+skinning) unified 2D/3D. Morphing is separate `Morph<G>` trait. Paths can drive rig params (spline IK, curve deformers). |

## Rigging

| Question | Status | Notes |
|----------|--------|-------|
| Unified 2D/3D rig | ✅ Resolved | Yes. Generic `Rig<G: HasPositions>` trait. Same bones/skinning concepts, just different dimensionality. Live2D validates 2D case. |
| Deformer stacking | ✅ Resolved | Graph internal, Stack API. See [deformer-stacking](./design/deformer-stacking.md) |
| Animation blending | 🔶 Leaning | Separate crate, bevy-style modularity |
| Procedural rigging | ✅ Resolved | Plugin crate (`resin-autorig`). Core provides rig primitives, auto-generation is domain-specific. |
| Real-time vs offline | ✅ Resolved | Same API, user manages budget. Solver params (iterations, tolerance) for quality/speed tradeoff. Bone count is authoring choice. |

---

## Summary by Status

### ✅ Resolved (36)
- GPU vs CPU abstraction (burn/CubeCL)
- Precision f32/f64 (generic `T: Float`)
- Winding rule (both, default non-zero)
- Curve types (trait-based)
- Vector networks (network internal, both APIs)
- Deformer stacking (graph internal, stack API)
- Expression backend (Cranelift/WGSL/Interpreted)
- Expression AST scope (math + conditionals + let, no loops)
- Expression built-ins (WGSL as reference, plugin extensions)
- Expression Value enum (not dyn trait, vectors/matrices feature-gated)
- Expression matrix ops (`*` works on matrices, type inference dispatches)
- Expression crate structure (core + macros + parse + wgsl + cranelift)
- Unified vs per-domain Expr (unified, domains bind different vars)
- General internal, constrained APIs pattern
- Plugin architecture (core = contract, host = loading)
- Graph caching (hash-based at node boundaries)
- 3D textures (same nodes, Vec3/Vec4 input)
- Tiling (explicit operators)
- Ops as values (derive macro for DynOp impl)
- Texture vs field (Field<I,O> trait, time handling → Time models)
- Time models (EvalContext, explicit baking, numeric rates, feedback = state)
- Mesh instancing (plugin crate, not core)
- Unified graph container (typed slots, Value enum execution)
- Portable workflows (JSON + binary, versioned, asset embedding)
- SDF integration (Field<Vec3, f32>, meshing via explicit ops)
- Fields for selection (Field for lazy, SelectionSet for materialized)
- Audio sample rate (runtime from EvalContext, lazy buffer init)
- Audio block size (host-controlled, variable)
- Audio modulation (hybrid, node author decides, graph resolves per-block)
- Audio polyphony (per-graph cloning, plugin crate)
- Audio control vs audio rate (no special types, explicit conversion)
- Audio state management (recurrent graphs, feedback edges)
- Path ↔ rigging (unified 2D/3D rig, separate Morph trait)
- Unified 2D/3D rig (generic Rig<G: HasPositions>)
- Procedural rigging (plugin crate)
- Real-time vs offline (same API, solver params for tradeoff)

### 🔶 Leaning (14)
- Type system for slots (simpler than maki)
- Parameter system (yes, first-class)
- Modularity (very modular)
- Text (outlines yes, layout no)
- Animation blending (separate crate)
- Bevy integration (low priority, standalone first)
- External references (IDs + context, maybe embed small assets)
- Mesh representation (half-edge internal, indexed export)
- Evaluation strategy (Evaluator trait, lazy default)
- Texture materialization (Field/Image split, explicit resolution)
- Expr → WGSL codegen (string generation, decomposition-first)
- Expr → Cranelift codegen (IR generation, external calls)
- Plugin function API (decompose or backend extension traits)
- Constant folding (resin-expr-opt crate, AST transform)

### ❓ Open (0)
All questions resolved or leaning!
