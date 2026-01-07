# Texture Materialization

When does a procedural texture (lazy field) become actual pixel data?

## The Tension

```rust
// Lazy field: resolution-independent, no memory cost
trait Field {
    fn sample(&self, uv: Vec2) -> Color;
}

// Materialized image: pixels at specific resolution
struct Image {
    pixels: Vec<Color>,  // or GpuTexture
    width: u32,
    height: u32,
}
```

**Lazy is good:**
- No memory until needed
- Resolution-independent (theoretically)
- Composable (chain ops without intermediate allocations)
- GPU-friendly (shader = field evaluation)

**But some ops need pixels:**
- Blur (average neighbors)
- Normal from heightmap (sample dx, dy)
- Edge detection, erosion, dilation
- Any convolution kernel

## Operations Classification

| Op type | Needs neighbors? | Can stay lazy? |
|---------|------------------|----------------|
| Color adjust (brightness, contrast) | No | Yes |
| Per-pixel math (multiply, add) | No | Yes |
| Noise sampling | No | Yes |
| UV transform (scale, rotate) | No | Yes |
| Blur | Yes | No - must materialize |
| Sharpen | Yes | No |
| Normal from height | Yes | No |
| Edge detection | Yes | No |

## Design: Lazy Until Forced

```rust
enum Texture {
    /// Lazy: composable, resolution-independent
    Field(Box<dyn Field>),

    /// Materialized: has pixels
    Image(Image),
}
```

**Lazy ops accumulate:**
```rust
let tex = Perlin::new()           // Field
    .map(|v| Color::gray(v))       // Field (fused)
    .brightness(0.1)               // Field (fused)
    .contrast(1.2);                // Field (fused)

// All above is ONE field - no pixels yet
```

**Neighbor ops force materialization:**
```rust
let blurred = tex.blur(5, 1024, 1024);  // Must specify resolution
// Now we have pixels
```

**Final output:**
```rust
let image = tex.render(2048, 2048);  // Materialize at output resolution
```

## Kernel Fusion

Multiple lazy ops should fuse into single evaluation:

```rust
// Conceptually three ops
noise.brightness(0.1).contrast(1.2).saturate(0.5)

// Should compile to single shader/function:
fn fused_sample(uv: Vec2) -> Color {
    let v = noise.sample(uv);
    let c = Color::gray(v);
    let c = c.brightness(0.1);
    let c = c.contrast(1.2);
    let c = c.saturate(0.5);
    c
}
```

**API implication:** Lazy ops return `Field`, preserving fusion opportunity. Only `render()` or neighbor ops trigger execution.

## Resolution: The Hard Part

### Who Specifies Resolution?

**For final output:** User specifies at render time.
```rust
tex.render(1024, 1024)  // User chooses
```

**For neighbor ops:** Needs resolution, but when?

```
noise → brightness → blur → contrast → render(1024)
         lazy        NEEDS    lazy
                     PIXELS
```

Options:

**A. Explicit at each neighbor op**
```rust
tex.blur(5, Resolution::Explicit(1024, 1024))
```
Verbose. User must know resolution early.

**B. Resolution flows backward from output**
```rust
// Blur doesn't know resolution yet
let graph = noise.brightness(0.1).blur(5).contrast(1.2);

// Resolution determined at render time, flows back to blur
graph.render(1024, 1024);
```
Requires two-phase evaluation: resolve resolutions, then execute.

**C. Resolution is graph-level setting**
```rust
let graph = TextureGraph::new()
    .resolution(1024, 1024)  // All materializations use this
    .add(noise)
    .add(blur(5))
    .add(contrast(1.2));
```
Simple but inflexible.

### Resolution Units Problem

If blur radius is in pixels:
- `blur(5)` at 512px ≠ `blur(5)` at 1024px (visually different)
- Field is NOT truly resolution-independent

Options:
- **Pixel units:** Accept resolution-dependence. User adjusts for target resolution.
- **UV units:** `blur(0.01)` = 1% of image width. Resolution-independent semantically.
- **Both:** `blur_px(5)` vs `blur_uv(0.01)`

```rust
impl Texture {
    /// Blur radius in pixels (resolution-dependent)
    fn blur_px(&self, radius_px: f32, resolution: (u32, u32)) -> Texture;

    /// Blur radius in UV space (resolution-independent)
    fn blur_uv(&self, radius_uv: f32) -> Texture;
}
```

## GPU Execution Model

On GPU:
- **Field** = shader program (runs per-pixel in parallel)
- **Materialize** = render to texture
- **Neighbor op** = shader that samples texture

```
Field (shader) → Materialize (render to texture) → Neighbor op (shader samples texture)
```

Fusion on GPU = compose shader functions before compilation.

## Two-Phase Evaluation Concern

**The problem:** If resolution flows backward, we need:
1. Phase 1: Traverse graph, determine resolutions
2. Phase 2: Execute with known resolutions

This feels like special-casing resolution.

**Alternative: Resolution as explicit input**

Resolution is a parameter like any other:

```rust
struct RenderContext {
    resolution: (u32, u32),
    // ...
}

// Neighbor ops read resolution from context
impl Blur {
    fn eval(&self, input: Texture, ctx: &RenderContext) -> Texture {
        let (w, h) = ctx.resolution;
        // materialize input at (w, h), then blur
    }
}
```

Context is set at eval time, flows forward. No two-phase.

But: what if different branches need different resolutions? (e.g., blur at full res, thumbnail at 1/4 res)

## Open Questions

1. **Resolution propagation:** Forward (context) or backward (from output)? Or both?

2. **UV vs pixel units:** Default to UV-space for resolution independence? Pixel ops for specific needs?

3. **Multi-resolution graphs:** Same graph, different resolutions for different outputs?

4. **Partial materialization:** Can we materialize only the region needed? (tiled evaluation)

5. **Mipmap integration:** When to generate mipmaps? Affects blur quality.

6. **Viewport preview:** Live preview at viewport resolution. How does this interact with final render resolution?

## Summary

| Aspect | Direction |
|--------|-----------|
| Default | Lazy (Field) until forced |
| Neighbor ops | Force materialization |
| Fusion | Lazy ops fuse into single evaluation |
| Resolution source | TBD - either context (forward) or output (backward) |
| Units | Probably UV-space default for resolution independence |

## Not Resolved

- Exact resolution propagation mechanism
- Two-phase vs context-based
- How viewport preview resolution relates to render resolution
