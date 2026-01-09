//! Field trait for lazy evaluation.
//!
//! A `Field<I, O>` represents a function that can be sampled at any point.
//! Fields are lazy - they describe computation, not data. Evaluation happens
//! on demand when you call `sample()`.
//!
//! # Examples
//!
//! ```
//! use rhizome_resin_field::{Field, EvalContext, Perlin2D};
//! use glam::Vec2;
//!
//! // Create a noise field
//! let noise = Perlin2D::new().scale(4.0);
//!
//! // Sample at a point
//! let ctx = EvalContext::new();
//! let value = noise.sample(Vec2::new(0.5, 0.5), &ctx);
//! ```

mod context;

use glam::{Vec2, Vec3};
use std::marker::PhantomData;

pub use context::EvalContext;

/// A field that can be sampled at any point.
///
/// Fields are the core abstraction for lazy, spatial computation.
/// They represent functions from input coordinates to output values.
pub trait Field<I, O> {
    /// Samples the field at a given input coordinate.
    fn sample(&self, input: I, ctx: &EvalContext) -> O;

    /// Transforms the output of this field.
    fn map<O2, F>(self, f: F) -> Map<Self, F, O>
    where
        Self: Sized,
        F: Fn(O) -> O2,
    {
        Map {
            field: self,
            f,
            _phantom: PhantomData,
        }
    }

    /// Scales the input coordinates.
    fn scale(self, factor: f32) -> Scale<Self>
    where
        Self: Sized,
    {
        Scale {
            field: self,
            factor,
        }
    }

    /// Translates the input coordinates.
    fn translate(self, offset: I) -> Translate<Self, I>
    where
        Self: Sized,
        I: Clone,
    {
        Translate {
            field: self,
            offset,
        }
    }

    /// Adds this field to another.
    fn add<F2>(self, other: F2) -> Add<Self, F2>
    where
        Self: Sized,
        F2: Field<I, O>,
    {
        Add { a: self, b: other }
    }

    /// Multiplies this field by another.
    fn mul<F2>(self, other: F2) -> Mul<Self, F2>
    where
        Self: Sized,
        F2: Field<I, O>,
    {
        Mul { a: self, b: other }
    }

    /// Mixes this field with another using a blend factor field.
    fn mix<F2, FB>(self, other: F2, blend: FB) -> Mix<Self, F2, FB>
    where
        Self: Sized,
        F2: Field<I, O>,
        FB: Field<I, f32>,
    {
        Mix {
            a: self,
            b: other,
            blend,
        }
    }
}

// Combinators

/// Maps the output of a field.
pub struct Map<F, M, O> {
    field: F,
    f: M,
    _phantom: PhantomData<O>,
}

impl<I, O, O2, F, M> Field<I, O2> for Map<F, M, O>
where
    F: Field<I, O>,
    M: Fn(O) -> O2,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> O2 {
        (self.f)(self.field.sample(input, ctx))
    }
}

/// Scales the input coordinates of a field.
pub struct Scale<F> {
    field: F,
    factor: f32,
}

impl<O, F> Field<Vec2, O> for Scale<F>
where
    F: Field<Vec2, O>,
{
    fn sample(&self, input: Vec2, ctx: &EvalContext) -> O {
        self.field.sample(input * self.factor, ctx)
    }
}

impl<O, F> Field<Vec3, O> for Scale<F>
where
    F: Field<Vec3, O>,
{
    fn sample(&self, input: Vec3, ctx: &EvalContext) -> O {
        self.field.sample(input * self.factor, ctx)
    }
}

impl<O, F> Field<f32, O> for Scale<F>
where
    F: Field<f32, O>,
{
    fn sample(&self, input: f32, ctx: &EvalContext) -> O {
        self.field.sample(input * self.factor, ctx)
    }
}

/// Translates the input coordinates of a field.
pub struct Translate<F, I> {
    field: F,
    offset: I,
}

impl<O, F> Field<Vec2, O> for Translate<F, Vec2>
where
    F: Field<Vec2, O>,
{
    fn sample(&self, input: Vec2, ctx: &EvalContext) -> O {
        self.field.sample(input - self.offset, ctx)
    }
}

impl<O, F> Field<Vec3, O> for Translate<F, Vec3>
where
    F: Field<Vec3, O>,
{
    fn sample(&self, input: Vec3, ctx: &EvalContext) -> O {
        self.field.sample(input - self.offset, ctx)
    }
}

impl<O, F> Field<f32, O> for Translate<F, f32>
where
    F: Field<f32, O>,
{
    fn sample(&self, input: f32, ctx: &EvalContext) -> O {
        self.field.sample(input - self.offset, ctx)
    }
}

/// Adds two fields together.
pub struct Add<A, B> {
    a: A,
    b: B,
}

impl<I, A, B> Field<I, f32> for Add<A, B>
where
    I: Clone,
    A: Field<I, f32>,
    B: Field<I, f32>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> f32 {
        self.a.sample(input.clone(), ctx) + self.b.sample(input, ctx)
    }
}

impl<I, A, B> Field<I, Vec2> for Add<A, B>
where
    I: Clone,
    A: Field<I, Vec2>,
    B: Field<I, Vec2>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> Vec2 {
        self.a.sample(input.clone(), ctx) + self.b.sample(input, ctx)
    }
}

impl<I, A, B> Field<I, Vec3> for Add<A, B>
where
    I: Clone,
    A: Field<I, Vec3>,
    B: Field<I, Vec3>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> Vec3 {
        self.a.sample(input.clone(), ctx) + self.b.sample(input, ctx)
    }
}

/// Multiplies two fields together.
pub struct Mul<A, B> {
    a: A,
    b: B,
}

impl<I, A, B> Field<I, f32> for Mul<A, B>
where
    I: Clone,
    A: Field<I, f32>,
    B: Field<I, f32>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> f32 {
        self.a.sample(input.clone(), ctx) * self.b.sample(input, ctx)
    }
}

impl<I, A, B> Field<I, Vec2> for Mul<A, B>
where
    I: Clone,
    A: Field<I, Vec2>,
    B: Field<I, Vec2>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> Vec2 {
        self.a.sample(input.clone(), ctx) * self.b.sample(input, ctx)
    }
}

impl<I, A, B> Field<I, Vec3> for Mul<A, B>
where
    I: Clone,
    A: Field<I, Vec3>,
    B: Field<I, Vec3>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> Vec3 {
        self.a.sample(input.clone(), ctx) * self.b.sample(input, ctx)
    }
}

/// Mixes two fields using a blend factor.
pub struct Mix<A, B, Blend> {
    a: A,
    b: B,
    blend: Blend,
}

impl<I, A, B, Blend> Field<I, f32> for Mix<A, B, Blend>
where
    I: Clone,
    A: Field<I, f32>,
    B: Field<I, f32>,
    Blend: Field<I, f32>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> f32 {
        let t = self.blend.sample(input.clone(), ctx);
        let a = self.a.sample(input.clone(), ctx);
        let b = self.b.sample(input, ctx);
        a * (1.0 - t) + b * t
    }
}

impl<I, A, B, Blend> Field<I, Vec2> for Mix<A, B, Blend>
where
    I: Clone,
    A: Field<I, Vec2>,
    B: Field<I, Vec2>,
    Blend: Field<I, f32>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> Vec2 {
        let t = self.blend.sample(input.clone(), ctx);
        let a = self.a.sample(input.clone(), ctx);
        let b = self.b.sample(input, ctx);
        a.lerp(b, t)
    }
}

impl<I, A, B, Blend> Field<I, Vec3> for Mix<A, B, Blend>
where
    I: Clone,
    A: Field<I, Vec3>,
    B: Field<I, Vec3>,
    Blend: Field<I, f32>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> Vec3 {
        let t = self.blend.sample(input.clone(), ctx);
        let a = self.a.sample(input.clone(), ctx);
        let b = self.b.sample(input, ctx);
        a.lerp(b, t)
    }
}

// Basic field implementations

/// A constant field that always returns the same value.
#[derive(Debug, Clone, Copy)]
pub struct Constant<O> {
    /// The constant value to return.
    pub value: O,
}

impl<O> Constant<O> {
    /// Create a new constant field.
    pub fn new(value: O) -> Self {
        Self { value }
    }
}

impl<I, O: Clone> Field<I, O> for Constant<O> {
    fn sample(&self, _input: I, _ctx: &EvalContext) -> O {
        self.value.clone()
    }
}

/// A field that returns the input coordinates.
#[derive(Debug, Clone, Copy, Default)]
pub struct Coordinates;

impl Field<Vec2, Vec2> for Coordinates {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> Vec2 {
        input
    }
}

impl Field<Vec3, Vec3> for Coordinates {
    fn sample(&self, input: Vec3, _ctx: &EvalContext) -> Vec3 {
        input
    }
}

impl Field<f32, f32> for Coordinates {
    fn sample(&self, input: f32, _ctx: &EvalContext) -> f32 {
        input
    }
}

/// Perlin noise field (2D).
#[derive(Debug, Clone, Copy, Default)]
pub struct Perlin2D {
    /// Random seed.
    pub seed: i32,
}

impl Perlin2D {
    /// Create a new Perlin noise field.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a specific seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed }
    }
}

impl Field<Vec2, f32> for Perlin2D {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        // Offset by seed
        let p = input + Vec2::new(self.seed as f32 * 17.0, self.seed as f32 * 31.0);
        rhizome_resin_noise::perlin2(p.x, p.y)
    }
}

/// Perlin noise field (3D).
#[derive(Debug, Clone, Copy, Default)]
pub struct Perlin3D {
    /// Random seed.
    pub seed: i32,
}

impl Perlin3D {
    /// Create a new Perlin noise field.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a specific seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed }
    }
}

impl Field<Vec3, f32> for Perlin3D {
    fn sample(&self, input: Vec3, _ctx: &EvalContext) -> f32 {
        let p = input
            + Vec3::new(
                self.seed as f32 * 17.0,
                self.seed as f32 * 31.0,
                self.seed as f32 * 47.0,
            );
        rhizome_resin_noise::perlin3(p.x, p.y, p.z)
    }
}

/// Simplex noise field (2D).
#[derive(Debug, Clone, Copy, Default)]
pub struct Simplex2D {
    /// Random seed.
    pub seed: i32,
}

impl Simplex2D {
    /// Create a new Simplex noise field.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a specific seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed }
    }
}

impl Field<Vec2, f32> for Simplex2D {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let p = input + Vec2::new(self.seed as f32 * 17.0, self.seed as f32 * 31.0);
        rhizome_resin_noise::simplex2(p.x, p.y)
    }
}

/// Simplex noise field (3D).
#[derive(Debug, Clone, Copy, Default)]
pub struct Simplex3D {
    /// Random seed.
    pub seed: i32,
}

impl Simplex3D {
    /// Create a new Simplex noise field.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a specific seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed }
    }
}

impl Field<Vec3, f32> for Simplex3D {
    fn sample(&self, input: Vec3, _ctx: &EvalContext) -> f32 {
        let p = input
            + Vec3::new(
                self.seed as f32 * 17.0,
                self.seed as f32 * 31.0,
                self.seed as f32 * 47.0,
            );
        rhizome_resin_noise::simplex3(p.x, p.y, p.z)
    }
}

/// Fractal Brownian Motion field (2D).
#[derive(Debug, Clone, Copy)]
pub struct Fbm2D<F> {
    /// Base noise field.
    pub base: F,
    /// Number of octaves (layers).
    pub octaves: u32,
    /// Frequency multiplier between octaves.
    pub lacunarity: f32,
    /// Amplitude multiplier between octaves.
    pub gain: f32,
}

impl<F> Fbm2D<F> {
    /// Create a new FBM field from a base noise.
    pub fn new(base: F) -> Self {
        Self {
            base,
            octaves: 6,
            lacunarity: 2.0,
            gain: 0.5,
        }
    }

    /// Set the number of octaves.
    pub fn octaves(mut self, octaves: u32) -> Self {
        self.octaves = octaves;
        self
    }

    /// Set the lacunarity (frequency multiplier).
    pub fn lacunarity(mut self, lacunarity: f32) -> Self {
        self.lacunarity = lacunarity;
        self
    }

    /// Set the gain (amplitude multiplier).
    pub fn gain(mut self, gain: f32) -> Self {
        self.gain = gain;
        self
    }
}

impl<F: Field<Vec2, f32>> Field<Vec2, f32> for Fbm2D<F> {
    fn sample(&self, input: Vec2, ctx: &EvalContext) -> f32 {
        let mut value = 0.0;
        let mut amplitude = 1.0;
        let mut frequency = 1.0;
        let mut max_value = 0.0;

        for _ in 0..self.octaves {
            value += amplitude * self.base.sample(input * frequency, ctx);
            max_value += amplitude;
            amplitude *= self.gain;
            frequency *= self.lacunarity;
        }

        (value / max_value).clamp(0.0, 1.0)
    }
}

/// Fractal Brownian Motion field (3D).
#[derive(Debug, Clone, Copy)]
pub struct Fbm3D<F> {
    /// Base noise field.
    pub base: F,
    /// Number of octaves (layers).
    pub octaves: u32,
    /// Frequency multiplier between octaves.
    pub lacunarity: f32,
    /// Amplitude multiplier between octaves.
    pub gain: f32,
}

impl<F> Fbm3D<F> {
    /// Create a new FBM field from a base noise.
    pub fn new(base: F) -> Self {
        Self {
            base,
            octaves: 6,
            lacunarity: 2.0,
            gain: 0.5,
        }
    }

    /// Set the number of octaves.
    pub fn octaves(mut self, octaves: u32) -> Self {
        self.octaves = octaves;
        self
    }

    /// Set the lacunarity (frequency multiplier).
    pub fn lacunarity(mut self, lacunarity: f32) -> Self {
        self.lacunarity = lacunarity;
        self
    }

    /// Set the gain (amplitude multiplier).
    pub fn gain(mut self, gain: f32) -> Self {
        self.gain = gain;
        self
    }
}

impl<F: Field<Vec3, f32>> Field<Vec3, f32> for Fbm3D<F> {
    fn sample(&self, input: Vec3, ctx: &EvalContext) -> f32 {
        let mut value = 0.0;
        let mut amplitude = 1.0;
        let mut frequency = 1.0;
        let mut max_value = 0.0;

        for _ in 0..self.octaves {
            value += amplitude * self.base.sample(input * frequency, ctx);
            max_value += amplitude;
            amplitude *= self.gain;
            frequency *= self.lacunarity;
        }

        (value / max_value).clamp(0.0, 1.0)
    }
}

// ============================================================================
// Terrain generation
// ============================================================================

/// Terrain heightfield generator.
///
/// Combines multiple noise octaves with configurable parameters for
/// realistic terrain generation.
#[derive(Debug, Clone, Copy)]
pub struct Terrain2D {
    /// Random seed.
    pub seed: i32,
    /// Number of noise octaves.
    pub octaves: u32,
    /// Frequency multiplier between octaves.
    pub lacunarity: f32,
    /// Amplitude multiplier between octaves.
    pub persistence: f32,
    /// Overall scale of the terrain features.
    pub scale: f32,
    /// Height exponent for terrain shaping (>1 = steeper peaks).
    pub exponent: f32,
}

impl Default for Terrain2D {
    fn default() -> Self {
        Self {
            seed: 0,
            octaves: 6,
            lacunarity: 2.0,
            persistence: 0.5,
            scale: 1.0,
            exponent: 1.0,
        }
    }
}

impl Terrain2D {
    /// Creates a new terrain generator with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the random seed.
    pub fn with_seed(mut self, seed: i32) -> Self {
        self.seed = seed;
        self
    }

    /// Sets the number of octaves.
    pub fn with_octaves(mut self, octaves: u32) -> Self {
        self.octaves = octaves;
        self
    }

    /// Sets the lacunarity (frequency multiplier).
    pub fn with_lacunarity(mut self, lacunarity: f32) -> Self {
        self.lacunarity = lacunarity;
        self
    }

    /// Sets the persistence (amplitude multiplier).
    pub fn with_persistence(mut self, persistence: f32) -> Self {
        self.persistence = persistence;
        self
    }

    /// Sets the overall scale.
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Sets the height exponent.
    pub fn with_exponent(mut self, exponent: f32) -> Self {
        self.exponent = exponent;
        self
    }

    /// Creates a preset for gentle rolling hills.
    pub fn rolling_hills() -> Self {
        Self {
            seed: 0,
            octaves: 4,
            lacunarity: 2.0,
            persistence: 0.4,
            scale: 0.5,
            exponent: 0.8,
        }
    }

    /// Creates a preset for mountainous terrain.
    pub fn mountains() -> Self {
        Self {
            seed: 0,
            octaves: 8,
            lacunarity: 2.1,
            persistence: 0.55,
            scale: 1.5,
            exponent: 1.5,
        }
    }

    /// Creates a preset for flat plains with minor variation.
    pub fn plains() -> Self {
        Self {
            seed: 0,
            octaves: 3,
            lacunarity: 2.0,
            persistence: 0.3,
            scale: 0.3,
            exponent: 0.5,
        }
    }

    /// Creates a preset for canyon-like terrain.
    pub fn canyons() -> Self {
        Self {
            seed: 0,
            octaves: 5,
            lacunarity: 2.5,
            persistence: 0.6,
            scale: 1.0,
            exponent: 2.0,
        }
    }
}

impl Field<Vec2, f32> for Terrain2D {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let p = input * self.scale + Vec2::new(self.seed as f32 * 17.0, self.seed as f32 * 31.0);

        let mut value = 0.0;
        let mut amplitude = 1.0;
        let mut frequency = 1.0;
        let mut max_value = 0.0;

        for _ in 0..self.octaves {
            value += amplitude * rhizome_resin_noise::simplex2(p.x * frequency, p.y * frequency);
            max_value += amplitude;
            amplitude *= self.persistence;
            frequency *= self.lacunarity;
        }

        // Normalize to 0-1 range
        let normalized = (value / max_value + 1.0) * 0.5;

        // Apply exponent for terrain shaping
        normalized.powf(self.exponent).clamp(0.0, 1.0)
    }
}

/// Ridged noise terrain for sharp mountain ridges.
#[derive(Debug, Clone, Copy)]
pub struct RidgedTerrain2D {
    /// Random seed.
    pub seed: i32,
    /// Number of noise octaves.
    pub octaves: u32,
    /// Frequency multiplier between octaves.
    pub lacunarity: f32,
    /// Overall scale.
    pub scale: f32,
    /// Ridge sharpness (higher = sharper).
    pub sharpness: f32,
}

impl Default for RidgedTerrain2D {
    fn default() -> Self {
        Self {
            seed: 0,
            octaves: 6,
            lacunarity: 2.0,
            scale: 1.0,
            sharpness: 2.0,
        }
    }
}

impl RidgedTerrain2D {
    /// Creates a new ridged terrain generator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the random seed.
    pub fn with_seed(mut self, seed: i32) -> Self {
        self.seed = seed;
        self
    }

    /// Sets the number of octaves.
    pub fn with_octaves(mut self, octaves: u32) -> Self {
        self.octaves = octaves;
        self
    }

    /// Sets the overall scale.
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Sets the ridge sharpness.
    pub fn with_sharpness(mut self, sharpness: f32) -> Self {
        self.sharpness = sharpness;
        self
    }
}

impl Field<Vec2, f32> for RidgedTerrain2D {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let p = input * self.scale + Vec2::new(self.seed as f32 * 17.0, self.seed as f32 * 31.0);

        let mut value = 0.0;
        let mut amplitude = 1.0;
        let mut frequency = 1.0;
        let mut weight = 1.0;

        for _ in 0..self.octaves {
            let noise = rhizome_resin_noise::simplex2(p.x * frequency, p.y * frequency);

            // Ridged noise: abs(noise) with inversion for ridges
            let ridge = 1.0 - noise.abs();
            let ridge = ridge.powf(self.sharpness);

            value += ridge * amplitude * weight;

            // Reduce weight for successive octaves based on current value
            weight = ridge.clamp(0.0, 1.0);

            amplitude *= 0.5;
            frequency *= self.lacunarity;
        }

        value.clamp(0.0, 1.0)
    }
}

/// Billowy terrain for rounded, cloud-like hills.
#[derive(Debug, Clone, Copy)]
pub struct BillowyTerrain2D {
    /// Random seed.
    pub seed: i32,
    /// Number of noise octaves.
    pub octaves: u32,
    /// Frequency multiplier between octaves.
    pub lacunarity: f32,
    /// Overall scale.
    pub scale: f32,
}

impl Default for BillowyTerrain2D {
    fn default() -> Self {
        Self {
            seed: 0,
            octaves: 5,
            lacunarity: 2.0,
            scale: 1.0,
        }
    }
}

impl BillowyTerrain2D {
    /// Creates a new billowy terrain generator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the random seed.
    pub fn with_seed(mut self, seed: i32) -> Self {
        self.seed = seed;
        self
    }

    /// Sets the number of octaves.
    pub fn with_octaves(mut self, octaves: u32) -> Self {
        self.octaves = octaves;
        self
    }

    /// Sets the overall scale.
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }
}

impl Field<Vec2, f32> for BillowyTerrain2D {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let p = input * self.scale + Vec2::new(self.seed as f32 * 17.0, self.seed as f32 * 31.0);

        let mut value = 0.0;
        let mut amplitude = 1.0;
        let mut frequency = 1.0;
        let mut max_value = 0.0;

        for _ in 0..self.octaves {
            let noise = rhizome_resin_noise::simplex2(p.x * frequency, p.y * frequency);

            // Billowy: abs(noise) gives rounded peaks
            value += noise.abs() * amplitude;
            max_value += amplitude;

            amplitude *= 0.5;
            frequency *= self.lacunarity;
        }

        (value / max_value).clamp(0.0, 1.0)
    }
}

/// Island terrain generator with falloff at edges.
#[derive(Debug, Clone, Copy)]
pub struct IslandTerrain2D {
    /// Base terrain noise.
    pub terrain: Terrain2D,
    /// Falloff radius (island size).
    pub radius: f32,
    /// Center of the island.
    pub center: Vec2,
    /// Falloff sharpness.
    pub falloff: f32,
}

impl Default for IslandTerrain2D {
    fn default() -> Self {
        Self {
            terrain: Terrain2D::default(),
            radius: 1.0,
            center: Vec2::ZERO,
            falloff: 2.0,
        }
    }
}

impl IslandTerrain2D {
    /// Creates a new island terrain generator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the island radius.
    pub fn with_radius(mut self, radius: f32) -> Self {
        self.radius = radius;
        self
    }

    /// Sets the island center.
    pub fn with_center(mut self, center: Vec2) -> Self {
        self.center = center;
        self
    }

    /// Sets the falloff sharpness.
    pub fn with_falloff(mut self, falloff: f32) -> Self {
        self.falloff = falloff;
        self
    }

    /// Sets the base terrain generator.
    pub fn with_terrain(mut self, terrain: Terrain2D) -> Self {
        self.terrain = terrain;
        self
    }
}

impl Field<Vec2, f32> for IslandTerrain2D {
    fn sample(&self, input: Vec2, ctx: &EvalContext) -> f32 {
        let dist = (input - self.center).length() / self.radius;

        // Smooth falloff using smoothstep-like curve
        let falloff = if dist >= 1.0 {
            0.0
        } else {
            let t = 1.0 - dist;
            let s = t.powf(self.falloff);
            s
        };

        let terrain_height = self.terrain.sample(input, ctx);

        (terrain_height * falloff).clamp(0.0, 1.0)
    }
}

/// Terraced terrain for stepped plateaus.
#[derive(Debug, Clone, Copy)]
pub struct TerracedTerrain2D<F> {
    /// Base terrain field.
    pub base: F,
    /// Number of terrace levels.
    pub levels: u32,
    /// Terrace sharpness (0=smooth, 1=sharp).
    pub sharpness: f32,
}

impl<F> TerracedTerrain2D<F> {
    /// Creates a new terraced terrain.
    pub fn new(base: F, levels: u32) -> Self {
        Self {
            base,
            levels,
            sharpness: 0.8,
        }
    }

    /// Sets the terrace sharpness.
    pub fn with_sharpness(mut self, sharpness: f32) -> Self {
        self.sharpness = sharpness.clamp(0.0, 1.0);
        self
    }
}

impl<F: Field<Vec2, f32>> Field<Vec2, f32> for TerracedTerrain2D<F> {
    fn sample(&self, input: Vec2, ctx: &EvalContext) -> f32 {
        let base_value = self.base.sample(input, ctx);

        // Quantize to terrace levels
        let scaled = base_value * self.levels as f32;
        let terrace = scaled.floor();
        let frac = scaled - terrace;

        // Smooth the transition between terraces
        let smoothed_frac = if self.sharpness >= 1.0 {
            0.0
        } else {
            // Smoothstep-like transition
            let t = frac / (1.0 - self.sharpness);
            if t >= 1.0 {
                1.0
            } else {
                t * t * (3.0 - 2.0 * t)
            }
        };

        ((terrace + smoothed_frac * (1.0 - self.sharpness)) / self.levels as f32).clamp(0.0, 1.0)
    }
}

/// Gradient field - returns gradient based on coordinates.
#[derive(Debug, Clone, Copy)]
pub struct Gradient2D {
    /// Start value.
    pub start: f32,
    /// End value.
    pub end: f32,
    /// Gradient direction.
    pub direction: Vec2,
}

impl Gradient2D {
    /// Create a horizontal gradient (left to right).
    pub fn horizontal() -> Self {
        Self {
            start: 0.0,
            end: 1.0,
            direction: Vec2::X,
        }
    }

    /// Create a vertical gradient (bottom to top).
    pub fn vertical() -> Self {
        Self {
            start: 0.0,
            end: 1.0,
            direction: Vec2::Y,
        }
    }

    /// Create a radial gradient.
    pub fn radial(center: Vec2) -> Radial2D {
        Radial2D {
            center,
            radius: 1.0,
        }
    }
}

impl Field<Vec2, f32> for Gradient2D {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let t = input.dot(self.direction);
        (self.start + t * (self.end - self.start)).clamp(0.0, 1.0)
    }
}

/// Radial gradient field.
#[derive(Debug, Clone, Copy)]
pub struct Radial2D {
    /// Center point.
    pub center: Vec2,
    /// Radius of the gradient.
    pub radius: f32,
}

impl Radial2D {
    /// Create a new radial gradient.
    pub fn new(center: Vec2, radius: f32) -> Self {
        Self { center, radius }
    }
}

impl Field<Vec2, f32> for Radial2D {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let d = (input - self.center).length();
        (1.0 - d / self.radius).clamp(0.0, 1.0)
    }
}

// ============================================================================
// Texture patterns
// ============================================================================

/// Checkerboard pattern.
#[derive(Debug, Clone, Copy)]
pub struct Checkerboard {
    /// Pattern scale.
    pub scale: f32,
}

impl Default for Checkerboard {
    fn default() -> Self {
        Self { scale: 1.0 }
    }
}

impl Checkerboard {
    /// Create a new checkerboard pattern.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a specific scale.
    pub fn with_scale(scale: f32) -> Self {
        Self { scale }
    }
}

impl Field<Vec2, f32> for Checkerboard {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let p = input * self.scale;
        let x = p.x.floor() as i32;
        let y = p.y.floor() as i32;
        if (x + y) % 2 == 0 { 1.0 } else { 0.0 }
    }
}

impl Field<Vec3, f32> for Checkerboard {
    fn sample(&self, input: Vec3, _ctx: &EvalContext) -> f32 {
        let p = input * self.scale;
        let x = p.x.floor() as i32;
        let y = p.y.floor() as i32;
        let z = p.z.floor() as i32;
        if (x + y + z) % 2 == 0 { 1.0 } else { 0.0 }
    }
}

/// Stripe pattern (horizontal by default).
#[derive(Debug, Clone, Copy)]
pub struct Stripes {
    /// Stripe frequency.
    pub frequency: f32,
    /// Stripe direction.
    pub direction: Vec2,
}

impl Default for Stripes {
    fn default() -> Self {
        Self {
            frequency: 1.0,
            direction: Vec2::Y,
        }
    }
}

impl Stripes {
    /// Create horizontal stripes.
    pub fn horizontal() -> Self {
        Self {
            direction: Vec2::Y,
            ..Self::default()
        }
    }

    /// Create vertical stripes.
    pub fn vertical() -> Self {
        Self {
            direction: Vec2::X,
            ..Self::default()
        }
    }

    /// Set the frequency.
    pub fn with_frequency(mut self, frequency: f32) -> Self {
        self.frequency = frequency;
        self
    }

    /// Set the direction.
    pub fn with_direction(mut self, direction: Vec2) -> Self {
        self.direction = direction.normalize();
        self
    }
}

impl Field<Vec2, f32> for Stripes {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let t = input.dot(self.direction) * self.frequency;
        if (t.floor() as i32) % 2 == 0 {
            1.0
        } else {
            0.0
        }
    }
}

/// Smooth stripe pattern (sine wave).
#[derive(Debug, Clone, Copy)]
pub struct SmoothStripes {
    /// Stripe frequency.
    pub frequency: f32,
    /// Stripe direction.
    pub direction: Vec2,
}

impl Default for SmoothStripes {
    fn default() -> Self {
        Self {
            frequency: 1.0,
            direction: Vec2::Y,
        }
    }
}

impl SmoothStripes {
    /// Create horizontal stripes.
    pub fn horizontal() -> Self {
        Self {
            direction: Vec2::Y,
            ..Self::default()
        }
    }

    /// Create vertical stripes.
    pub fn vertical() -> Self {
        Self {
            direction: Vec2::X,
            ..Self::default()
        }
    }

    /// Set the frequency.
    pub fn with_frequency(mut self, frequency: f32) -> Self {
        self.frequency = frequency;
        self
    }
}

impl Field<Vec2, f32> for SmoothStripes {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let t = input.dot(self.direction) * self.frequency * std::f32::consts::TAU;
        (t.sin() * 0.5 + 0.5).clamp(0.0, 1.0)
    }
}

/// Brick pattern.
#[derive(Debug, Clone, Copy)]
pub struct Brick {
    /// Brick size.
    pub scale: Vec2,
    /// Mortar width.
    pub mortar: f32,
    /// Row offset (0.5 for standard brick).
    pub offset: f32,
}

impl Default for Brick {
    fn default() -> Self {
        Self {
            scale: Vec2::new(2.0, 1.0),
            mortar: 0.05,
            offset: 0.5,
        }
    }
}

impl Brick {
    /// Create a new brick pattern.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the brick scale.
    pub fn with_scale(mut self, scale: Vec2) -> Self {
        self.scale = scale;
        self
    }

    /// Set the mortar width.
    pub fn with_mortar(mut self, mortar: f32) -> Self {
        self.mortar = mortar;
        self
    }
}

impl Field<Vec2, f32> for Brick {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let p = input * self.scale;
        let row = p.y.floor() as i32;
        let x = if row % 2 == 0 { p.x } else { p.x + self.offset };

        let fx = x.fract();
        let fy = p.y.fract();

        // Check if in mortar
        if fx < self.mortar || fx > 1.0 - self.mortar || fy < self.mortar || fy > 1.0 - self.mortar
        {
            0.0
        } else {
            1.0
        }
    }
}

/// Polka dots pattern.
#[derive(Debug, Clone, Copy)]
pub struct Dots {
    /// Grid scale.
    pub scale: f32,
    /// Dot radius.
    pub radius: f32,
}

impl Default for Dots {
    fn default() -> Self {
        Self {
            scale: 1.0,
            radius: 0.3,
        }
    }
}

impl Dots {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    pub fn with_radius(mut self, radius: f32) -> Self {
        self.radius = radius;
        self
    }
}

impl Field<Vec2, f32> for Dots {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let p = input * self.scale;
        let cell = Vec2::new(p.x.floor(), p.y.floor());
        let center = cell + Vec2::splat(0.5);
        let d = (p - center).length();
        if d < self.radius { 1.0 } else { 0.0 }
    }
}

/// Smooth dots (falloff from center).
#[derive(Debug, Clone, Copy)]
pub struct SmoothDots {
    pub scale: f32,
    pub radius: f32,
}

impl Default for SmoothDots {
    fn default() -> Self {
        Self {
            scale: 1.0,
            radius: 0.4,
        }
    }
}

impl SmoothDots {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    pub fn with_radius(mut self, radius: f32) -> Self {
        self.radius = radius;
        self
    }
}

impl Field<Vec2, f32> for SmoothDots {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let p = input * self.scale;
        let cell = Vec2::new(p.x.floor(), p.y.floor());
        let center = cell + Vec2::splat(0.5);
        let d = (p - center).length();
        (1.0 - d / self.radius).clamp(0.0, 1.0)
    }
}

/// Voronoi/Cellular noise - returns distance to nearest cell center.
#[derive(Debug, Clone, Copy)]
pub struct Voronoi {
    pub scale: f32,
    pub seed: i32,
}

impl Default for Voronoi {
    fn default() -> Self {
        Self {
            scale: 1.0,
            seed: 0,
        }
    }
}

impl Voronoi {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    pub fn with_seed(mut self, seed: i32) -> Self {
        self.seed = seed;
        self
    }

    /// Hash function for cell randomization.
    pub fn hash(x: i32, y: i32, seed: i32) -> Vec2 {
        // Simple hash based on prime multipliers
        let n = (x.wrapping_mul(374761393))
            .wrapping_add(y.wrapping_mul(668265263))
            .wrapping_add(seed.wrapping_mul(1013904223));
        let n = (n ^ (n >> 13)).wrapping_mul(1274126177);
        let fx = ((n & 0xFFFF) as f32) / 65535.0;
        let fy = (((n >> 16) & 0xFFFF) as f32) / 65535.0;
        Vec2::new(fx, fy)
    }
}

impl Field<Vec2, f32> for Voronoi {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let p = input * self.scale;
        let cell = Vec2::new(p.x.floor(), p.y.floor());
        let mut min_dist = f32::MAX;

        // Check 3x3 neighborhood
        for dy in -1..=1 {
            for dx in -1..=1 {
                let neighbor = cell + Vec2::new(dx as f32, dy as f32);
                let offset = Self::hash(neighbor.x as i32, neighbor.y as i32, self.seed);
                let point = neighbor + offset;
                let dist = (p - point).length();
                min_dist = min_dist.min(dist);
            }
        }

        // Normalize roughly to 0-1 (max distance in a cell is ~sqrt(2))
        (min_dist / 1.5).clamp(0.0, 1.0)
    }
}

/// Voronoi that returns cell ID (for coloring).
#[derive(Debug, Clone, Copy)]
pub struct VoronoiId {
    pub scale: f32,
    pub seed: i32,
}

impl Default for VoronoiId {
    fn default() -> Self {
        Self {
            scale: 1.0,
            seed: 0,
        }
    }
}

impl VoronoiId {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }
}

impl Field<Vec2, f32> for VoronoiId {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let p = input * self.scale;
        let cell = Vec2::new(p.x.floor(), p.y.floor());
        let mut min_dist = f32::MAX;
        let mut closest_cell = cell;

        for dy in -1..=1 {
            for dx in -1..=1 {
                let neighbor = cell + Vec2::new(dx as f32, dy as f32);
                let offset = Voronoi::hash(neighbor.x as i32, neighbor.y as i32, self.seed);
                let point = neighbor + offset;
                let dist = (p - point).length();
                if dist < min_dist {
                    min_dist = dist;
                    closest_cell = neighbor;
                }
            }
        }

        // Return a pseudo-random value based on cell coordinates
        let h = Voronoi::hash(
            closest_cell.x as i32,
            closest_cell.y as i32,
            self.seed + 12345,
        );
        h.x
    }
}

/// Domain warping - distorts input coordinates using another field.
pub struct Warp<F, D> {
    pub field: F,
    pub displacement: D,
    pub amount: f32,
}

impl<F, D> Warp<F, D> {
    pub fn new(field: F, displacement: D, amount: f32) -> Self {
        Self {
            field,
            displacement,
            amount,
        }
    }
}

impl<F, D> Field<Vec2, f32> for Warp<F, D>
where
    F: Field<Vec2, f32>,
    D: Field<Vec2, Vec2>,
{
    fn sample(&self, input: Vec2, ctx: &EvalContext) -> f32 {
        let offset = self.displacement.sample(input, ctx);
        let warped = input + offset * self.amount;
        self.field.sample(warped, ctx)
    }
}

/// Creates displacement from two scalar fields (for x and y).
pub struct Displacement<X, Y> {
    pub x: X,
    pub y: Y,
}

impl<X, Y> Displacement<X, Y> {
    pub fn new(x: X, y: Y) -> Self {
        Self { x, y }
    }
}

impl<X, Y> Field<Vec2, Vec2> for Displacement<X, Y>
where
    X: Field<Vec2, f32>,
    Y: Field<Vec2, f32>,
{
    fn sample(&self, input: Vec2, ctx: &EvalContext) -> Vec2 {
        Vec2::new(self.x.sample(input, ctx), self.y.sample(input, ctx))
    }
}

/// Distance field - returns distance from a point.
#[derive(Debug, Clone, Copy)]
pub struct DistancePoint {
    pub point: Vec2,
}

impl DistancePoint {
    pub fn new(point: Vec2) -> Self {
        Self { point }
    }
}

impl Field<Vec2, f32> for DistancePoint {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        (input - self.point).length()
    }
}

/// Distance field - returns distance from a line segment.
#[derive(Debug, Clone, Copy)]
pub struct DistanceLine {
    pub a: Vec2,
    pub b: Vec2,
}

impl DistanceLine {
    pub fn new(a: Vec2, b: Vec2) -> Self {
        Self { a, b }
    }
}

impl Field<Vec2, f32> for DistanceLine {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let pa = input - self.a;
        let ba = self.b - self.a;
        let t = (pa.dot(ba) / ba.dot(ba)).clamp(0.0, 1.0);
        (pa - ba * t).length()
    }
}

/// Distance field - returns distance from a circle.
#[derive(Debug, Clone, Copy)]
pub struct DistanceCircle {
    pub center: Vec2,
    pub radius: f32,
}

impl DistanceCircle {
    pub fn new(center: Vec2, radius: f32) -> Self {
        Self { center, radius }
    }
}

impl Field<Vec2, f32> for DistanceCircle {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        (input - self.center).length() - self.radius
    }
}

/// Distance field - returns distance from a box.
#[derive(Debug, Clone, Copy)]
pub struct DistanceBox {
    pub center: Vec2,
    pub half_size: Vec2,
}

impl DistanceBox {
    pub fn new(center: Vec2, half_size: Vec2) -> Self {
        Self { center, half_size }
    }
}

impl Field<Vec2, f32> for DistanceBox {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let p = (input - self.center).abs();
        let d = p - self.half_size;
        d.max(Vec2::ZERO).length() + d.x.max(d.y).min(0.0)
    }
}

// ============================================================================
// SDF operations
// ============================================================================

/// Union of two SDFs (min).
pub struct SdfUnion<A, B> {
    pub a: A,
    pub b: B,
}

impl<A, B> SdfUnion<A, B> {
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<I: Clone, A, B> Field<I, f32> for SdfUnion<A, B>
where
    A: Field<I, f32>,
    B: Field<I, f32>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> f32 {
        let a = self.a.sample(input.clone(), ctx);
        let b = self.b.sample(input, ctx);
        a.min(b)
    }
}

/// Intersection of two SDFs (max).
pub struct SdfIntersection<A, B> {
    pub a: A,
    pub b: B,
}

impl<A, B> SdfIntersection<A, B> {
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<I: Clone, A, B> Field<I, f32> for SdfIntersection<A, B>
where
    A: Field<I, f32>,
    B: Field<I, f32>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> f32 {
        let a = self.a.sample(input.clone(), ctx);
        let b = self.b.sample(input, ctx);
        a.max(b)
    }
}

/// Subtraction of SDF B from A.
pub struct SdfSubtraction<A, B> {
    pub a: A,
    pub b: B,
}

impl<A, B> SdfSubtraction<A, B> {
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<I: Clone, A, B> Field<I, f32> for SdfSubtraction<A, B>
where
    A: Field<I, f32>,
    B: Field<I, f32>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> f32 {
        let a = self.a.sample(input.clone(), ctx);
        let b = self.b.sample(input, ctx);
        a.max(-b)
    }
}

/// Smooth union of two SDFs using polynomial smooth min.
pub struct SdfSmoothUnion<A, B> {
    pub a: A,
    pub b: B,
    pub k: f32,
}

impl<A, B> SdfSmoothUnion<A, B> {
    pub fn new(a: A, b: B, k: f32) -> Self {
        Self { a, b, k }
    }
}

impl<I: Clone, A, B> Field<I, f32> for SdfSmoothUnion<A, B>
where
    A: Field<I, f32>,
    B: Field<I, f32>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> f32 {
        let a = self.a.sample(input.clone(), ctx);
        let b = self.b.sample(input, ctx);
        smooth_min(a, b, self.k)
    }
}

/// Smooth intersection of two SDFs.
pub struct SdfSmoothIntersection<A, B> {
    pub a: A,
    pub b: B,
    pub k: f32,
}

impl<A, B> SdfSmoothIntersection<A, B> {
    pub fn new(a: A, b: B, k: f32) -> Self {
        Self { a, b, k }
    }
}

impl<I: Clone, A, B> Field<I, f32> for SdfSmoothIntersection<A, B>
where
    A: Field<I, f32>,
    B: Field<I, f32>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> f32 {
        let a = self.a.sample(input.clone(), ctx);
        let b = self.b.sample(input, ctx);
        smooth_max(a, b, self.k)
    }
}

/// Smooth subtraction of SDF B from A.
pub struct SdfSmoothSubtraction<A, B> {
    pub a: A,
    pub b: B,
    pub k: f32,
}

impl<A, B> SdfSmoothSubtraction<A, B> {
    pub fn new(a: A, b: B, k: f32) -> Self {
        Self { a, b, k }
    }
}

impl<I: Clone, A, B> Field<I, f32> for SdfSmoothSubtraction<A, B>
where
    A: Field<I, f32>,
    B: Field<I, f32>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> f32 {
        let a = self.a.sample(input.clone(), ctx);
        let b = self.b.sample(input, ctx);
        smooth_max(a, -b, self.k)
    }
}

/// Polynomial smooth minimum.
fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
    if k <= 0.0 {
        return a.min(b);
    }
    let h = (k - (a - b).abs()).max(0.0) / k;
    a.min(b) - h * h * k * 0.25
}

/// Polynomial smooth maximum.
fn smooth_max(a: f32, b: f32, k: f32) -> f32 {
    -smooth_min(-a, -b, k)
}

/// Rounds/expands an SDF by a radius.
pub struct SdfRound<F> {
    pub field: F,
    pub radius: f32,
}

impl<F> SdfRound<F> {
    pub fn new(field: F, radius: f32) -> Self {
        Self { field, radius }
    }
}

impl<I, F: Field<I, f32>> Field<I, f32> for SdfRound<F> {
    fn sample(&self, input: I, ctx: &EvalContext) -> f32 {
        self.field.sample(input, ctx) - self.radius
    }
}

/// Annular (ring/shell) version of an SDF.
pub struct SdfAnnular<F> {
    pub field: F,
    pub thickness: f32,
}

impl<F> SdfAnnular<F> {
    pub fn new(field: F, thickness: f32) -> Self {
        Self { field, thickness }
    }
}

impl<I, F: Field<I, f32>> Field<I, f32> for SdfAnnular<F> {
    fn sample(&self, input: I, ctx: &EvalContext) -> f32 {
        self.field.sample(input, ctx).abs() - self.thickness
    }
}

// ============================================================================
// Domain modifiers
// ============================================================================

/// Twists space around the Y axis (for 3D fields).
pub struct Twist<F> {
    pub field: F,
    pub amount: f32,
}

impl<F> Twist<F> {
    pub fn new(field: F, amount: f32) -> Self {
        Self { field, amount }
    }
}

impl<O, F: Field<Vec3, O>> Field<Vec3, O> for Twist<F> {
    fn sample(&self, input: Vec3, ctx: &EvalContext) -> O {
        let angle = input.y * self.amount;
        let cos = angle.cos();
        let sin = angle.sin();
        let twisted = Vec3::new(
            input.x * cos - input.z * sin,
            input.y,
            input.x * sin + input.z * cos,
        );
        self.field.sample(twisted, ctx)
    }
}

/// Bends space around the Y axis (for 3D fields).
pub struct Bend<F> {
    pub field: F,
    pub amount: f32,
}

impl<F> Bend<F> {
    pub fn new(field: F, amount: f32) -> Self {
        Self { field, amount }
    }
}

impl<O, F: Field<Vec3, O>> Field<Vec3, O> for Bend<F> {
    fn sample(&self, input: Vec3, ctx: &EvalContext) -> O {
        let angle = input.x * self.amount;
        let cos = angle.cos();
        let sin = angle.sin();
        let bent = Vec3::new(
            cos * input.x - sin * input.y,
            sin * input.x + cos * input.y,
            input.z,
        );
        self.field.sample(bent, ctx)
    }
}

/// Repeats space infinitely.
pub struct Repeat<F> {
    pub field: F,
    pub period: Vec2,
}

impl<F> Repeat<F> {
    pub fn new(field: F, period: Vec2) -> Self {
        Self { field, period }
    }
}

impl<O, F: Field<Vec2, O>> Field<Vec2, O> for Repeat<F> {
    fn sample(&self, input: Vec2, ctx: &EvalContext) -> O {
        let repeated = Vec2::new(
            ((input.x % self.period.x) + self.period.x) % self.period.x - self.period.x * 0.5,
            ((input.y % self.period.y) + self.period.y) % self.period.y - self.period.y * 0.5,
        );
        self.field.sample(repeated, ctx)
    }
}

/// Repeats space infinitely (3D).
pub struct Repeat3D<F> {
    pub field: F,
    pub period: Vec3,
}

impl<F> Repeat3D<F> {
    pub fn new(field: F, period: Vec3) -> Self {
        Self { field, period }
    }
}

impl<O, F: Field<Vec3, O>> Field<Vec3, O> for Repeat3D<F> {
    fn sample(&self, input: Vec3, ctx: &EvalContext) -> O {
        let repeated = Vec3::new(
            ((input.x % self.period.x) + self.period.x) % self.period.x - self.period.x * 0.5,
            ((input.y % self.period.y) + self.period.y) % self.period.y - self.period.y * 0.5,
            ((input.z % self.period.z) + self.period.z) % self.period.z - self.period.z * 0.5,
        );
        self.field.sample(repeated, ctx)
    }
}

/// Rotates 2D input coordinates.
pub struct Rotate2D<F> {
    pub field: F,
    pub angle: f32,
}

impl<F> Rotate2D<F> {
    pub fn new(field: F, angle: f32) -> Self {
        Self { field, angle }
    }
}

impl<O, F: Field<Vec2, O>> Field<Vec2, O> for Rotate2D<F> {
    fn sample(&self, input: Vec2, ctx: &EvalContext) -> O {
        let cos = self.angle.cos();
        let sin = self.angle.sin();
        let rotated = Vec2::new(input.x * cos - input.y * sin, input.x * sin + input.y * cos);
        self.field.sample(rotated, ctx)
    }
}

/// Mirrors space across an axis.
pub struct Mirror<F> {
    pub field: F,
    pub axis: Vec2,
}

impl<F> Mirror<F> {
    pub fn new(field: F, axis: Vec2) -> Self {
        Self {
            field,
            axis: axis.normalize(),
        }
    }

    pub fn x(field: F) -> Self {
        Self::new(field, Vec2::X)
    }

    pub fn y(field: F) -> Self {
        Self::new(field, Vec2::Y)
    }
}

impl<O, F: Field<Vec2, O>> Field<Vec2, O> for Mirror<F> {
    fn sample(&self, input: Vec2, ctx: &EvalContext) -> O {
        let d = input.dot(self.axis);
        let mirrored = if d < 0.0 {
            input - 2.0 * d * self.axis
        } else {
            input
        };
        self.field.sample(mirrored, ctx)
    }
}

// ============================================================================
// Function adapter
// ============================================================================

/// Function adapter - wraps a closure as a field.
pub struct FnField<I, O, F> {
    f: F,
    _phantom: PhantomData<(I, O)>,
}

impl<I, O, F: Fn(I, &EvalContext) -> O> FnField<I, O, F> {
    pub fn new(f: F) -> Self {
        Self {
            f,
            _phantom: PhantomData,
        }
    }
}

impl<I, O, F: Fn(I, &EvalContext) -> O> Field<I, O> for FnField<I, O, F> {
    fn sample(&self, input: I, ctx: &EvalContext) -> O {
        (self.f)(input, ctx)
    }
}

/// Creates a field from a closure.
pub fn from_fn<I, O, F: Fn(I, &EvalContext) -> O>(f: F) -> FnField<I, O, F> {
    FnField::new(f)
}

// ============================================================================
// Metaballs
// ============================================================================

/// A single metaball (blob) center for use in metaball fields.
#[derive(Debug, Clone, Copy)]
pub struct Metaball {
    /// Center position.
    pub center: Vec3,
    /// Radius of influence.
    pub radius: f32,
    /// Strength (default 1.0).
    pub strength: f32,
}

impl Metaball {
    /// Creates a new metaball at the given position with the given radius.
    pub fn new(center: Vec3, radius: f32) -> Self {
        Self {
            center,
            radius,
            strength: 1.0,
        }
    }

    /// Creates a 2D metaball (z=0).
    pub fn new_2d(center: Vec2, radius: f32) -> Self {
        Self::new(center.extend(0.0), radius)
    }

    /// Sets the strength of this metaball.
    pub fn with_strength(mut self, strength: f32) -> Self {
        self.strength = strength;
        self
    }
}

/// 2D metaball field - computes the sum of influences at each point.
///
/// The field value is the sum of `strength * f(distance)` for each ball,
/// where f is the falloff function. Values above 1.0 are typically "inside"
/// the merged surface.
///
/// # Example
///
/// ```
/// use glam::Vec2;
/// use rhizome_resin_field::{Field, EvalContext, Metaball, Metaballs2D};
///
/// let balls = vec![
///     Metaball::new_2d(Vec2::new(0.0, 0.0), 1.0),
///     Metaball::new_2d(Vec2::new(1.5, 0.0), 1.0),
/// ];
///
/// let field = Metaballs2D::new(balls);
/// let ctx = EvalContext::new();
///
/// // Sample the field - values > 1.0 are "inside"
/// let value = field.sample(Vec2::new(0.75, 0.0), &ctx);
/// ```
#[derive(Debug, Clone)]
pub struct Metaballs2D {
    balls: Vec<Metaball>,
    threshold: f32,
}

impl Metaballs2D {
    /// Creates a new 2D metaball field.
    pub fn new(balls: Vec<Metaball>) -> Self {
        Self {
            balls,
            threshold: 1.0,
        }
    }

    /// Sets the threshold for the implicit surface (default 1.0).
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Adds a metaball to the field.
    pub fn add_ball(&mut self, ball: Metaball) {
        self.balls.push(ball);
    }

    /// Returns the threshold value.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Returns a reference to the balls.
    pub fn balls(&self) -> &[Metaball] {
        &self.balls
    }
}

impl Field<Vec2, f32> for Metaballs2D {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let mut sum = 0.0;

        for ball in &self.balls {
            let ball_pos = Vec2::new(ball.center.x, ball.center.y);
            let dist_sq = (input - ball_pos).length_squared();
            let radius_sq = ball.radius * ball.radius;

            if dist_sq < 0.0001 {
                // Very close to center - return large value
                sum += ball.strength * 100.0;
            } else {
                // Classic metaball falloff: r^2 / d^2
                sum += ball.strength * radius_sq / dist_sq;
            }
        }

        sum
    }
}

/// 3D metaball field.
///
/// Similar to Metaballs2D but operates in 3D space.
#[derive(Debug, Clone)]
pub struct Metaballs3D {
    balls: Vec<Metaball>,
    threshold: f32,
}

impl Metaballs3D {
    /// Creates a new 3D metaball field.
    pub fn new(balls: Vec<Metaball>) -> Self {
        Self {
            balls,
            threshold: 1.0,
        }
    }

    /// Sets the threshold for the implicit surface (default 1.0).
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Adds a metaball to the field.
    pub fn add_ball(&mut self, ball: Metaball) {
        self.balls.push(ball);
    }

    /// Returns the threshold value.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Returns a reference to the balls.
    pub fn balls(&self) -> &[Metaball] {
        &self.balls
    }
}

impl Field<Vec3, f32> for Metaballs3D {
    fn sample(&self, input: Vec3, _ctx: &EvalContext) -> f32 {
        let mut sum = 0.0;

        for ball in &self.balls {
            let dist_sq = (input - ball.center).length_squared();
            let radius_sq = ball.radius * ball.radius;

            if dist_sq < 0.0001 {
                sum += ball.strength * 100.0;
            } else {
                // Classic metaball falloff: r^2 / d^2
                sum += ball.strength * radius_sq / dist_sq;
            }
        }

        sum
    }
}

/// Converts a 2D metaball field to an SDF-like representation.
///
/// Returns negative values inside (where field > threshold),
/// positive values outside. This makes it compatible with SDF
/// operations like smooth union.
#[derive(Debug, Clone)]
pub struct MetaballSdf2D {
    field: Metaballs2D,
}

impl MetaballSdf2D {
    /// Creates a new SDF from a metaball field.
    pub fn new(field: Metaballs2D) -> Self {
        Self { field }
    }

    /// Creates an SDF from balls directly.
    pub fn from_balls(balls: Vec<Metaball>) -> Self {
        Self::new(Metaballs2D::new(balls))
    }
}

impl Field<Vec2, f32> for MetaballSdf2D {
    fn sample(&self, input: Vec2, ctx: &EvalContext) -> f32 {
        let value = self.field.sample(input, ctx);
        // Convert to SDF-like: negative inside, positive outside
        // When value > threshold, we're inside, so return negative
        self.field.threshold - value
    }
}

/// Converts a 3D metaball field to an SDF-like representation.
#[derive(Debug, Clone)]
pub struct MetaballSdf3D {
    field: Metaballs3D,
}

impl MetaballSdf3D {
    /// Creates a new SDF from a metaball field.
    pub fn new(field: Metaballs3D) -> Self {
        Self { field }
    }

    /// Creates an SDF from balls directly.
    pub fn from_balls(balls: Vec<Metaball>) -> Self {
        Self::new(Metaballs3D::new(balls))
    }
}

impl Field<Vec3, f32> for MetaballSdf3D {
    fn sample(&self, input: Vec3, ctx: &EvalContext) -> f32 {
        let value = self.field.sample(input, ctx);
        self.field.threshold - value
    }
}

// ============================================================================
// Terrain erosion simulation
// ============================================================================

/// A heightmap for terrain erosion simulation.
///
/// Stores height values as a 2D grid.
#[derive(Clone)]
pub struct Heightmap {
    /// Height values (row-major).
    pub data: Vec<f32>,
    /// Width of the heightmap.
    pub width: usize,
    /// Height of the heightmap.
    pub height: usize,
}

impl Heightmap {
    /// Creates a new heightmap filled with zeros.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            data: vec![0.0; width * height],
            width,
            height,
        }
    }

    /// Creates a heightmap from a Field.
    pub fn from_field<F: Field<Vec2, f32>>(
        field: &F,
        width: usize,
        height: usize,
        scale: f32,
    ) -> Self {
        let ctx = EvalContext::new();
        let mut data = vec![0.0; width * height];

        for y in 0..height {
            for x in 0..width {
                let pos = Vec2::new(x as f32 * scale, y as f32 * scale);
                data[y * width + x] = field.sample(pos, &ctx);
            }
        }

        Self {
            data,
            width,
            height,
        }
    }

    /// Gets the height at grid coordinates.
    pub fn get(&self, x: usize, y: usize) -> f32 {
        if x < self.width && y < self.height {
            self.data[y * self.width + x]
        } else {
            0.0
        }
    }

    /// Sets the height at grid coordinates.
    pub fn set(&mut self, x: usize, y: usize, value: f32) {
        if x < self.width && y < self.height {
            self.data[y * self.width + x] = value;
        }
    }

    /// Gets the height with bilinear interpolation.
    pub fn sample(&self, x: f32, y: f32) -> f32 {
        let x0 = x.floor() as i32;
        let y0 = y.floor() as i32;
        let fx = x - x0 as f32;
        let fy = y - y0 as f32;

        let h00 = self.get_clamped(x0, y0);
        let h10 = self.get_clamped(x0 + 1, y0);
        let h01 = self.get_clamped(x0, y0 + 1);
        let h11 = self.get_clamped(x0 + 1, y0 + 1);

        let h0 = h00 * (1.0 - fx) + h10 * fx;
        let h1 = h01 * (1.0 - fx) + h11 * fx;

        h0 * (1.0 - fy) + h1 * fy
    }

    /// Gets height with boundary clamping.
    fn get_clamped(&self, x: i32, y: i32) -> f32 {
        let x = x.clamp(0, self.width as i32 - 1) as usize;
        let y = y.clamp(0, self.height as i32 - 1) as usize;
        self.data[y * self.width + x]
    }

    /// Computes the gradient at a position.
    pub fn gradient(&self, x: f32, y: f32) -> Vec2 {
        let h = 0.5;
        let dx = (self.sample(x + h, y) - self.sample(x - h, y)) / (2.0 * h);
        let dy = (self.sample(x, y + h) - self.sample(x, y - h)) / (2.0 * h);
        Vec2::new(dx, dy)
    }

    /// Adds sediment at a position (with bilinear distribution).
    fn add_sediment(&mut self, x: f32, y: f32, amount: f32) {
        let x0 = x.floor() as i32;
        let y0 = y.floor() as i32;
        let fx = x - x0 as f32;
        let fy = y - y0 as f32;

        // Bilinear weights
        let w00 = (1.0 - fx) * (1.0 - fy);
        let w10 = fx * (1.0 - fy);
        let w01 = (1.0 - fx) * fy;
        let w11 = fx * fy;

        self.add_clamped(x0, y0, amount * w00);
        self.add_clamped(x0 + 1, y0, amount * w10);
        self.add_clamped(x0, y0 + 1, amount * w01);
        self.add_clamped(x0 + 1, y0 + 1, amount * w11);
    }

    /// Adds to height with boundary clamping.
    fn add_clamped(&mut self, x: i32, y: i32, amount: f32) {
        if x >= 0 && x < self.width as i32 && y >= 0 && y < self.height as i32 {
            let idx = y as usize * self.width + x as usize;
            self.data[idx] += amount;
        }
    }

    /// Gets the minimum and maximum heights.
    pub fn bounds(&self) -> (f32, f32) {
        let min = self.data.iter().copied().fold(f32::INFINITY, f32::min);
        let max = self.data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        (min, max)
    }

    /// Normalizes heights to 0-1 range.
    pub fn normalize(&mut self) {
        let (min, max) = self.bounds();
        let range = max - min;
        if range > 0.0001 {
            for h in &mut self.data {
                *h = (*h - min) / range;
            }
        }
    }
}

/// Configuration for hydraulic erosion simulation.
#[derive(Clone, Copy, Debug)]
pub struct HydraulicErosionConfig {
    /// Number of water droplets to simulate.
    pub iterations: usize,
    /// Maximum steps per droplet lifetime.
    pub max_lifetime: usize,
    /// Initial water volume per droplet.
    pub initial_water: f32,
    /// Initial speed of droplets.
    pub initial_speed: f32,
    /// Inertia factor (0 = follow gradient exactly, 1 = maintain direction).
    pub inertia: f32,
    /// Minimum slope for erosion to occur.
    pub min_slope: f32,
    /// Sediment capacity multiplier.
    pub capacity_factor: f32,
    /// Rate of sediment pickup.
    pub erosion_rate: f32,
    /// Rate of sediment deposition.
    pub deposition_rate: f32,
    /// Water evaporation rate per step.
    pub evaporation_rate: f32,
    /// Gravity strength.
    pub gravity: f32,
    /// Erosion brush radius.
    pub brush_radius: usize,
}

impl Default for HydraulicErosionConfig {
    fn default() -> Self {
        Self {
            iterations: 10000,
            max_lifetime: 64,
            initial_water: 1.0,
            initial_speed: 1.0,
            inertia: 0.3,
            min_slope: 0.01,
            capacity_factor: 4.0,
            erosion_rate: 0.3,
            deposition_rate: 0.3,
            evaporation_rate: 0.01,
            gravity: 4.0,
            brush_radius: 3,
        }
    }
}

/// Simulates hydraulic erosion on a heightmap.
///
/// Water droplets flow downhill, picking up sediment and depositing it
/// as they slow down, creating realistic river valleys and gullies.
pub fn hydraulic_erosion(heightmap: &mut Heightmap, config: &HydraulicErosionConfig, seed: u64) {
    let mut rng = SimpleRng::new(seed);
    let width = heightmap.width;
    let height = heightmap.height;

    // Pre-compute erosion brush weights
    let brush = compute_erosion_brush(config.brush_radius);

    for _ in 0..config.iterations {
        // Random starting position
        let mut x = rng.next_f32() * (width - 1) as f32;
        let mut y = rng.next_f32() * (height - 1) as f32;

        // Initial direction (random)
        let angle = rng.next_f32() * std::f32::consts::TAU;
        let mut dir = Vec2::new(angle.cos(), angle.sin());

        let mut speed = config.initial_speed;
        let mut water = config.initial_water;
        let mut sediment = 0.0;

        for _ in 0..config.max_lifetime {
            let xi = x as usize;
            let yi = y as usize;

            // Get gradient at current position
            let gradient = heightmap.gradient(x, y);
            let gradient_len = gradient.length();

            // Update direction with inertia
            if gradient_len > 0.0001 {
                dir = dir * config.inertia - gradient.normalize() * (1.0 - config.inertia);
                let dir_len = dir.length();
                if dir_len > 0.0001 {
                    dir /= dir_len;
                }
            }

            // Move to next position
            let new_x = x + dir.x;
            let new_y = y + dir.y;

            // Check boundaries
            if new_x < 0.0
                || new_x >= (width - 1) as f32
                || new_y < 0.0
                || new_y >= (height - 1) as f32
            {
                break;
            }

            // Calculate height difference
            let old_height = heightmap.sample(x, y);
            let new_height = heightmap.sample(new_x, new_y);
            let height_diff = new_height - old_height;

            // Calculate sediment capacity
            let slope = (-height_diff).max(config.min_slope);
            let capacity = slope * speed * water * config.capacity_factor;

            if sediment > capacity || height_diff > 0.0 {
                // Deposit sediment
                let deposit_amount = if height_diff > 0.0 {
                    // Moving uphill - deposit all sediment
                    height_diff.min(sediment)
                } else {
                    // Deposit excess sediment
                    (sediment - capacity) * config.deposition_rate
                };

                sediment -= deposit_amount;
                heightmap.add_sediment(x, y, deposit_amount);
            } else {
                // Erode terrain
                let erode_amount = ((capacity - sediment) * config.erosion_rate).min(-height_diff);

                // Apply erosion with brush
                apply_erosion_brush(heightmap, xi, yi, erode_amount, &brush);
                sediment += erode_amount;
            }

            // Update physics
            speed = (speed * speed + height_diff.abs() * config.gravity)
                .sqrt()
                .max(0.1);
            water *= 1.0 - config.evaporation_rate;

            // Stop if water dried up
            if water < 0.01 {
                break;
            }

            x = new_x;
            y = new_y;
        }
    }
}

/// Brush weights for erosion.
struct ErosionBrush {
    offsets: Vec<(i32, i32)>,
    weights: Vec<f32>,
}

fn compute_erosion_brush(radius: usize) -> ErosionBrush {
    let mut offsets = Vec::new();
    let mut weights = Vec::new();
    let mut total_weight = 0.0;
    let r = radius as i32;

    for dy in -r..=r {
        for dx in -r..=r {
            let dist = ((dx * dx + dy * dy) as f32).sqrt();
            if dist <= radius as f32 {
                let weight = 1.0 - dist / (radius as f32 + 1.0);
                offsets.push((dx, dy));
                weights.push(weight);
                total_weight += weight;
            }
        }
    }

    // Normalize weights
    if total_weight > 0.0 {
        for w in &mut weights {
            *w /= total_weight;
        }
    }

    ErosionBrush { offsets, weights }
}

fn apply_erosion_brush(
    heightmap: &mut Heightmap,
    cx: usize,
    cy: usize,
    amount: f32,
    brush: &ErosionBrush,
) {
    for (i, (dx, dy)) in brush.offsets.iter().enumerate() {
        let x = cx as i32 + dx;
        let y = cy as i32 + dy;
        if x >= 0 && x < heightmap.width as i32 && y >= 0 && y < heightmap.height as i32 {
            let idx = y as usize * heightmap.width + x as usize;
            heightmap.data[idx] -= amount * brush.weights[i];
        }
    }
}

/// Configuration for thermal erosion simulation.
#[derive(Clone, Copy, Debug)]
pub struct ThermalErosionConfig {
    /// Number of iterations.
    pub iterations: usize,
    /// Maximum slope angle (as tangent) before material slides.
    pub talus_angle: f32,
    /// Rate of material transfer per iteration.
    pub transfer_rate: f32,
}

impl Default for ThermalErosionConfig {
    fn default() -> Self {
        Self {
            iterations: 50,
            talus_angle: 0.8, // ~40 degrees
            transfer_rate: 0.5,
        }
    }
}

/// Simulates thermal erosion (talus slopes).
///
/// Material slides down when slopes exceed the talus angle,
/// creating realistic scree slopes and smoothing sharp features.
pub fn thermal_erosion(heightmap: &mut Heightmap, config: &ThermalErosionConfig) {
    let width = heightmap.width;
    let height = heightmap.height;

    // Neighbor offsets
    let neighbors: [(i32, i32); 8] = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];

    // Distance for each neighbor (diagonal = sqrt(2))
    let distances: [f32; 8] = [
        std::f32::consts::SQRT_2,
        1.0,
        std::f32::consts::SQRT_2,
        1.0,
        1.0,
        std::f32::consts::SQRT_2,
        1.0,
        std::f32::consts::SQRT_2,
    ];

    let mut deltas = vec![0.0; width * height];

    for _ in 0..config.iterations {
        // Clear deltas
        deltas.fill(0.0);

        // Calculate transfers
        for y in 0..height {
            for x in 0..width {
                let h = heightmap.get(x, y);
                let mut total_diff: f32 = 0.0;
                let mut max_diff: f32 = 0.0;

                // Find total height difference to lower neighbors
                for (i, (dx, dy)) in neighbors.iter().enumerate() {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;

                    if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                        let nh = heightmap.get(nx as usize, ny as usize);
                        let slope = (h - nh) / distances[i];

                        if slope > config.talus_angle {
                            let diff = slope - config.talus_angle;
                            total_diff += diff;
                            max_diff = max_diff.max(diff);
                        }
                    }
                }

                if total_diff > 0.0 {
                    // Distribute material to lower neighbors
                    let transfer_amount = max_diff * config.transfer_rate * 0.5;

                    for (i, (dx, dy)) in neighbors.iter().enumerate() {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;

                        if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                            let nh = heightmap.get(nx as usize, ny as usize);
                            let slope = (h - nh) / distances[i];

                            if slope > config.talus_angle {
                                let diff = slope - config.talus_angle;
                                let ratio = diff / total_diff;
                                let amount = transfer_amount * ratio;

                                deltas[y * width + x] -= amount;
                                deltas[ny as usize * width + nx as usize] += amount;
                            }
                        }
                    }
                }
            }
        }

        // Apply deltas
        for (i, delta) in deltas.iter().enumerate() {
            heightmap.data[i] += delta;
        }
    }
}

/// Simple random number generator for erosion simulation.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        (self.next() as f32) / (u64::MAX as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_field() {
        let field = Constant::new(42.0f32);
        let ctx = EvalContext::new();

        assert_eq!(field.sample(Vec2::ZERO, &ctx), 42.0);
        assert_eq!(field.sample(Vec2::new(100.0, 100.0), &ctx), 42.0);
    }

    #[test]
    fn test_coordinates_field() {
        let field = Coordinates;
        let ctx = EvalContext::new();

        assert_eq!(field.sample(Vec2::new(1.0, 2.0), &ctx), Vec2::new(1.0, 2.0));
    }

    #[test]
    fn test_scale_combinator() {
        let field = <Coordinates as Field<Vec2, Vec2>>::scale(Coordinates, 2.0);
        let ctx = EvalContext::new();

        // Scaling input by 2 means we query at 2x the position
        let result: Vec2 = field.sample(Vec2::new(1.0, 1.0), &ctx);
        assert_eq!(result, Vec2::new(2.0, 2.0));
    }

    #[test]
    fn test_translate_combinator() {
        let field = <Coordinates as Field<Vec2, Vec2>>::translate(Coordinates, Vec2::new(1.0, 1.0));
        let ctx = EvalContext::new();

        // Translating subtracts from input before sampling
        let result: Vec2 = field.sample(Vec2::new(2.0, 2.0), &ctx);
        assert_eq!(result, Vec2::new(1.0, 1.0));
    }

    #[test]
    fn test_map_combinator() {
        let field = <Constant<f32> as Field<Vec2, f32>>::map(Constant::new(5.0f32), |x| x * 2.0);
        let ctx = EvalContext::new();

        assert_eq!(field.sample(Vec2::ZERO, &ctx), 10.0);
    }

    #[test]
    fn test_add_combinator() {
        let a = Constant::new(3.0f32);
        let b = Constant::new(4.0f32);
        let field = <Constant<f32> as Field<Vec2, f32>>::add(a, b);
        let ctx = EvalContext::new();

        assert_eq!(field.sample(Vec2::ZERO, &ctx), 7.0);
    }

    #[test]
    fn test_perlin_field() {
        let field = Perlin2D::new().scale(4.0);
        let ctx = EvalContext::new();

        let v1 = field.sample(Vec2::new(0.0, 0.0), &ctx);
        let v2 = field.sample(Vec2::new(1.0, 1.0), &ctx);

        assert!((0.0..=1.0).contains(&v1));
        assert!((0.0..=1.0).contains(&v2));
    }

    #[test]
    fn test_checkerboard() {
        let field = Checkerboard::new();
        let ctx = EvalContext::new();

        assert_eq!(field.sample(Vec2::new(0.5, 0.5), &ctx), 1.0);
        assert_eq!(field.sample(Vec2::new(1.5, 0.5), &ctx), 0.0);
    }

    #[test]
    fn test_distance_circle() {
        let field = DistanceCircle::new(Vec2::ZERO, 1.0);
        let ctx = EvalContext::new();

        assert!(field.sample(Vec2::ZERO, &ctx) < 0.0);
        assert!((field.sample(Vec2::new(1.0, 0.0), &ctx) - 0.0).abs() < 0.001);
        assert!(field.sample(Vec2::new(2.0, 0.0), &ctx) > 0.0);
    }

    #[test]
    fn test_metaballs() {
        let balls = vec![Metaball::new_2d(Vec2::ZERO, 1.0)];
        let field = Metaballs2D::new(balls);
        let ctx = EvalContext::new();

        let center_val = field.sample(Vec2::ZERO, &ctx);
        assert!(center_val > 10.0);

        let radius_val = field.sample(Vec2::new(1.0, 0.0), &ctx);
        assert!((radius_val - 1.0).abs() < 0.01);
    }

    // Terrain generation tests

    #[test]
    fn test_terrain_basic() {
        let terrain = Terrain2D::new();
        let ctx = EvalContext::new();

        // Sample at various points
        let v1 = terrain.sample(Vec2::new(0.0, 0.0), &ctx);
        let v2 = terrain.sample(Vec2::new(1.0, 1.0), &ctx);
        let v3 = terrain.sample(Vec2::new(10.0, 10.0), &ctx);

        // All values should be in [0, 1] range
        assert!((0.0..=1.0).contains(&v1));
        assert!((0.0..=1.0).contains(&v2));
        assert!((0.0..=1.0).contains(&v3));

        // Values should vary (not all the same)
        assert!((v1 - v2).abs() > 0.001 || (v2 - v3).abs() > 0.001);
    }

    #[test]
    fn test_terrain_presets() {
        let ctx = EvalContext::new();
        let point = Vec2::new(0.5, 0.5);

        // Test all presets produce valid values
        let hills = Terrain2D::rolling_hills().sample(point, &ctx);
        let mountains = Terrain2D::mountains().sample(point, &ctx);
        let plains = Terrain2D::plains().sample(point, &ctx);
        let canyons = Terrain2D::canyons().sample(point, &ctx);

        assert!((0.0..=1.0).contains(&hills));
        assert!((0.0..=1.0).contains(&mountains));
        assert!((0.0..=1.0).contains(&plains));
        assert!((0.0..=1.0).contains(&canyons));
    }

    #[test]
    fn test_terrain_deterministic() {
        let ctx = EvalContext::new();
        let terrain1 = Terrain2D::new().with_seed(42);
        let terrain2 = Terrain2D::new().with_seed(42);

        let point = Vec2::new(0.5, 0.5);
        assert_eq!(terrain1.sample(point, &ctx), terrain2.sample(point, &ctx));
    }

    #[test]
    fn test_ridged_terrain() {
        let terrain = RidgedTerrain2D::new();
        let ctx = EvalContext::new();

        let v = terrain.sample(Vec2::new(0.5, 0.5), &ctx);
        assert!((0.0..=1.0).contains(&v));
    }

    #[test]
    fn test_billowy_terrain() {
        let terrain = BillowyTerrain2D::new();
        let ctx = EvalContext::new();

        let v = terrain.sample(Vec2::new(0.5, 0.5), &ctx);
        assert!((0.0..=1.0).contains(&v));
    }

    #[test]
    fn test_island_terrain() {
        let terrain = IslandTerrain2D::new().with_radius(1.0);
        let ctx = EvalContext::new();

        // Center should have terrain
        let center = terrain.sample(Vec2::ZERO, &ctx);
        assert!(center >= 0.0);

        // Far from center should be zero (outside island)
        let far = terrain.sample(Vec2::new(10.0, 10.0), &ctx);
        assert_eq!(far, 0.0);
    }

    #[test]
    fn test_terraced_terrain() {
        let base = Terrain2D::new();
        let terraced = TerracedTerrain2D::new(base, 5);
        let ctx = EvalContext::new();

        let v = terraced.sample(Vec2::new(0.5, 0.5), &ctx);
        assert!((0.0..=1.0).contains(&v));
    }

    #[test]
    fn test_terrain_builder() {
        let terrain = Terrain2D::new()
            .with_seed(123)
            .with_octaves(4)
            .with_lacunarity(2.5)
            .with_persistence(0.4)
            .with_scale(2.0)
            .with_exponent(1.5);

        assert_eq!(terrain.seed, 123);
        assert_eq!(terrain.octaves, 4);
        assert!((terrain.lacunarity - 2.5).abs() < 0.001);
        assert!((terrain.persistence - 0.4).abs() < 0.001);
        assert!((terrain.scale - 2.0).abs() < 0.001);
        assert!((terrain.exponent - 1.5).abs() < 0.001);
    }

    // ========================================================================
    // Heightmap tests
    // ========================================================================

    #[test]
    fn test_heightmap_new() {
        let hm = Heightmap::new(10, 20);
        assert_eq!(hm.width, 10);
        assert_eq!(hm.height, 20);
        assert_eq!(hm.data.len(), 200);
        assert!(hm.data.iter().all(|&h| h == 0.0));
    }

    #[test]
    fn test_heightmap_get_set() {
        let mut hm = Heightmap::new(10, 10);
        hm.set(5, 5, 1.0);
        assert_eq!(hm.get(5, 5), 1.0);
        assert_eq!(hm.get(0, 0), 0.0);

        // Out of bounds returns 0
        assert_eq!(hm.get(100, 100), 0.0);
    }

    #[test]
    fn test_heightmap_sample_bilinear() {
        let mut hm = Heightmap::new(4, 4);
        hm.set(1, 1, 0.0);
        hm.set(2, 1, 1.0);
        hm.set(1, 2, 0.0);
        hm.set(2, 2, 1.0);

        // Sample at integer position
        assert!((hm.sample(1.0, 1.0) - 0.0).abs() < 0.001);
        assert!((hm.sample(2.0, 1.0) - 1.0).abs() < 0.001);

        // Sample at midpoint - should interpolate
        let mid = hm.sample(1.5, 1.5);
        assert!((mid - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_heightmap_gradient() {
        let mut hm = Heightmap::new(10, 10);
        // Create a slope: height increases with x
        for y in 0..10 {
            for x in 0..10 {
                hm.set(x, y, x as f32);
            }
        }

        let grad = hm.gradient(5.0, 5.0);
        // Gradient should point in +x direction
        assert!(grad.x > 0.5);
        assert!(grad.y.abs() < 0.1);
    }

    #[test]
    fn test_heightmap_bounds() {
        let mut hm = Heightmap::new(5, 5);
        hm.set(0, 0, -10.0);
        hm.set(2, 2, 20.0);

        let (min, max) = hm.bounds();
        assert_eq!(min, -10.0);
        assert_eq!(max, 20.0);
    }

    #[test]
    fn test_heightmap_normalize() {
        let mut hm = Heightmap::new(5, 5);
        hm.set(0, 0, -10.0);
        hm.set(2, 2, 20.0);
        hm.set(4, 4, 5.0);

        hm.normalize();

        let (min, max) = hm.bounds();
        assert!((min - 0.0).abs() < 0.001);
        assert!((max - 1.0).abs() < 0.001);

        // Check specific values
        assert!((hm.get(0, 0) - 0.0).abs() < 0.001); // -10 -> 0
        assert!((hm.get(2, 2) - 1.0).abs() < 0.001); // 20 -> 1
        assert!((hm.get(4, 4) - 0.5).abs() < 0.001); // 5 -> 0.5
    }

    #[test]
    fn test_heightmap_from_field() {
        let field = Constant::new(0.5f32);
        let hm = Heightmap::from_field(&field, 10, 10, 1.0);

        assert_eq!(hm.width, 10);
        assert_eq!(hm.height, 10);
        assert!(hm.data.iter().all(|&h| (h - 0.5).abs() < 0.001));
    }

    // ========================================================================
    // Hydraulic erosion tests
    // ========================================================================

    #[test]
    fn test_hydraulic_erosion_config_default() {
        let config = HydraulicErosionConfig::default();
        assert_eq!(config.iterations, 10000);
        assert_eq!(config.max_lifetime, 64);
        assert_eq!(config.brush_radius, 3);
    }

    #[test]
    fn test_hydraulic_erosion_modifies_terrain() {
        // Create a simple cone terrain
        let mut hm = Heightmap::new(32, 32);
        let cx = 16.0;
        let cy = 16.0;

        for y in 0..32 {
            for x in 0..32 {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                hm.set(x, y, (16.0 - dist).max(0.0));
            }
        }

        let original_sum: f32 = hm.data.iter().sum();

        let config = HydraulicErosionConfig {
            iterations: 1000,
            max_lifetime: 30,
            brush_radius: 2,
            ..Default::default()
        };

        hydraulic_erosion(&mut hm, &config, 12345);

        // Terrain should be modified (not identical to original)
        let new_sum: f32 = hm.data.iter().sum();
        assert!(new_sum != original_sum, "terrain should be modified");

        // Most material should remain (within bounds)
        assert!(new_sum > 0.0, "terrain should have material");

        // Heights should still be reasonable
        assert!(hm.data.iter().all(|&h| h >= -5.0 && h < 25.0));
    }

    #[test]
    fn test_hydraulic_erosion_deterministic() {
        let create_terrain = || {
            let mut hm = Heightmap::new(16, 16);
            for y in 0..16 {
                for x in 0..16 {
                    hm.set(x, y, ((x + y) as f32 / 30.0).sin() * 5.0);
                }
            }
            hm
        };

        let config = HydraulicErosionConfig {
            iterations: 500,
            ..Default::default()
        };

        let mut hm1 = create_terrain();
        let mut hm2 = create_terrain();

        hydraulic_erosion(&mut hm1, &config, 42);
        hydraulic_erosion(&mut hm2, &config, 42);

        // Same seed should produce same result
        assert_eq!(hm1.data, hm2.data);

        // Different seed should produce different result
        let mut hm3 = create_terrain();
        hydraulic_erosion(&mut hm3, &config, 999);
        assert_ne!(hm1.data, hm3.data);
    }

    // ========================================================================
    // Thermal erosion tests
    // ========================================================================

    #[test]
    fn test_thermal_erosion_config_default() {
        let config = ThermalErosionConfig::default();
        assert_eq!(config.iterations, 50);
        assert!((config.talus_angle - 0.8).abs() < 0.001);
        assert!((config.transfer_rate - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_thermal_erosion_smooths_steep_slopes() {
        // Create terrain with a sharp spike
        let mut hm = Heightmap::new(16, 16);
        hm.set(8, 8, 10.0); // Sharp spike

        // Calculate initial max slope
        let initial_spike = hm.get(8, 8);

        let config = ThermalErosionConfig {
            iterations: 100,
            talus_angle: 0.5,
            transfer_rate: 0.5,
        };

        thermal_erosion(&mut hm, &config);

        // Spike should be reduced
        assert!(hm.get(8, 8) < initial_spike);

        // Material should have spread to neighbors
        assert!(hm.get(7, 8) > 0.0);
        assert!(hm.get(9, 8) > 0.0);
        assert!(hm.get(8, 7) > 0.0);
        assert!(hm.get(8, 9) > 0.0);
    }

    #[test]
    fn test_thermal_erosion_preserves_mass() {
        let mut hm = Heightmap::new(16, 16);
        // Create terrain where mass is away from edges
        for y in 4..12 {
            for x in 4..12 {
                hm.set(x, y, 5.0 + ((x + y) as f32 * 0.1));
            }
        }

        let original_sum: f32 = hm.data.iter().sum();

        let config = ThermalErosionConfig::default();
        thermal_erosion(&mut hm, &config);

        let new_sum: f32 = hm.data.iter().sum();

        // Mass should be conserved (within floating point tolerance)
        assert!((new_sum - original_sum).abs() < 0.001);
    }

    #[test]
    fn test_thermal_erosion_flat_terrain_unchanged() {
        let mut hm = Heightmap::new(16, 16);
        // Flat terrain
        for y in 0..16 {
            for x in 0..16 {
                hm.set(x, y, 5.0);
            }
        }

        let original_data = hm.data.clone();

        let config = ThermalErosionConfig::default();
        thermal_erosion(&mut hm, &config);

        // Flat terrain should remain unchanged
        assert_eq!(hm.data, original_data);
    }

    #[test]
    fn test_combined_erosion() {
        // Create initial terrain from noise
        let terrain = Terrain2D::new().with_seed(42);
        let mut hm = Heightmap::from_field(&terrain, 32, 32, 0.1);
        hm.normalize();

        // Scale up for erosion
        for h in &mut hm.data {
            *h *= 10.0;
        }

        // Apply both erosion types
        let hydro_config = HydraulicErosionConfig {
            iterations: 500,
            ..Default::default()
        };
        hydraulic_erosion(&mut hm, &hydro_config, 12345);

        let thermal_config = ThermalErosionConfig::default();
        thermal_erosion(&mut hm, &thermal_config);

        // Result should be valid terrain
        assert!(hm.data.iter().all(|&h| h.is_finite()));
        let (min, max) = hm.bounds();
        assert!(min < max);
    }
}
