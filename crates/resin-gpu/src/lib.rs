//! GPU compute backend for resin fields.
//!
//! Provides GPU-accelerated evaluation of field functions using wgpu compute shaders.
//!
//! # Features
//!
//! - Batch field sampling: evaluate fields at thousands of points in parallel
//! - Texture generation: render fields to 2D textures
//! - Built-in noise shaders: Perlin, Simplex, FBM
//!
//! # Example
//!
//! ```ignore
//! use resin_gpu::{GpuContext, NoiseType, generate_noise_texture};
//!
//! let ctx = GpuContext::new()?;
//! let texture = generate_noise_texture(&ctx, 512, 512, NoiseType::Perlin, 4.0)?;
//! ```

mod context;
mod error;
mod noise;
mod texture;

pub use context::GpuContext;
pub use error::GpuError;
pub use noise::{NoiseConfig, NoiseType, generate_noise_texture, noise_texture};
pub use texture::{GpuTexture, TextureFormat};
