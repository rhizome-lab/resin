//! glTF 2.0 export for resin meshes.
//!
//! Provides export functionality to glTF format (`.gltf` with embedded data or `.glb` binary).
//!
//! # Example
//!
//! ```ignore
//! use resin_mesh::{Mesh, uv_sphere};
//! use resin_gltf::{GltfExporter, ExportFormat};
//!
//! let mesh = uv_sphere(32, 16);
//! let exporter = GltfExporter::new().with_mesh("sphere", mesh);
//! exporter.export_glb("sphere.glb").unwrap();
//! ```

mod export;

pub use export::{ExportFormat, GltfExporter, GltfMaterial, GltfMesh, GltfResult};
