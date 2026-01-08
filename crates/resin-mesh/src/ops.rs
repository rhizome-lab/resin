//! Mesh operations for procedural geometry.
//!
//! Provides operations like extrude, inset, solidify for mesh manipulation.

use std::collections::HashMap;

use glam::Vec3;

use crate::Mesh;

/// Configuration for extrusion operations.
#[derive(Debug, Clone)]
pub struct ExtrudeConfig {
    /// Distance to extrude (positive = outward, negative = inward).
    pub amount: f32,
    /// Whether to create side faces connecting old and new vertices.
    pub create_sides: bool,
    /// Whether to keep the original faces.
    pub keep_original: bool,
}

impl Default for ExtrudeConfig {
    fn default() -> Self {
        Self {
            amount: 1.0,
            create_sides: true,
            keep_original: false,
        }
    }
}

/// Extrudes the entire mesh along vertex normals.
///
/// Creates a shell by duplicating all vertices and offsetting them along
/// their normals, then connecting old and new vertices with side faces.
pub fn extrude(mesh: &Mesh, amount: f32) -> Mesh {
    extrude_with_config(
        mesh,
        ExtrudeConfig {
            amount,
            ..Default::default()
        },
    )
}

/// Extrudes with full configuration options.
pub fn extrude_with_config(mesh: &Mesh, config: ExtrudeConfig) -> Mesh {
    let vertex_count = mesh.positions.len();
    let mut result = Mesh::with_capacity(vertex_count * 2, mesh.triangle_count() * 3);

    // Compute normals if not present
    let normals: Vec<Vec3> = if mesh.normals.len() == vertex_count {
        mesh.normals.clone()
    } else {
        compute_vertex_normals(mesh)
    };

    // Copy original vertices
    result.positions.extend_from_slice(&mesh.positions);
    result.normals.extend_from_slice(&normals);

    // Create extruded vertices
    for i in 0..vertex_count {
        let extruded_pos = mesh.positions[i] + normals[i] * config.amount;
        result.positions.push(extruded_pos);
        result.normals.push(normals[i]);
    }

    // Copy UVs for both original and extruded vertices
    if !mesh.uvs.is_empty() {
        result.uvs.extend_from_slice(&mesh.uvs);
        result.uvs.extend_from_slice(&mesh.uvs);
    }

    // Add original faces (with reversed winding if not keeping original)
    if config.keep_original {
        // Keep original faces pointing inward
        for tri in mesh.indices.chunks(3) {
            result.indices.push(tri[2]);
            result.indices.push(tri[1]);
            result.indices.push(tri[0]);
        }
    }

    // Add extruded faces (with same winding as original, offset by vertex_count)
    for tri in mesh.indices.chunks(3) {
        let offset = vertex_count as u32;
        result.indices.push(tri[0] + offset);
        result.indices.push(tri[1] + offset);
        result.indices.push(tri[2] + offset);
    }

    // Create side faces if requested
    if config.create_sides {
        // Build edge map: edge -> [triangle indices using it]
        let edges = collect_boundary_edges(mesh);

        for (i0, i1) in edges {
            let offset = vertex_count as u32;

            // Create quad connecting original and extruded edges
            // i0 -> i1 (original edge)
            // i0+offset -> i1+offset (extruded edge)
            // Quad: (i0, i1, i1+offset, i0+offset)

            // First triangle
            result.indices.push(i0);
            result.indices.push(i1);
            result.indices.push(i1 + offset);

            // Second triangle
            result.indices.push(i0);
            result.indices.push(i1 + offset);
            result.indices.push(i0 + offset);
        }
    }

    result
}

/// Solidifies a mesh by adding thickness.
///
/// Creates a solid shell from a surface mesh by extruding inward and
/// creating side walls at boundaries.
pub fn solidify(mesh: &Mesh, thickness: f32) -> Mesh {
    extrude_with_config(
        mesh,
        ExtrudeConfig {
            amount: -thickness,
            create_sides: true,
            keep_original: true,
        },
    )
}

/// Configuration for inset operation.
#[derive(Debug, Clone)]
pub struct InsetConfig {
    /// Inset amount (0.0 = no change, 1.0 = shrink to center).
    pub amount: f32,
    /// Optional depth (positive = extrude inward after inset).
    pub depth: f32,
    /// Whether to create the connecting faces.
    pub create_bridge: bool,
}

impl Default for InsetConfig {
    fn default() -> Self {
        Self {
            amount: 0.2,
            depth: 0.0,
            create_bridge: true,
        }
    }
}

/// Insets all faces toward their centers.
///
/// Creates a new inner face for each triangle, connected by bridge faces.
pub fn inset(mesh: &Mesh, amount: f32) -> Mesh {
    inset_with_config(
        mesh,
        InsetConfig {
            amount,
            ..Default::default()
        },
    )
}

/// Insets faces with full configuration.
pub fn inset_with_config(mesh: &Mesh, config: InsetConfig) -> Mesh {
    let triangle_count = mesh.triangle_count();
    // Each triangle becomes: 1 center triangle + 3 bridge quads (6 triangles) = 7 triangles
    let estimated_triangles = triangle_count * 7;
    let estimated_vertices = triangle_count * 6; // 3 original + 3 inset per face

    let mut result = Mesh::with_capacity(estimated_vertices, estimated_triangles);

    // Compute face normals for depth extrusion
    let face_normals = compute_face_normals(mesh);

    for (face_idx, tri) in mesh.indices.chunks(3).enumerate() {
        let [i0, i1, i2] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];

        let p0 = mesh.positions[i0];
        let p1 = mesh.positions[i1];
        let p2 = mesh.positions[i2];

        // Face center
        let center = (p0 + p1 + p2) / 3.0;

        // Inset positions (lerp toward center)
        let depth_offset = face_normals[face_idx] * -config.depth;
        let ip0 = p0.lerp(center, config.amount) + depth_offset;
        let ip1 = p1.lerp(center, config.amount) + depth_offset;
        let ip2 = p2.lerp(center, config.amount) + depth_offset;

        // Get normals
        let normal = face_normals[face_idx];

        // Add original vertices
        let base = result.positions.len() as u32;
        result.positions.push(p0);
        result.positions.push(p1);
        result.positions.push(p2);
        result.normals.push(normal);
        result.normals.push(normal);
        result.normals.push(normal);

        // Add inset vertices
        result.positions.push(ip0);
        result.positions.push(ip1);
        result.positions.push(ip2);
        result.normals.push(normal);
        result.normals.push(normal);
        result.normals.push(normal);

        // Indices: base+0,1,2 = original, base+3,4,5 = inset

        // Inner triangle (inset face)
        result.indices.push(base + 3);
        result.indices.push(base + 4);
        result.indices.push(base + 5);

        if config.create_bridge {
            // Bridge quads connecting original to inset edges
            // Edge 0-1: (0, 1, 4, 3)
            result.indices.push(base);
            result.indices.push(base + 1);
            result.indices.push(base + 4);

            result.indices.push(base);
            result.indices.push(base + 4);
            result.indices.push(base + 3);

            // Edge 1-2: (1, 2, 5, 4)
            result.indices.push(base + 1);
            result.indices.push(base + 2);
            result.indices.push(base + 5);

            result.indices.push(base + 1);
            result.indices.push(base + 5);
            result.indices.push(base + 4);

            // Edge 2-0: (2, 0, 3, 5)
            result.indices.push(base + 2);
            result.indices.push(base);
            result.indices.push(base + 3);

            result.indices.push(base + 2);
            result.indices.push(base + 3);
            result.indices.push(base + 5);
        }
    }

    // Copy UVs if present (simple copy for now, could interpolate)
    if mesh.has_uvs() {
        for tri in mesh.indices.chunks(3) {
            let [i0, i1, i2] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
            // Original vertices
            result.uvs.push(mesh.uvs[i0]);
            result.uvs.push(mesh.uvs[i1]);
            result.uvs.push(mesh.uvs[i2]);
            // Inset vertices (lerp toward center)
            let uv_center = (mesh.uvs[i0] + mesh.uvs[i1] + mesh.uvs[i2]) / 3.0;
            result.uvs.push(mesh.uvs[i0].lerp(uv_center, config.amount));
            result.uvs.push(mesh.uvs[i1].lerp(uv_center, config.amount));
            result.uvs.push(mesh.uvs[i2].lerp(uv_center, config.amount));
        }
    }

    result
}

/// Flips the winding order of all faces (reverses normals).
pub fn flip_normals(mesh: &Mesh) -> Mesh {
    let mut result = mesh.clone();

    // Reverse each triangle's winding
    for tri in result.indices.chunks_mut(3) {
        tri.swap(0, 2);
    }

    // Flip normals
    for normal in &mut result.normals {
        *normal = -*normal;
    }

    result
}

/// Duplicates the mesh with reversed normals for double-sided rendering.
pub fn make_double_sided(mesh: &Mesh) -> Mesh {
    let mut result = mesh.clone();
    let flipped = flip_normals(mesh);
    result.merge(&flipped);
    result
}

/// Welds vertices that are within the given distance.
///
/// Reduces vertex count by merging coincident vertices.
pub fn weld_vertices(mesh: &Mesh, tolerance: f32) -> Mesh {
    let tol_sq = tolerance * tolerance;

    // Map old vertex indices to new (welded) indices
    let mut vertex_map: Vec<usize> = Vec::with_capacity(mesh.positions.len());
    let mut new_positions: Vec<Vec3> = Vec::new();
    let mut new_normals: Vec<Vec3> = Vec::new();
    let mut new_uvs: Vec<glam::Vec2> = Vec::new();

    // Accumulated normals for averaging
    let mut normal_accum: Vec<Vec3> = Vec::new();
    let mut normal_counts: Vec<usize> = Vec::new();

    for (i, pos) in mesh.positions.iter().enumerate() {
        // Find existing vertex within tolerance
        let mut found = None;
        for (j, existing) in new_positions.iter().enumerate() {
            if pos.distance_squared(*existing) <= tol_sq {
                found = Some(j);
                break;
            }
        }

        if let Some(j) = found {
            vertex_map.push(j);
            // Accumulate normal for averaging
            if i < mesh.normals.len() {
                normal_accum[j] += mesh.normals[i];
                normal_counts[j] += 1;
            }
        } else {
            vertex_map.push(new_positions.len());
            new_positions.push(*pos);

            if i < mesh.normals.len() {
                normal_accum.push(mesh.normals[i]);
                new_normals.push(mesh.normals[i]);
                normal_counts.push(1);
            }

            if i < mesh.uvs.len() {
                new_uvs.push(mesh.uvs[i]);
            }
        }
    }

    // Average accumulated normals
    for (i, normal) in new_normals.iter_mut().enumerate() {
        if normal_counts[i] > 1 {
            *normal = (normal_accum[i] / normal_counts[i] as f32).normalize_or_zero();
        }
    }

    // Remap indices
    let new_indices: Vec<u32> = mesh
        .indices
        .iter()
        .map(|&i| vertex_map[i as usize] as u32)
        .collect();

    Mesh {
        positions: new_positions,
        normals: new_normals,
        uvs: new_uvs,
        indices: new_indices,
    }
}

/// Splits all faces so each face has its own vertices (no sharing).
///
/// Useful for flat shading or per-face attributes.
pub fn split_faces(mesh: &Mesh) -> Mesh {
    let triangle_count = mesh.triangle_count();
    let mut result = Mesh::with_capacity(triangle_count * 3, triangle_count);

    let face_normals = compute_face_normals(mesh);

    for (face_idx, tri) in mesh.indices.chunks(3).enumerate() {
        let base = result.positions.len() as u32;

        for &idx in tri {
            let i = idx as usize;
            result.positions.push(mesh.positions[i]);

            if !mesh.uvs.is_empty() {
                result.uvs.push(mesh.uvs[i]);
            }
        }

        // Use face normal for all three vertices
        let normal = face_normals[face_idx];
        result.normals.push(normal);
        result.normals.push(normal);
        result.normals.push(normal);

        result.indices.push(base);
        result.indices.push(base + 1);
        result.indices.push(base + 2);
    }

    result
}

/// Recalculates normals using the specified mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalMode {
    /// Flat shading - each face has a uniform normal.
    Flat,
    /// Smooth shading - normals are averaged at shared vertices.
    Smooth,
    /// Area-weighted smooth shading.
    AreaWeighted,
}

/// Recalculates mesh normals.
pub fn recalculate_normals(mesh: &Mesh, mode: NormalMode) -> Mesh {
    let mut result = mesh.clone();

    match mode {
        NormalMode::Flat => {
            // Split faces first for proper flat normals
            result = split_faces(&result);
        }
        NormalMode::Smooth | NormalMode::AreaWeighted => {
            result.compute_smooth_normals();
        }
    }

    result
}

// ============================================================================
// Helper functions
// ============================================================================

/// Computes smooth vertex normals from face normals.
fn compute_vertex_normals(mesh: &Mesh) -> Vec<Vec3> {
    let mut normals = vec![Vec3::ZERO; mesh.positions.len()];

    for tri in mesh.indices.chunks(3) {
        let [i0, i1, i2] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
        let v0 = mesh.positions[i0];
        let v1 = mesh.positions[i1];
        let v2 = mesh.positions[i2];

        let normal = (v1 - v0).cross(v2 - v0); // unnormalized = area-weighted
        normals[i0] += normal;
        normals[i1] += normal;
        normals[i2] += normal;
    }

    for normal in &mut normals {
        *normal = normal.normalize_or_zero();
    }

    normals
}

/// Computes face normals (one per triangle).
fn compute_face_normals(mesh: &Mesh) -> Vec<Vec3> {
    mesh.indices
        .chunks(3)
        .map(|tri| {
            let [i0, i1, i2] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
            let v0 = mesh.positions[i0];
            let v1 = mesh.positions[i1];
            let v2 = mesh.positions[i2];
            (v1 - v0).cross(v2 - v0).normalize_or_zero()
        })
        .collect()
}

/// Collects all edges from the mesh.
///
/// Returns edges as (lower_index, higher_index) pairs for deduplication.
fn collect_boundary_edges(mesh: &Mesh) -> Vec<(u32, u32)> {
    // Count how many times each edge appears
    let mut edge_count: HashMap<(u32, u32), usize> = HashMap::new();

    for tri in mesh.indices.chunks(3) {
        let edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])];

        for (a, b) in edges {
            // Normalize edge direction for counting
            let key = if a < b { (a, b) } else { (b, a) };
            *edge_count.entry(key).or_insert(0) += 1;
        }
    }

    // Return all edges (for extrusion we want all edges, not just boundary)
    // Keep original winding order for correct face generation
    let mut result = Vec::new();
    for tri in mesh.indices.chunks(3) {
        result.push((tri[0], tri[1]));
        result.push((tri[1], tri[2]));
        result.push((tri[2], tri[0]));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::box_mesh;

    #[test]
    fn test_extrude_basic() {
        let cube = box_mesh();
        let extruded = extrude(&cube, 0.5);

        // Should have more vertices and triangles
        assert!(extruded.vertex_count() > cube.vertex_count());
        assert!(extruded.triangle_count() > cube.triangle_count());
    }

    #[test]
    fn test_solidify() {
        let cube = box_mesh();
        let solid = solidify(&cube, 0.1);

        // Solidify creates inner and outer shell plus sides
        assert!(solid.vertex_count() > cube.vertex_count());
    }

    #[test]
    fn test_inset() {
        let cube = box_mesh();
        let inseted = inset(&cube, 0.3);

        // Inset creates many more triangles
        assert!(inseted.triangle_count() > cube.triangle_count());
    }

    #[test]
    fn test_flip_normals() {
        let cube = box_mesh();
        let flipped = flip_normals(&cube);

        // Same vertex/triangle count
        assert_eq!(flipped.vertex_count(), cube.vertex_count());
        assert_eq!(flipped.triangle_count(), cube.triangle_count());

        // Normals should be reversed
        for (orig, flip) in cube.normals.iter().zip(flipped.normals.iter()) {
            assert!((orig + flip).length() < 0.001);
        }
    }

    #[test]
    fn test_double_sided() {
        let cube = box_mesh();
        let doubled = make_double_sided(&cube);

        // Should have exactly double the geometry
        assert_eq!(doubled.vertex_count(), cube.vertex_count() * 2);
        assert_eq!(doubled.triangle_count(), cube.triangle_count() * 2);
    }

    #[test]
    fn test_weld_vertices() {
        // Create a mesh with duplicate vertices
        let mut mesh = Mesh::new();
        mesh.positions.push(Vec3::new(0.0, 0.0, 0.0));
        mesh.positions.push(Vec3::new(1.0, 0.0, 0.0));
        mesh.positions.push(Vec3::new(0.5, 1.0, 0.0));
        mesh.positions.push(Vec3::new(0.0001, 0.0, 0.0)); // Near duplicate of first
        mesh.positions.push(Vec3::new(1.0001, 0.0, 0.0)); // Near duplicate of second
        mesh.positions.push(Vec3::new(0.5, 1.0, 0.5));

        mesh.indices = vec![0, 1, 2, 3, 4, 5];

        let welded = weld_vertices(&mesh, 0.01);

        // Should have fewer vertices due to merging
        assert!(welded.vertex_count() < mesh.vertex_count());
    }

    #[test]
    fn test_split_faces() {
        let cube = box_mesh();
        let split = split_faces(&cube);

        // Each triangle gets 3 unique vertices
        assert_eq!(split.vertex_count(), cube.triangle_count() * 3);
    }

    #[test]
    fn test_inset_with_depth() {
        let cube = box_mesh();
        let inseted = inset_with_config(
            &cube,
            InsetConfig {
                amount: 0.3,
                depth: 0.2,
                create_bridge: true,
            },
        );

        // Should have the expected structure
        assert!(inseted.triangle_count() > cube.triangle_count());
        assert!(inseted.has_normals());
    }
}
