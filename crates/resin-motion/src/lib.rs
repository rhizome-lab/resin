//! 2D motion graphics scene graph.
//!
//! Provides a hierarchical scene structure for After Effects-style motion graphics:
//! - [`Transform2D`] - 2D transform with anchor point (pivot for rotation/scale)
//! - [`Layer`] - Scene node with transform, opacity, blend mode, and children
//! - [`Scene`] - Root container for layers
//!
//! # Example
//!
//! ```
//! use rhizome_resin_motion::{Scene, Layer, Transform2D};
//! use glam::Vec2;
//!
//! // Create a scene with nested layers
//! let mut scene = Scene::new();
//!
//! let mut parent = Layer::new("group");
//! parent.transform.position = Vec2::new(100.0, 100.0);
//!
//! let mut child = Layer::new("shape");
//! child.transform.position = Vec2::new(50.0, 0.0);
//! child.transform.rotation = 45.0_f32.to_radians();
//!
//! parent.add_child(child);
//! scene.add_layer(parent);
//!
//! // Resolve world transforms
//! let world_transforms = scene.resolve_transforms();
//! ```

use glam::{Mat3, Vec2};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub use rhizome_resin_color::BlendMode;

// ============================================================================
// Transform2D
// ============================================================================

/// A 2D transform with anchor point (pivot for rotation/scale).
///
/// The transform is applied in this order:
/// 1. Translate by -anchor (move pivot to origin)
/// 2. Scale
/// 3. Rotate
/// 4. Translate by +anchor (restore pivot position)
/// 5. Translate by position
///
/// This matches After Effects' transform behavior where the anchor point
/// defines the center of rotation and scaling.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Transform2D {
    /// Position in parent space.
    pub position: Vec2,

    /// Anchor point (pivot) in local space.
    /// Rotation and scale happen around this point.
    pub anchor: Vec2,

    /// Rotation in radians.
    pub rotation: f32,

    /// Scale factors (1.0 = 100%).
    pub scale: Vec2,

    /// Skew angle in radians.
    pub skew: f32,

    /// Skew axis angle in radians.
    pub skew_axis: f32,
}

impl Default for Transform2D {
    fn default() -> Self {
        Self {
            position: Vec2::ZERO,
            anchor: Vec2::ZERO,
            rotation: 0.0,
            scale: Vec2::ONE,
            skew: 0.0,
            skew_axis: 0.0,
        }
    }
}

impl Transform2D {
    /// Creates a new identity transform.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a transform with just position.
    pub fn from_position(position: Vec2) -> Self {
        Self {
            position,
            ..Default::default()
        }
    }

    /// Creates a transform with position and rotation.
    pub fn from_position_rotation(position: Vec2, rotation: f32) -> Self {
        Self {
            position,
            rotation,
            ..Default::default()
        }
    }

    /// Builder: set position.
    pub fn with_position(mut self, position: Vec2) -> Self {
        self.position = position;
        self
    }

    /// Builder: set anchor point.
    pub fn with_anchor(mut self, anchor: Vec2) -> Self {
        self.anchor = anchor;
        self
    }

    /// Builder: set rotation (radians).
    pub fn with_rotation(mut self, rotation: f32) -> Self {
        self.rotation = rotation;
        self
    }

    /// Builder: set rotation (degrees).
    pub fn with_rotation_degrees(mut self, degrees: f32) -> Self {
        self.rotation = degrees.to_radians();
        self
    }

    /// Builder: set uniform scale.
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = Vec2::splat(scale);
        self
    }

    /// Builder: set non-uniform scale.
    pub fn with_scale_xy(mut self, scale: Vec2) -> Self {
        self.scale = scale;
        self
    }

    /// Builder: set skew.
    pub fn with_skew(mut self, skew: f32, axis: f32) -> Self {
        self.skew = skew;
        self.skew_axis = axis;
        self
    }

    /// Computes the 3x3 transformation matrix.
    ///
    /// The matrix transforms points from local space to parent space.
    pub fn to_matrix(&self) -> Mat3 {
        // Build transform: translate(-anchor) -> scale -> skew -> rotate -> translate(anchor) -> translate(position)
        let cos_r = self.rotation.cos();
        let sin_r = self.rotation.sin();

        // Skew matrix components
        let cos_axis = self.skew_axis.cos();
        let sin_axis = self.skew_axis.sin();
        let tan_skew = self.skew.tan();

        // Scale matrix
        let sx = self.scale.x;
        let sy = self.scale.y;

        // Combined rotation + skew + scale
        // Skew is applied along skew_axis direction
        let skew_x = tan_skew * cos_axis * cos_axis;
        let skew_y = tan_skew * sin_axis * cos_axis;

        // Build the 2x2 part: R * Skew * S
        // Simplified: we apply scale, then skew, then rotation
        let m00 = (cos_r * sx) + (sin_r * skew_y * sx);
        let m01 = (-sin_r * sy) + (cos_r * skew_x * sy);
        let m10 = (sin_r * sx) + (-cos_r * skew_y * sx);
        let m11 = (cos_r * sy) + (sin_r * skew_x * sy);

        // Translation: position + rotation_around_anchor
        // Point transform: p' = R * S * (p - anchor) + anchor + position
        //                    = R * S * p - R * S * anchor + anchor + position
        let anchor_offset = Vec2::new(
            m00 * (-self.anchor.x) + m01 * (-self.anchor.y),
            m10 * (-self.anchor.x) + m11 * (-self.anchor.y),
        );
        let tx = self.position.x + self.anchor.x + anchor_offset.x;
        let ty = self.position.y + self.anchor.y + anchor_offset.y;

        Mat3::from_cols(
            glam::Vec3::new(m00, m10, 0.0),
            glam::Vec3::new(m01, m11, 0.0),
            glam::Vec3::new(tx, ty, 1.0),
        )
    }

    /// Transform a point from local space to parent space.
    pub fn transform_point(&self, point: Vec2) -> Vec2 {
        let m = self.to_matrix();
        let p = m * glam::Vec3::new(point.x, point.y, 1.0);
        Vec2::new(p.x, p.y)
    }

    /// Concatenate with another transform (self * other).
    ///
    /// Returns a matrix representing applying `other` first, then `self`.
    pub fn concat(&self, other: &Transform2D) -> Mat3 {
        self.to_matrix() * other.to_matrix()
    }

    /// Interpolate between two transforms.
    pub fn lerp(&self, other: &Transform2D, t: f32) -> Transform2D {
        Transform2D {
            position: self.position.lerp(other.position, t),
            anchor: self.anchor.lerp(other.anchor, t),
            rotation: self.rotation + (other.rotation - self.rotation) * t,
            scale: self.scale.lerp(other.scale, t),
            skew: self.skew + (other.skew - self.skew) * t,
            skew_axis: self.skew_axis + (other.skew_axis - self.skew_axis) * t,
        }
    }
}

// ============================================================================
// Layer
// ============================================================================

/// A layer in the scene graph.
///
/// Layers can contain:
/// - Transform (position, rotation, scale with anchor point)
/// - Visual properties (opacity, blend mode)
/// - Children (nested layers)
/// - Content (path, image, etc.) - stored as generic content ID
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Layer {
    /// Unique name/identifier for this layer.
    pub name: String,

    /// Transform relative to parent.
    pub transform: Transform2D,

    /// Opacity (0.0 = transparent, 1.0 = opaque).
    pub opacity: f32,

    /// Blend mode for compositing.
    pub blend_mode: BlendMode,

    /// Whether this layer is visible.
    pub visible: bool,

    /// Whether this layer is locked (for UI purposes).
    pub locked: bool,

    /// Child layers.
    pub children: Vec<Layer>,

    /// Optional content identifier (references external content like paths, images).
    /// The actual content is stored separately; this is just a reference.
    pub content_id: Option<String>,
}

impl Layer {
    /// Creates a new empty layer.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            transform: Transform2D::default(),
            opacity: 1.0,
            blend_mode: BlendMode::Normal,
            visible: true,
            locked: false,
            children: Vec::new(),
            content_id: None,
        }
    }

    /// Creates a layer with content.
    pub fn with_content(name: impl Into<String>, content_id: impl Into<String>) -> Self {
        Self {
            content_id: Some(content_id.into()),
            ..Self::new(name)
        }
    }

    /// Builder: set transform.
    pub fn with_transform(mut self, transform: Transform2D) -> Self {
        self.transform = transform;
        self
    }

    /// Builder: set opacity.
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity;
        self
    }

    /// Builder: set blend mode.
    pub fn with_blend_mode(mut self, blend_mode: BlendMode) -> Self {
        self.blend_mode = blend_mode;
        self
    }

    /// Builder: set visibility.
    pub fn with_visible(mut self, visible: bool) -> Self {
        self.visible = visible;
        self
    }

    /// Add a child layer.
    pub fn add_child(&mut self, child: Layer) {
        self.children.push(child);
    }

    /// Get child by name.
    pub fn get_child(&self, name: &str) -> Option<&Layer> {
        self.children.iter().find(|c| c.name == name)
    }

    /// Get mutable child by name.
    pub fn get_child_mut(&mut self, name: &str) -> Option<&mut Layer> {
        self.children.iter_mut().find(|c| c.name == name)
    }

    /// Find a layer by path (e.g., "parent/child/grandchild").
    pub fn find_by_path(&self, path: &str) -> Option<&Layer> {
        let mut parts = path.split('/');
        let first = parts.next()?;

        if first == self.name {
            let rest: Vec<&str> = parts.collect();
            if rest.is_empty() {
                return Some(self);
            }
            let child_path = rest.join("/");
            for child in &self.children {
                if let Some(found) = child.find_by_path(&child_path) {
                    return Some(found);
                }
            }
            None
        } else {
            None
        }
    }

    /// Iterate over all layers (depth-first).
    pub fn iter(&self) -> LayerIter<'_> {
        LayerIter { stack: vec![self] }
    }

    /// Count total layers including self and all descendants.
    pub fn count(&self) -> usize {
        1 + self.children.iter().map(|c| c.count()).sum::<usize>()
    }
}

/// Iterator over layers in depth-first order.
pub struct LayerIter<'a> {
    stack: Vec<&'a Layer>,
}

impl<'a> Iterator for LayerIter<'a> {
    type Item = &'a Layer;

    fn next(&mut self) -> Option<Self::Item> {
        let layer = self.stack.pop()?;
        // Push children in reverse order so first child is processed first
        for child in layer.children.iter().rev() {
            self.stack.push(child);
        }
        Some(layer)
    }
}

// ============================================================================
// ResolvedLayer
// ============================================================================

/// A layer with its world transform resolved.
#[derive(Debug, Clone)]
pub struct ResolvedLayer<'a> {
    /// Reference to the original layer.
    pub layer: &'a Layer,

    /// World transform matrix (from local to scene root).
    pub world_matrix: Mat3,

    /// Accumulated opacity (parent opacity * layer opacity).
    pub world_opacity: f32,

    /// Depth in the hierarchy (0 = root).
    pub depth: usize,
}

impl<'a> ResolvedLayer<'a> {
    /// Transform a point from this layer's local space to world space.
    pub fn transform_point(&self, point: Vec2) -> Vec2 {
        let p = self.world_matrix * glam::Vec3::new(point.x, point.y, 1.0);
        Vec2::new(p.x, p.y)
    }
}

// ============================================================================
// Scene
// ============================================================================

/// A scene containing layers.
///
/// The scene is the root container for all layers. It provides methods
/// for traversal, transform resolution, and layer lookup.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Scene {
    /// Root-level layers.
    pub layers: Vec<Layer>,

    /// Scene dimensions (for reference; doesn't clip).
    pub width: f32,
    pub height: f32,

    /// Frame rate for animation.
    pub frame_rate: f32,

    /// Duration in seconds.
    pub duration: f32,
}

impl Scene {
    /// Creates a new empty scene.
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            width: 1920.0,
            height: 1080.0,
            frame_rate: 30.0,
            duration: 10.0,
        }
    }

    /// Creates a scene with specified dimensions.
    pub fn with_size(width: f32, height: f32) -> Self {
        Self {
            width,
            height,
            ..Self::new()
        }
    }

    /// Add a layer to the scene root.
    pub fn add_layer(&mut self, layer: Layer) {
        self.layers.push(layer);
    }

    /// Get layer by name (searches root level only).
    pub fn get_layer(&self, name: &str) -> Option<&Layer> {
        self.layers.iter().find(|l| l.name == name)
    }

    /// Get mutable layer by name (searches root level only).
    pub fn get_layer_mut(&mut self, name: &str) -> Option<&mut Layer> {
        self.layers.iter_mut().find(|l| l.name == name)
    }

    /// Find a layer by path (e.g., "parent/child/grandchild").
    pub fn find_by_path(&self, path: &str) -> Option<&Layer> {
        let first_part = path.split('/').next()?;
        for layer in &self.layers {
            if layer.name == first_part {
                return layer.find_by_path(path);
            }
        }
        None
    }

    /// Resolve world transforms for all visible layers.
    ///
    /// Returns a flat list of layers with their world transforms computed.
    /// Layers are returned in depth-first order (parent before children).
    pub fn resolve_transforms(&self) -> Vec<ResolvedLayer<'_>> {
        let mut result = Vec::new();
        for layer in &self.layers {
            if layer.visible {
                self.resolve_layer(layer, Mat3::IDENTITY, 1.0, 0, &mut result);
            }
        }
        result
    }

    fn resolve_layer<'a>(
        &self,
        layer: &'a Layer,
        parent_matrix: Mat3,
        parent_opacity: f32,
        depth: usize,
        result: &mut Vec<ResolvedLayer<'a>>,
    ) {
        let world_matrix = parent_matrix * layer.transform.to_matrix();
        let world_opacity = parent_opacity * layer.opacity;

        result.push(ResolvedLayer {
            layer,
            world_matrix,
            world_opacity,
            depth,
        });

        for child in &layer.children {
            if child.visible {
                self.resolve_layer(child, world_matrix, world_opacity, depth + 1, result);
            }
        }
    }

    /// Count total layers in the scene.
    pub fn layer_count(&self) -> usize {
        self.layers.iter().map(|l| l.count()).sum()
    }

    /// Iterate over all layers in depth-first order.
    pub fn iter_layers(&self) -> impl Iterator<Item = &Layer> {
        self.layers.iter().flat_map(|l| l.iter())
    }

    /// Get frame number from time.
    pub fn time_to_frame(&self, time: f32) -> f32 {
        time * self.frame_rate
    }

    /// Get time from frame number.
    pub fn frame_to_time(&self, frame: f32) -> f32 {
        frame / self.frame_rate
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform2d_identity() {
        let t = Transform2D::new();
        let point = Vec2::new(10.0, 20.0);
        let result = t.transform_point(point);
        assert!((result - point).length() < 0.001);
    }

    #[test]
    fn test_transform2d_position() {
        let t = Transform2D::from_position(Vec2::new(100.0, 50.0));
        let result = t.transform_point(Vec2::ZERO);
        assert!((result - Vec2::new(100.0, 50.0)).length() < 0.001);
    }

    #[test]
    fn test_transform2d_rotation() {
        let t = Transform2D::new().with_rotation(std::f32::consts::FRAC_PI_2); // 90 degrees
        let point = Vec2::new(1.0, 0.0);
        let result = t.transform_point(point);
        // (1, 0) rotated 90 degrees CCW = (0, 1)
        assert!((result - Vec2::new(0.0, 1.0)).length() < 0.001);
    }

    #[test]
    fn test_transform2d_rotation_with_anchor() {
        // Rotate around point (1, 0) instead of origin
        let t = Transform2D::new()
            .with_anchor(Vec2::new(1.0, 0.0))
            .with_rotation(std::f32::consts::FRAC_PI_2);

        // Point at origin, anchor at (1, 0)
        // After rotation: origin moves to (1, -1) relative to anchor, then anchor restored
        let result = t.transform_point(Vec2::ZERO);
        assert!(
            (result - Vec2::new(1.0, -1.0)).length() < 0.001,
            "Expected (1, -1), got {:?}",
            result
        );
    }

    #[test]
    fn test_transform2d_scale() {
        let t = Transform2D::new().with_scale_xy(Vec2::new(2.0, 3.0));
        let result = t.transform_point(Vec2::new(10.0, 10.0));
        assert!((result - Vec2::new(20.0, 30.0)).length() < 0.001);
    }

    #[test]
    fn test_transform2d_scale_with_anchor() {
        // Scale 2x around point (10, 10)
        let t = Transform2D::new()
            .with_anchor(Vec2::new(10.0, 10.0))
            .with_scale(2.0);

        // Point at (20, 10) - 10 units right of anchor
        // After 2x scale: 20 units right of anchor = (30, 10)
        let result = t.transform_point(Vec2::new(20.0, 10.0));
        assert!(
            (result - Vec2::new(30.0, 10.0)).length() < 0.001,
            "Expected (30, 10), got {:?}",
            result
        );
    }

    #[test]
    fn test_transform2d_combined() {
        // Position + rotation + scale
        let t = Transform2D::new()
            .with_position(Vec2::new(100.0, 100.0))
            .with_scale(2.0)
            .with_rotation(std::f32::consts::FRAC_PI_2);

        let result = t.transform_point(Vec2::new(1.0, 0.0));
        // (1, 0) * scale 2 = (2, 0)
        // (2, 0) rotated 90 = (0, 2)
        // + position (100, 100) = (100, 102)
        assert!(
            (result - Vec2::new(100.0, 102.0)).length() < 0.001,
            "Expected (100, 102), got {:?}",
            result
        );
    }

    #[test]
    fn test_transform2d_lerp() {
        let t1 = Transform2D::from_position(Vec2::ZERO);
        let t2 = Transform2D::from_position(Vec2::new(100.0, 100.0));
        let mid = t1.lerp(&t2, 0.5);
        assert!((mid.position - Vec2::new(50.0, 50.0)).length() < 0.001);
    }

    #[test]
    fn test_layer_hierarchy() {
        let mut parent = Layer::new("parent");
        parent.add_child(Layer::new("child1"));
        parent.add_child(Layer::new("child2"));

        assert_eq!(parent.children.len(), 2);
        assert!(parent.get_child("child1").is_some());
        assert!(parent.get_child("nonexistent").is_none());
    }

    #[test]
    fn test_layer_count() {
        let mut parent = Layer::new("parent");
        let mut child = Layer::new("child");
        child.add_child(Layer::new("grandchild"));
        parent.add_child(child);

        assert_eq!(parent.count(), 3);
    }

    #[test]
    fn test_layer_iter() {
        let mut parent = Layer::new("parent");
        parent.add_child(Layer::new("child1"));
        parent.add_child(Layer::new("child2"));

        let names: Vec<&str> = parent.iter().map(|l| l.name.as_str()).collect();
        assert_eq!(names, vec!["parent", "child1", "child2"]);
    }

    #[test]
    fn test_scene_add_and_find() {
        let mut scene = Scene::new();
        scene.add_layer(Layer::new("layer1"));

        let mut group = Layer::new("group");
        group.add_child(Layer::new("nested"));
        scene.add_layer(group);

        assert!(scene.get_layer("layer1").is_some());
        assert!(scene.get_layer("group").is_some());
        assert!(scene.find_by_path("group/nested").is_some());
    }

    #[test]
    fn test_scene_resolve_transforms() {
        let mut scene = Scene::new();

        let mut parent = Layer::new("parent");
        parent.transform.position = Vec2::new(100.0, 0.0);

        let mut child = Layer::new("child");
        child.transform.position = Vec2::new(50.0, 0.0);
        child.opacity = 0.5;

        parent.add_child(child);
        scene.add_layer(parent);

        let resolved = scene.resolve_transforms();
        assert_eq!(resolved.len(), 2);

        // Parent at (100, 0)
        let parent_resolved = &resolved[0];
        assert_eq!(parent_resolved.layer.name, "parent");
        let parent_origin = parent_resolved.transform_point(Vec2::ZERO);
        assert!((parent_origin - Vec2::new(100.0, 0.0)).length() < 0.001);

        // Child at parent (100, 0) + local (50, 0) = (150, 0)
        let child_resolved = &resolved[1];
        assert_eq!(child_resolved.layer.name, "child");
        let child_origin = child_resolved.transform_point(Vec2::ZERO);
        assert!(
            (child_origin - Vec2::new(150.0, 0.0)).length() < 0.001,
            "Expected (150, 0), got {:?}",
            child_origin
        );

        // Child world opacity = parent (1.0) * child (0.5) = 0.5
        assert!((child_resolved.world_opacity - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_scene_hidden_layers() {
        let mut scene = Scene::new();
        scene.add_layer(Layer::new("visible"));
        scene.add_layer(Layer::new("hidden").with_visible(false));

        let resolved = scene.resolve_transforms();
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].layer.name, "visible");
    }

    #[test]
    fn test_scene_layer_count() {
        let mut scene = Scene::new();
        let mut group = Layer::new("group");
        group.add_child(Layer::new("child1"));
        group.add_child(Layer::new("child2"));
        scene.add_layer(group);
        scene.add_layer(Layer::new("solo"));

        assert_eq!(scene.layer_count(), 4);
    }

    #[test]
    fn test_scene_time_frame_conversion() {
        let scene = Scene {
            frame_rate: 24.0,
            ..Scene::new()
        };

        assert!((scene.time_to_frame(1.0) - 24.0).abs() < 0.001);
        assert!((scene.frame_to_time(24.0) - 1.0).abs() < 0.001);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_transform2d_serde() {
        let t = Transform2D::new()
            .with_position(Vec2::new(100.0, 50.0))
            .with_rotation(0.5)
            .with_scale(2.0);

        let json = serde_json::to_string(&t).unwrap();
        let parsed: Transform2D = serde_json::from_str(&json).unwrap();

        assert!((parsed.position - t.position).length() < 0.001);
        assert!((parsed.rotation - t.rotation).abs() < 0.001);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_scene_serde() {
        let mut scene = Scene::with_size(1280.0, 720.0);
        let mut layer = Layer::new("test");
        layer.transform.position = Vec2::new(100.0, 100.0);
        scene.add_layer(layer);

        let json = serde_json::to_string(&scene).unwrap();
        let parsed: Scene = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.layers.len(), 1);
        assert_eq!(parsed.layers[0].name, "test");
    }
}
