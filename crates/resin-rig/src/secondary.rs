//! Secondary motion effects for procedural animation.
//!
//! Provides physics-based secondary motion:
//! - Jiggle (spring dynamics for soft body parts like hair, clothing, flesh)
//! - Follow-through (delayed reaction to parent motion)
//! - Overlapping action (different parts move at different rates)
//! - Drag (resistance to motion)
//!
//! # Example
//!
//! ```
//! use rhizome_resin_rig::secondary::{JiggleBone, SecondaryMotion, SecondaryConfig};
//! use glam::Vec3;
//!
//! // Create a jiggle bone for hair
//! let mut jiggle = JiggleBone::new()
//!     .with_stiffness(50.0)
//!     .with_damping(5.0)
//!     .with_gravity(Vec3::new(0.0, -9.81, 0.0));
//!
//! // Update each frame
//! let parent_position = Vec3::ZERO;
//! let parent_rotation = glam::Quat::IDENTITY;
//! jiggle.update(parent_position, parent_rotation, 0.016);
//!
//! // Get the resulting offset
//! let offset = jiggle.offset();
//! ```

use glam::{Quat, Vec3};

/// Configuration for secondary motion effects.
#[derive(Debug, Clone)]
pub struct SecondaryConfig {
    /// Spring stiffness (higher = snappier return to rest).
    pub stiffness: f32,
    /// Damping coefficient (higher = less oscillation).
    pub damping: f32,
    /// Mass of the simulated point.
    pub mass: f32,
    /// Gravity vector.
    pub gravity: Vec3,
    /// Maximum displacement from rest position.
    pub max_displacement: f32,
    /// Whether to enable collision with parent bone.
    pub enable_collision: bool,
}

impl Default for SecondaryConfig {
    fn default() -> Self {
        Self {
            stiffness: 100.0,
            damping: 10.0,
            mass: 1.0,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            max_displacement: 1.0,
            enable_collision: false,
        }
    }
}

impl SecondaryConfig {
    /// Creates a config for soft/bouncy motion (low stiffness, low damping).
    pub fn soft() -> Self {
        Self {
            stiffness: 30.0,
            damping: 3.0,
            mass: 1.0,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            max_displacement: 2.0,
            enable_collision: false,
        }
    }

    /// Creates a config for stiff motion (high stiffness, high damping).
    pub fn stiff() -> Self {
        Self {
            stiffness: 300.0,
            damping: 30.0,
            mass: 1.0,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            max_displacement: 0.3,
            enable_collision: false,
        }
    }

    /// Creates a config for hair-like motion.
    pub fn hair() -> Self {
        Self {
            stiffness: 50.0,
            damping: 5.0,
            mass: 0.5,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            max_displacement: 1.5,
            enable_collision: true,
        }
    }

    /// Creates a config for tail-like motion.
    pub fn tail() -> Self {
        Self {
            stiffness: 80.0,
            damping: 8.0,
            mass: 1.5,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            max_displacement: 2.0,
            enable_collision: false,
        }
    }

    /// Creates a config for cloth-like motion.
    pub fn cloth() -> Self {
        Self {
            stiffness: 40.0,
            damping: 6.0,
            mass: 0.3,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            max_displacement: 3.0,
            enable_collision: true,
        }
    }
}

/// A single jiggle bone with spring physics.
#[derive(Debug, Clone)]
pub struct JiggleBone {
    /// Current position offset from rest.
    position: Vec3,
    /// Current velocity.
    velocity: Vec3,
    /// Rest position in local space.
    rest_position: Vec3,
    /// Previous parent position (for inertia calculation).
    prev_parent_position: Vec3,
    /// Configuration.
    config: SecondaryConfig,
}

impl Default for JiggleBone {
    fn default() -> Self {
        Self::new()
    }
}

impl JiggleBone {
    /// Creates a new jiggle bone.
    pub fn new() -> Self {
        Self {
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            rest_position: Vec3::ZERO,
            prev_parent_position: Vec3::ZERO,
            config: SecondaryConfig::default(),
        }
    }

    /// Creates from configuration.
    pub fn from_config(config: SecondaryConfig) -> Self {
        Self {
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            rest_position: Vec3::ZERO,
            prev_parent_position: Vec3::ZERO,
            config,
        }
    }

    /// Sets the stiffness.
    pub fn with_stiffness(mut self, stiffness: f32) -> Self {
        self.config.stiffness = stiffness;
        self
    }

    /// Sets the damping.
    pub fn with_damping(mut self, damping: f32) -> Self {
        self.config.damping = damping;
        self
    }

    /// Sets the gravity.
    pub fn with_gravity(mut self, gravity: Vec3) -> Self {
        self.config.gravity = gravity;
        self
    }

    /// Sets the rest position in local space.
    pub fn with_rest_position(mut self, rest: Vec3) -> Self {
        self.rest_position = rest;
        self.position = rest;
        self
    }

    /// Sets the maximum displacement.
    pub fn with_max_displacement(mut self, max: f32) -> Self {
        self.config.max_displacement = max;
        self
    }

    /// Updates the jiggle bone simulation.
    ///
    /// # Arguments
    /// * `parent_position` - Parent bone's world position
    /// * `parent_rotation` - Parent bone's world rotation
    /// * `dt` - Time step in seconds
    pub fn update(&mut self, parent_position: Vec3, parent_rotation: Quat, dt: f32) {
        if dt <= 0.0 {
            return;
        }

        // Compute target position in world space
        let target_world = parent_position + parent_rotation * self.rest_position;

        // Inertia: react to parent acceleration
        let parent_velocity = (parent_position - self.prev_parent_position) / dt;
        self.prev_parent_position = parent_position;

        // Current world position
        let current_world = parent_position + parent_rotation * self.position;

        // Spring force toward target
        let displacement = current_world - target_world;
        let spring_force = -self.config.stiffness * displacement;

        // Damping force
        let damping_force = -self.config.damping * self.velocity;

        // Gravity
        let gravity_force = self.config.gravity * self.config.mass;

        // Inertia force (react to parent motion)
        let inertia_force = -parent_velocity * self.config.mass * 0.5;

        // Total acceleration
        let total_force = spring_force + damping_force + gravity_force + inertia_force;
        let acceleration = total_force / self.config.mass;

        // Integrate velocity and position (semi-implicit Euler)
        self.velocity += acceleration * dt;
        let new_world = current_world + self.velocity * dt;

        // Convert back to local space
        let inv_rotation = parent_rotation.inverse();
        self.position = inv_rotation * (new_world - parent_position);

        // Clamp displacement
        let local_displacement = self.position - self.rest_position;
        if local_displacement.length() > self.config.max_displacement {
            self.position =
                self.rest_position + local_displacement.normalize() * self.config.max_displacement;
            // Reduce velocity when hitting limit
            self.velocity *= 0.5;
        }
    }

    /// Returns the current position offset from rest.
    pub fn offset(&self) -> Vec3 {
        self.position - self.rest_position
    }

    /// Returns the current position in local space.
    pub fn position(&self) -> Vec3 {
        self.position
    }

    /// Returns the current velocity.
    pub fn velocity(&self) -> Vec3 {
        self.velocity
    }

    /// Resets to rest state.
    pub fn reset(&mut self) {
        self.position = self.rest_position;
        self.velocity = Vec3::ZERO;
    }
}

/// A chain of jiggle bones for hair, tails, etc.
#[derive(Debug, Clone)]
pub struct JiggleChain {
    /// Bones in the chain.
    bones: Vec<JiggleBone>,
    /// Segment lengths between bones.
    lengths: Vec<f32>,
    /// Root position.
    root_position: Vec3,
    /// Root rotation.
    root_rotation: Quat,
}

impl JiggleChain {
    /// Creates a new jiggle chain.
    pub fn new(bone_count: usize, segment_length: f32, config: SecondaryConfig) -> Self {
        let mut bones = Vec::with_capacity(bone_count);
        let lengths = vec![segment_length; bone_count.saturating_sub(1)];

        for i in 0..bone_count {
            let rest_pos = Vec3::new(0.0, -(i as f32) * segment_length, 0.0);
            bones.push(JiggleBone::from_config(config.clone()).with_rest_position(rest_pos));
        }

        Self {
            bones,
            lengths,
            root_position: Vec3::ZERO,
            root_rotation: Quat::IDENTITY,
        }
    }

    /// Updates the chain simulation.
    pub fn update(&mut self, root_position: Vec3, root_rotation: Quat, dt: f32) {
        self.root_position = root_position;
        self.root_rotation = root_rotation;

        if self.bones.is_empty() {
            return;
        }

        // Update first bone from root
        self.bones[0].update(root_position, root_rotation, dt);

        // Update subsequent bones from previous bone
        for i in 1..self.bones.len() {
            let prev_world = root_position + root_rotation * self.bones[i - 1].position();

            // Compute direction from previous bone
            let curr_world = root_position + root_rotation * self.bones[i].position();
            let dir = (curr_world - prev_world).normalize_or_zero();

            // Create rotation that looks along the chain
            let look_rot = if dir.length_squared() > 0.01 {
                Quat::from_rotation_arc(Vec3::NEG_Y, dir)
            } else {
                Quat::IDENTITY
            };

            self.bones[i].update(prev_world, look_rot * root_rotation, dt);
        }

        // Apply distance constraints
        self.apply_distance_constraints();
    }

    fn apply_distance_constraints(&mut self) {
        for i in 1..self.bones.len() {
            let target_length = self.lengths[i - 1];
            let prev_pos = self.bones[i - 1].position;
            let curr_pos = self.bones[i].position;

            let delta = curr_pos - prev_pos;
            let current_length = delta.length();

            if current_length > 0.001 {
                let correction = delta.normalize() * (target_length - current_length);
                self.bones[i].position = curr_pos + correction * 0.5;
            }
        }
    }

    /// Returns positions of all bones in world space.
    pub fn world_positions(&self) -> Vec<Vec3> {
        self.bones
            .iter()
            .map(|b| self.root_position + self.root_rotation * b.position())
            .collect()
    }

    /// Returns the number of bones.
    pub fn bone_count(&self) -> usize {
        self.bones.len()
    }

    /// Resets all bones to rest state.
    pub fn reset(&mut self) {
        for bone in &mut self.bones {
            bone.reset();
        }
    }
}

/// Follow-through motion that delays response to parent changes.
#[derive(Debug, Clone)]
pub struct FollowThrough {
    /// Current value.
    value: Vec3,
    /// Target value.
    target: Vec3,
    /// Smoothing factor (0-1, lower = more lag).
    smoothing: f32,
}

impl Default for FollowThrough {
    fn default() -> Self {
        Self::new(0.1)
    }
}

impl FollowThrough {
    /// Creates a new follow-through with given smoothing.
    pub fn new(smoothing: f32) -> Self {
        Self {
            value: Vec3::ZERO,
            target: Vec3::ZERO,
            smoothing: smoothing.clamp(0.01, 1.0),
        }
    }

    /// Sets the smoothing factor.
    pub fn with_smoothing(mut self, smoothing: f32) -> Self {
        self.smoothing = smoothing.clamp(0.01, 1.0);
        self
    }

    /// Updates with a new target.
    pub fn update(&mut self, target: Vec3, dt: f32) {
        self.target = target;
        // Exponential smoothing
        let factor = 1.0 - (-self.smoothing * dt * 60.0).exp();
        self.value = self.value + (self.target - self.value) * factor;
    }

    /// Returns the current smoothed value.
    pub fn value(&self) -> Vec3 {
        self.value
    }

    /// Returns the lag (difference from target).
    pub fn lag(&self) -> Vec3 {
        self.target - self.value
    }

    /// Resets to target.
    pub fn reset(&mut self, value: Vec3) {
        self.value = value;
        self.target = value;
    }
}

/// Rotation follow-through for quaternions.
#[derive(Debug, Clone)]
pub struct RotationFollowThrough {
    /// Current rotation.
    rotation: Quat,
    /// Target rotation.
    target: Quat,
    /// Smoothing factor.
    smoothing: f32,
}

impl Default for RotationFollowThrough {
    fn default() -> Self {
        Self::new(0.1)
    }
}

impl RotationFollowThrough {
    /// Creates a new rotation follow-through.
    pub fn new(smoothing: f32) -> Self {
        Self {
            rotation: Quat::IDENTITY,
            target: Quat::IDENTITY,
            smoothing: smoothing.clamp(0.01, 1.0),
        }
    }

    /// Updates with a new target.
    pub fn update(&mut self, target: Quat, dt: f32) {
        self.target = target;
        let factor = 1.0 - (-self.smoothing * dt * 60.0).exp();
        self.rotation = self.rotation.slerp(self.target, factor);
    }

    /// Returns the current rotation.
    pub fn rotation(&self) -> Quat {
        self.rotation
    }

    /// Resets to target.
    pub fn reset(&mut self, rotation: Quat) {
        self.rotation = rotation;
        self.target = rotation;
    }
}

/// Overlapping action with multiple time delays.
#[derive(Debug, Clone)]
pub struct OverlappingAction {
    /// Follow-through instances with different delays.
    layers: Vec<FollowThrough>,
    /// Weights for each layer.
    weights: Vec<f32>,
}

impl OverlappingAction {
    /// Creates overlapping action with given delay values.
    ///
    /// Each delay creates a layer that follows at that rate.
    pub fn new(delays: &[f32]) -> Self {
        let layers: Vec<_> = delays.iter().map(|&d| FollowThrough::new(d)).collect();
        let weights = vec![1.0 / layers.len() as f32; layers.len()];
        Self { layers, weights }
    }

    /// Creates with custom weights.
    pub fn with_weights(mut self, weights: &[f32]) -> Self {
        let total: f32 = weights.iter().sum();
        self.weights = weights.iter().map(|&w| w / total).collect();
        if self.weights.len() < self.layers.len() {
            let last = *self.weights.last().unwrap_or(&0.0);
            self.weights.resize(self.layers.len(), last);
        }
        self
    }

    /// Updates all layers.
    pub fn update(&mut self, target: Vec3, dt: f32) {
        for layer in &mut self.layers {
            layer.update(target, dt);
        }
    }

    /// Returns the weighted average value.
    pub fn value(&self) -> Vec3 {
        let mut result = Vec3::ZERO;
        for (layer, &weight) in self.layers.iter().zip(&self.weights) {
            result += layer.value() * weight;
        }
        result
    }

    /// Returns the value at a specific layer.
    pub fn layer_value(&self, index: usize) -> Option<Vec3> {
        self.layers.get(index).map(|l| l.value())
    }

    /// Resets all layers.
    pub fn reset(&mut self, value: Vec3) {
        for layer in &mut self.layers {
            layer.reset(value);
        }
    }
}

/// Drag effect that resists velocity.
#[derive(Debug, Clone)]
pub struct Drag {
    /// Current velocity.
    velocity: Vec3,
    /// Drag coefficient.
    coefficient: f32,
}

impl Default for Drag {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl Drag {
    /// Creates a new drag effect.
    pub fn new(coefficient: f32) -> Self {
        Self {
            velocity: Vec3::ZERO,
            coefficient: coefficient.max(0.0),
        }
    }

    /// Updates with current position and computes drag force.
    pub fn update(&mut self, position: Vec3, prev_position: Vec3, dt: f32) -> Vec3 {
        if dt <= 0.0 {
            return Vec3::ZERO;
        }

        self.velocity = (position - prev_position) / dt;
        -self.velocity * self.coefficient
    }

    /// Returns the current velocity.
    pub fn velocity(&self) -> Vec3 {
        self.velocity
    }
}

/// Combined secondary motion controller.
#[derive(Debug, Clone)]
pub struct SecondaryMotion {
    /// Jiggle bones by name.
    jiggle_bones: Vec<(String, JiggleBone)>,
    /// Jiggle chains by name.
    jiggle_chains: Vec<(String, JiggleChain)>,
    /// Follow-through effects by name.
    follow_throughs: Vec<(String, FollowThrough)>,
    /// Rotation follow-throughs by name.
    rotation_follow_throughs: Vec<(String, RotationFollowThrough)>,
}

impl Default for SecondaryMotion {
    fn default() -> Self {
        Self::new()
    }
}

impl SecondaryMotion {
    /// Creates a new secondary motion controller.
    pub fn new() -> Self {
        Self {
            jiggle_bones: Vec::new(),
            jiggle_chains: Vec::new(),
            follow_throughs: Vec::new(),
            rotation_follow_throughs: Vec::new(),
        }
    }

    /// Adds a jiggle bone.
    pub fn add_jiggle_bone(&mut self, name: &str, bone: JiggleBone) {
        self.jiggle_bones.push((name.to_string(), bone));
    }

    /// Adds a jiggle chain.
    pub fn add_jiggle_chain(&mut self, name: &str, chain: JiggleChain) {
        self.jiggle_chains.push((name.to_string(), chain));
    }

    /// Adds a follow-through.
    pub fn add_follow_through(&mut self, name: &str, ft: FollowThrough) {
        self.follow_throughs.push((name.to_string(), ft));
    }

    /// Adds a rotation follow-through.
    pub fn add_rotation_follow_through(&mut self, name: &str, ft: RotationFollowThrough) {
        self.rotation_follow_throughs.push((name.to_string(), ft));
    }

    /// Updates a jiggle bone by name.
    pub fn update_jiggle_bone(
        &mut self,
        name: &str,
        parent_position: Vec3,
        parent_rotation: Quat,
        dt: f32,
    ) {
        if let Some((_, bone)) = self.jiggle_bones.iter_mut().find(|(n, _)| n == name) {
            bone.update(parent_position, parent_rotation, dt);
        }
    }

    /// Updates a jiggle chain by name.
    pub fn update_jiggle_chain(
        &mut self,
        name: &str,
        root_position: Vec3,
        root_rotation: Quat,
        dt: f32,
    ) {
        if let Some((_, chain)) = self.jiggle_chains.iter_mut().find(|(n, _)| n == name) {
            chain.update(root_position, root_rotation, dt);
        }
    }

    /// Updates a follow-through by name.
    pub fn update_follow_through(&mut self, name: &str, target: Vec3, dt: f32) {
        if let Some((_, ft)) = self.follow_throughs.iter_mut().find(|(n, _)| n == name) {
            ft.update(target, dt);
        }
    }

    /// Updates a rotation follow-through by name.
    pub fn update_rotation_follow_through(&mut self, name: &str, target: Quat, dt: f32) {
        if let Some((_, ft)) = self
            .rotation_follow_throughs
            .iter_mut()
            .find(|(n, _)| n == name)
        {
            ft.update(target, dt);
        }
    }

    /// Gets a jiggle bone offset by name.
    pub fn get_jiggle_offset(&self, name: &str) -> Option<Vec3> {
        self.jiggle_bones
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, b)| b.offset())
    }

    /// Gets jiggle chain positions by name.
    pub fn get_chain_positions(&self, name: &str) -> Option<Vec<Vec3>> {
        self.jiggle_chains
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, c)| c.world_positions())
    }

    /// Gets a follow-through value by name.
    pub fn get_follow_through(&self, name: &str) -> Option<Vec3> {
        self.follow_throughs
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, ft)| ft.value())
    }

    /// Gets a rotation follow-through by name.
    pub fn get_rotation_follow_through(&self, name: &str) -> Option<Quat> {
        self.rotation_follow_throughs
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, ft)| ft.rotation())
    }

    /// Resets all secondary motion.
    pub fn reset(&mut self) {
        for (_, bone) in &mut self.jiggle_bones {
            bone.reset();
        }
        for (_, chain) in &mut self.jiggle_chains {
            chain.reset();
        }
        for (_, ft) in &mut self.follow_throughs {
            ft.reset(Vec3::ZERO);
        }
        for (_, ft) in &mut self.rotation_follow_throughs {
            ft.reset(Quat::IDENTITY);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secondary_config_presets() {
        let soft = SecondaryConfig::soft();
        let stiff = SecondaryConfig::stiff();

        assert!(soft.stiffness < stiff.stiffness);
        assert!(soft.max_displacement > stiff.max_displacement);
    }

    #[test]
    fn test_jiggle_bone_creation() {
        let bone = JiggleBone::new()
            .with_stiffness(50.0)
            .with_damping(5.0)
            .with_rest_position(Vec3::new(0.0, 1.0, 0.0));

        assert_eq!(bone.position(), Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(bone.velocity(), Vec3::ZERO);
    }

    #[test]
    fn test_jiggle_bone_update() {
        let mut bone = JiggleBone::new()
            .with_stiffness(100.0)
            .with_damping(10.0)
            .with_gravity(Vec3::ZERO);

        // Move parent
        bone.update(Vec3::ZERO, Quat::IDENTITY, 0.016);
        bone.update(Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY, 0.016);

        // Bone should lag behind
        let offset = bone.offset();
        assert!(offset.x < 1.0);
    }

    #[test]
    fn test_jiggle_bone_reset() {
        let mut bone = JiggleBone::new().with_rest_position(Vec3::X);

        bone.update(Vec3::Y, Quat::IDENTITY, 0.1);
        bone.reset();

        assert_eq!(bone.position(), Vec3::X);
        assert_eq!(bone.velocity(), Vec3::ZERO);
    }

    #[test]
    fn test_jiggle_chain() {
        let config = SecondaryConfig::tail();
        let mut chain = JiggleChain::new(5, 0.2, config);

        assert_eq!(chain.bone_count(), 5);

        chain.update(Vec3::ZERO, Quat::IDENTITY, 0.016);
        let positions = chain.world_positions();
        assert_eq!(positions.len(), 5);
    }

    #[test]
    fn test_follow_through() {
        let mut ft = FollowThrough::new(0.1);

        ft.update(Vec3::X, 0.016);
        assert!(ft.value().x > 0.0 && ft.value().x < 1.0);

        // After many updates, should approach target
        for _ in 0..100 {
            ft.update(Vec3::X, 0.016);
        }
        assert!((ft.value() - Vec3::X).length() < 0.01);
    }

    #[test]
    fn test_follow_through_lag() {
        let mut ft = FollowThrough::new(0.05);
        ft.reset(Vec3::ZERO);

        ft.update(Vec3::X, 0.016);
        let lag = ft.lag();

        assert!(lag.x > 0.0); // Target is ahead
    }

    #[test]
    fn test_rotation_follow_through() {
        let mut ft = RotationFollowThrough::new(0.1);

        let target = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        ft.update(target, 0.016);

        assert!(ft.rotation() != Quat::IDENTITY);
        assert!(ft.rotation() != target);
    }

    #[test]
    fn test_overlapping_action() {
        let mut overlap = OverlappingAction::new(&[0.5, 0.1, 0.05]);

        // After multiple updates, faster layers should be closer to target
        for _ in 0..10 {
            overlap.update(Vec3::X, 0.016);
        }

        // Each layer should have different values
        let v0 = overlap.layer_value(0).unwrap();
        let v1 = overlap.layer_value(1).unwrap();
        let v2 = overlap.layer_value(2).unwrap();

        // Faster layer (higher smoothing) should be closer to target
        assert!(v0.x >= v1.x, "v0={} should be >= v1={}", v0.x, v1.x);
        assert!(v1.x >= v2.x, "v1={} should be >= v2={}", v1.x, v2.x);
    }

    #[test]
    fn test_overlapping_weights() {
        let overlap = OverlappingAction::new(&[0.1, 0.2]).with_weights(&[0.8, 0.2]);

        assert_eq!(overlap.weights.len(), 2);
    }

    #[test]
    fn test_drag() {
        let mut drag = Drag::new(0.5);

        let force = drag.update(Vec3::X, Vec3::ZERO, 0.016);

        // Drag force should oppose velocity
        assert!(force.x < 0.0);
        assert!(drag.velocity().x > 0.0);
    }

    #[test]
    fn test_secondary_motion_controller() {
        let mut sm = SecondaryMotion::new();

        sm.add_jiggle_bone("hair", JiggleBone::new());
        sm.add_jiggle_chain("tail", JiggleChain::new(3, 0.5, SecondaryConfig::tail()));
        sm.add_follow_through("hip_sway", FollowThrough::new(0.1));

        sm.update_jiggle_bone("hair", Vec3::ZERO, Quat::IDENTITY, 0.016);
        sm.update_jiggle_chain("tail", Vec3::ZERO, Quat::IDENTITY, 0.016);
        sm.update_follow_through("hip_sway", Vec3::X, 0.016);

        assert!(sm.get_jiggle_offset("hair").is_some());
        assert!(sm.get_chain_positions("tail").is_some());
        assert!(sm.get_follow_through("hip_sway").is_some());
    }

    #[test]
    fn test_secondary_motion_reset() {
        let mut sm = SecondaryMotion::new();
        sm.add_jiggle_bone("test", JiggleBone::new());

        sm.update_jiggle_bone("test", Vec3::X, Quat::IDENTITY, 0.1);
        sm.reset();

        assert_eq!(sm.get_jiggle_offset("test"), Some(Vec3::ZERO));
    }

    #[test]
    fn test_max_displacement() {
        let mut bone = JiggleBone::new()
            .with_stiffness(1.0)
            .with_damping(0.1)
            .with_max_displacement(0.5)
            .with_gravity(Vec3::new(0.0, -100.0, 0.0));

        // Apply strong gravity for many frames
        for _ in 0..100 {
            bone.update(Vec3::ZERO, Quat::IDENTITY, 0.016);
        }

        // Should be clamped to max displacement
        assert!(bone.offset().length() <= 0.5 + 0.01);
    }
}
