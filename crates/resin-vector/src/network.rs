//! Vector network representation for node-based vector editing.
//!
//! A vector network is a graph-based representation of 2D vector art where paths
//! can branch and merge, unlike traditional linear path representations.
//!
//! # Structure
//!
//! - Nodes are points in 2D space
//! - Edges connect nodes and can be lines or Bezier curves
//! - Paths can fork (one node connects to multiple edges) and join
//! - Closed regions are cycles in the graph that can be filled
//!
//! # Usage
//!
//! ```ignore
//! let mut net = VectorNetwork::new();
//!
//! // Add nodes
//! let n0 = net.add_node(Vec2::ZERO);
//! let n1 = net.add_node(Vec2::new(100.0, 0.0));
//! let n2 = net.add_node(Vec2::new(50.0, 100.0));
//!
//! // Connect with edges
//! net.add_line(n0, n1);
//! net.add_line(n1, n2);
//! net.add_line(n2, n0);
//!
//! // Detect closed regions for filling
//! let regions = net.find_regions();
//! ```

use crate::{Path, PathBuilder};
use glam::Vec2;
use std::collections::{HashMap, HashSet};

/// Index for a node in the network.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct NodeId(pub u32);

/// Index for an edge in the network.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct EdgeId(pub u32);

/// Sentinel value for null references.
const NULL_ID: u32 = u32::MAX;

impl NodeId {
    pub const NULL: NodeId = NodeId(NULL_ID);

    pub fn is_null(self) -> bool {
        self.0 == NULL_ID
    }
}

impl EdgeId {
    pub const NULL: EdgeId = EdgeId(NULL_ID);

    pub fn is_null(self) -> bool {
        self.0 == NULL_ID
    }
}

/// Handle style at a node for connected edges.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HandleStyle {
    /// No handles (sharp corner).
    None,
    /// Symmetric handles (same direction and length on both sides).
    Symmetric,
    /// Smooth handles (same direction but different lengths).
    Smooth,
    /// Free handles (independent directions and lengths).
    Free,
}

impl Default for HandleStyle {
    fn default() -> Self {
        HandleStyle::None
    }
}

/// A node in the vector network.
#[derive(Debug, Clone, Default)]
pub struct Node {
    /// Position in 2D space.
    pub position: Vec2,
    /// Handle style at this node.
    pub handle_style: HandleStyle,
    /// Connected edges (edge id, handle offset for that edge at this node).
    pub edges: Vec<EdgeHandle>,
}

/// An edge's handle at a specific node.
#[derive(Debug, Clone, Copy, Default)]
pub struct EdgeHandle {
    /// The edge this handle belongs to.
    pub edge: EdgeId,
    /// Handle offset relative to the node position.
    /// For the start of an edge, this is control1 offset.
    /// For the end of an edge, this is control2 offset.
    pub handle: Vec2,
}

/// Type of edge connection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EdgeType {
    /// Straight line segment.
    Line,
    /// Cubic Bezier curve.
    Cubic,
}

impl Default for EdgeType {
    fn default() -> Self {
        EdgeType::Line
    }
}

/// An edge connecting two nodes.
#[derive(Debug, Clone, Default)]
pub struct Edge {
    /// Start node.
    pub start: NodeId,
    /// End node.
    pub end: NodeId,
    /// Type of edge.
    pub edge_type: EdgeType,
    /// Control point 1 (for cubic curves, absolute position).
    pub control1: Vec2,
    /// Control point 2 (for cubic curves, absolute position).
    pub control2: Vec2,
}

impl Edge {
    /// Returns true if this edge connects the given node.
    pub fn connects(&self, node: NodeId) -> bool {
        self.start == node || self.end == node
    }

    /// Returns the other node connected by this edge.
    pub fn other_node(&self, node: NodeId) -> NodeId {
        if self.start == node {
            self.end
        } else {
            self.start
        }
    }
}

/// A closed region in the vector network (detected cycle).
#[derive(Debug, Clone)]
pub struct Region {
    /// Nodes forming the boundary of this region, in order.
    pub nodes: Vec<NodeId>,
    /// Edges forming the boundary of this region, in order.
    pub edges: Vec<EdgeId>,
}

impl Region {
    /// Converts this region to a Path for rendering.
    pub fn to_path(&self, network: &VectorNetwork) -> Path {
        if self.edges.is_empty() {
            return Path::new();
        }

        let mut builder = PathBuilder::new();

        // Start at first node
        let first_node = self.nodes[0];
        builder = builder.move_to(network.nodes[first_node.0 as usize].position);

        // Walk around the region
        for i in 0..self.edges.len() {
            let edge_id = self.edges[i];
            let edge = &network.edges[edge_id.0 as usize];
            let current_node = self.nodes[i];
            let next_node = self.nodes[(i + 1) % self.nodes.len()];

            // Determine direction
            let forward = edge.start == current_node;

            match edge.edge_type {
                EdgeType::Line => {
                    builder = builder.line_to(network.nodes[next_node.0 as usize].position);
                }
                EdgeType::Cubic => {
                    let (c1, c2) = if forward {
                        (edge.control1, edge.control2)
                    } else {
                        (edge.control2, edge.control1)
                    };
                    builder =
                        builder.cubic_to(c1, c2, network.nodes[next_node.0 as usize].position);
                }
            }
        }

        builder.close().build()
    }
}

/// A vector network: a graph of nodes connected by edges.
#[derive(Debug, Clone, Default)]
pub struct VectorNetwork {
    /// All nodes.
    pub nodes: Vec<Node>,
    /// All edges.
    pub edges: Vec<Edge>,
}

impl VectorNetwork {
    /// Creates an empty vector network.
    pub fn new() -> Self {
        Self::default()
    }

    // ==================== Node Operations ====================

    /// Adds a node at the given position.
    pub fn add_node(&mut self, position: Vec2) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(Node {
            position,
            handle_style: HandleStyle::None,
            edges: Vec::new(),
        });
        id
    }

    /// Returns the position of a node.
    pub fn node_position(&self, node: NodeId) -> Vec2 {
        self.nodes[node.0 as usize].position
    }

    /// Sets the position of a node.
    pub fn set_node_position(&mut self, node: NodeId, position: Vec2) {
        let delta = position - self.nodes[node.0 as usize].position;
        self.nodes[node.0 as usize].position = position;

        // Update connected edge control points
        for eh in &self.nodes[node.0 as usize].edges.clone() {
            let edge = &mut self.edges[eh.edge.0 as usize];
            if edge.edge_type == EdgeType::Cubic {
                if edge.start == node {
                    edge.control1 += delta;
                }
                if edge.end == node {
                    edge.control2 += delta;
                }
            }
        }
    }

    /// Returns the number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the valence (number of connected edges) of a node.
    pub fn node_valence(&self, node: NodeId) -> usize {
        self.nodes[node.0 as usize].edges.len()
    }

    /// Returns the edges connected to a node.
    pub fn node_edges(&self, node: NodeId) -> Vec<EdgeId> {
        self.nodes[node.0 as usize]
            .edges
            .iter()
            .map(|eh| eh.edge)
            .collect()
    }

    /// Returns nodes adjacent to a given node.
    pub fn node_neighbors(&self, node: NodeId) -> Vec<NodeId> {
        self.nodes[node.0 as usize]
            .edges
            .iter()
            .map(|eh| self.edges[eh.edge.0 as usize].other_node(node))
            .collect()
    }

    // ==================== Edge Operations ====================

    /// Adds a line edge between two nodes.
    pub fn add_line(&mut self, start: NodeId, end: NodeId) -> EdgeId {
        let id = EdgeId(self.edges.len() as u32);

        let start_pos = self.nodes[start.0 as usize].position;
        let end_pos = self.nodes[end.0 as usize].position;

        self.edges.push(Edge {
            start,
            end,
            edge_type: EdgeType::Line,
            control1: start_pos,
            control2: end_pos,
        });

        // Add to node edge lists
        self.nodes[start.0 as usize].edges.push(EdgeHandle {
            edge: id,
            handle: Vec2::ZERO,
        });
        self.nodes[end.0 as usize].edges.push(EdgeHandle {
            edge: id,
            handle: Vec2::ZERO,
        });

        id
    }

    /// Adds a cubic Bezier edge between two nodes.
    pub fn add_cubic(
        &mut self,
        start: NodeId,
        control1: Vec2,
        control2: Vec2,
        end: NodeId,
    ) -> EdgeId {
        let id = EdgeId(self.edges.len() as u32);

        let start_pos = self.nodes[start.0 as usize].position;
        let end_pos = self.nodes[end.0 as usize].position;

        self.edges.push(Edge {
            start,
            end,
            edge_type: EdgeType::Cubic,
            control1,
            control2,
        });

        // Add to node edge lists with handle offsets
        self.nodes[start.0 as usize].edges.push(EdgeHandle {
            edge: id,
            handle: control1 - start_pos,
        });
        self.nodes[end.0 as usize].edges.push(EdgeHandle {
            edge: id,
            handle: control2 - end_pos,
        });

        id
    }

    /// Adds a smooth cubic edge where control points are computed automatically.
    pub fn add_smooth_edge(&mut self, start: NodeId, end: NodeId) -> EdgeId {
        let start_pos = self.nodes[start.0 as usize].position;
        let end_pos = self.nodes[end.0 as usize].position;

        // Place control points at 1/3 and 2/3 along the line
        let control1 = start_pos.lerp(end_pos, 1.0 / 3.0);
        let control2 = start_pos.lerp(end_pos, 2.0 / 3.0);

        self.add_cubic(start, control1, control2, end)
    }

    /// Returns the number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Sets control points for an edge (converts to cubic if needed).
    pub fn set_edge_controls(&mut self, edge: EdgeId, control1: Vec2, control2: Vec2) {
        let e = &mut self.edges[edge.0 as usize];
        e.edge_type = EdgeType::Cubic;
        e.control1 = control1;
        e.control2 = control2;

        // Update handles in connected nodes
        let start = e.start;
        let end = e.end;
        let start_pos = self.nodes[start.0 as usize].position;
        let end_pos = self.nodes[end.0 as usize].position;

        for eh in &mut self.nodes[start.0 as usize].edges {
            if eh.edge == edge {
                eh.handle = control1 - start_pos;
            }
        }
        for eh in &mut self.nodes[end.0 as usize].edges {
            if eh.edge == edge {
                eh.handle = control2 - end_pos;
            }
        }
    }

    /// Removes an edge from the network.
    pub fn remove_edge(&mut self, edge: EdgeId) {
        let e = &self.edges[edge.0 as usize];
        let start = e.start;
        let end = e.end;

        // Remove from node edge lists
        self.nodes[start.0 as usize]
            .edges
            .retain(|eh| eh.edge != edge);
        self.nodes[end.0 as usize]
            .edges
            .retain(|eh| eh.edge != edge);

        // Mark edge as removed (we don't actually remove to keep indices stable)
        self.edges[edge.0 as usize].start = NodeId::NULL;
        self.edges[edge.0 as usize].end = NodeId::NULL;
    }

    /// Returns true if an edge is valid (not removed).
    pub fn is_edge_valid(&self, edge: EdgeId) -> bool {
        !self.edges[edge.0 as usize].start.is_null()
    }

    // ==================== Path Conversion ====================

    /// Converts a sequence of connected edges to a Path.
    pub fn edges_to_path(&self, edges: &[EdgeId]) -> Path {
        if edges.is_empty() {
            return Path::new();
        }

        let mut builder = PathBuilder::new();

        // Find starting node
        let first_edge = &self.edges[edges[0].0 as usize];
        let mut current_node = first_edge.start;
        builder = builder.move_to(self.nodes[current_node.0 as usize].position);

        for &edge_id in edges {
            let edge = &self.edges[edge_id.0 as usize];

            // Determine direction
            let forward = edge.start == current_node;
            let next_node = if forward { edge.end } else { edge.start };

            match edge.edge_type {
                EdgeType::Line => {
                    builder = builder.line_to(self.nodes[next_node.0 as usize].position);
                }
                EdgeType::Cubic => {
                    let (c1, c2) = if forward {
                        (edge.control1, edge.control2)
                    } else {
                        (edge.control2, edge.control1)
                    };
                    builder = builder.cubic_to(c1, c2, self.nodes[next_node.0 as usize].position);
                }
            }

            current_node = next_node;
        }

        builder.build()
    }

    /// Converts the entire network to paths (one path per connected component stroke).
    pub fn to_paths(&self) -> Vec<Path> {
        let mut paths = Vec::new();
        let mut visited: HashSet<EdgeId> = HashSet::new();

        for i in 0..self.edges.len() {
            let edge_id = EdgeId(i as u32);
            if visited.contains(&edge_id) || !self.is_edge_valid(edge_id) {
                continue;
            }

            // Find a chain of edges
            let chain = self.trace_edge_chain(edge_id, &mut visited);
            if !chain.is_empty() {
                paths.push(self.edges_to_path(&chain));
            }
        }

        paths
    }

    /// Traces a chain of connected edges starting from the given edge.
    fn trace_edge_chain(&self, start: EdgeId, visited: &mut HashSet<EdgeId>) -> Vec<EdgeId> {
        let mut chain = vec![start];
        visited.insert(start);

        // Extend forward from end node
        let edge = &self.edges[start.0 as usize];
        let mut current = edge.end;

        loop {
            // Find an unvisited edge from current node
            let next = self.nodes[current.0 as usize]
                .edges
                .iter()
                .find(|eh| !visited.contains(&eh.edge) && self.is_edge_valid(eh.edge))
                .map(|eh| eh.edge);

            match next {
                Some(edge_id) => {
                    visited.insert(edge_id);
                    chain.push(edge_id);
                    current = self.edges[edge_id.0 as usize].other_node(current);
                }
                None => break,
            }
        }

        // Extend backward from start node
        let mut current = edge.start;
        let mut backward = Vec::new();

        loop {
            let next = self.nodes[current.0 as usize]
                .edges
                .iter()
                .find(|eh| !visited.contains(&eh.edge) && self.is_edge_valid(eh.edge))
                .map(|eh| eh.edge);

            match next {
                Some(edge_id) => {
                    visited.insert(edge_id);
                    backward.push(edge_id);
                    current = self.edges[edge_id.0 as usize].other_node(current);
                }
                None => break,
            }
        }

        // Combine: backward (reversed) + chain
        backward.reverse();
        backward.extend(chain);
        backward
    }

    // ==================== Region Detection ====================

    /// Finds all closed regions (faces) in the network.
    ///
    /// Uses a planar graph algorithm to detect minimal cycles.
    pub fn find_regions(&self) -> Vec<Region> {
        let mut regions = Vec::new();

        if self.edges.is_empty() {
            return regions;
        }

        // Build adjacency information sorted by angle
        let adjacency = self.build_angular_adjacency();

        // Track which directed edges have been used
        let mut used: HashSet<(NodeId, NodeId)> = HashSet::new();

        // For each directed edge, try to trace a face
        for i in 0..self.edges.len() {
            let edge = &self.edges[i];
            if edge.start.is_null() {
                continue;
            }

            // Try both directions
            for &(start, end) in &[(edge.start, edge.end), (edge.end, edge.start)] {
                if used.contains(&(start, end)) {
                    continue;
                }

                if let Some(region) = self.trace_region(start, end, &adjacency, &mut used) {
                    regions.push(region);
                }
            }
        }

        regions
    }

    /// Builds adjacency lists sorted by angle for planar face detection.
    fn build_angular_adjacency(&self) -> HashMap<NodeId, Vec<(NodeId, EdgeId)>> {
        let mut adj: HashMap<NodeId, Vec<(NodeId, EdgeId)>> = HashMap::new();

        for (i, edge) in self.edges.iter().enumerate() {
            if edge.start.is_null() {
                continue;
            }

            let edge_id = EdgeId(i as u32);
            adj.entry(edge.start).or_default().push((edge.end, edge_id));
            adj.entry(edge.end).or_default().push((edge.start, edge_id));
        }

        // Sort each adjacency list by angle
        for (node_id, neighbors) in &mut adj {
            let center = self.nodes[node_id.0 as usize].position;
            neighbors.sort_by(|a, b| {
                let angle_a = (self.nodes[a.0.0 as usize].position - center).angle();
                let angle_b = (self.nodes[b.0.0 as usize].position - center).angle();
                angle_a.partial_cmp(&angle_b).unwrap()
            });
        }

        adj
    }

    /// Traces a region starting from a directed edge.
    fn trace_region(
        &self,
        start: NodeId,
        second: NodeId,
        adjacency: &HashMap<NodeId, Vec<(NodeId, EdgeId)>>,
        used: &mut HashSet<(NodeId, NodeId)>,
    ) -> Option<Region> {
        let mut nodes = vec![start];
        let mut edges = Vec::new();

        let mut prev = start;
        let mut current = second;

        let max_steps = self.nodes.len() * 2;
        let mut steps = 0;

        loop {
            if steps > max_steps {
                return None; // Prevent infinite loops
            }
            steps += 1;

            // Find the edge between prev and current
            let edge_id = self.find_edge_between(prev, current)?;
            edges.push(edge_id);
            used.insert((prev, current));

            nodes.push(current);

            // If we've returned to start, we have a region
            if current == start {
                // Remove the duplicate start node at the end
                nodes.pop();
                return Some(Region { nodes, edges });
            }

            // Find next node: turn right (clockwise) from incoming direction
            let neighbors = adjacency.get(&current)?;
            let incoming_angle = (self.nodes[prev.0 as usize].position
                - self.nodes[current.0 as usize].position)
                .angle();

            // Find the neighbor just after the incoming direction (clockwise)
            let mut next = None;
            let mut best_angle = f32::MAX;

            for &(neighbor, _) in neighbors {
                if neighbor == prev {
                    continue;
                }

                let outgoing_angle = (self.nodes[neighbor.0 as usize].position
                    - self.nodes[current.0 as usize].position)
                    .angle();

                // Compute clockwise angle difference
                let mut diff = incoming_angle - outgoing_angle;
                if diff <= 0.0 {
                    diff += std::f32::consts::TAU;
                }

                if diff < best_angle {
                    best_angle = diff;
                    next = Some(neighbor);
                }
            }

            match next {
                Some(n) => {
                    prev = current;
                    current = n;
                }
                None => return None, // Dead end
            }
        }
    }

    /// Finds the edge connecting two nodes.
    fn find_edge_between(&self, a: NodeId, b: NodeId) -> Option<EdgeId> {
        for eh in &self.nodes[a.0 as usize].edges {
            let edge = &self.edges[eh.edge.0 as usize];
            if (edge.start == a && edge.end == b) || (edge.start == b && edge.end == a) {
                return Some(eh.edge);
            }
        }
        None
    }

    // ==================== Network Editing ====================

    /// Splits an edge at a parameter value t (0-1).
    pub fn split_edge(&mut self, edge: EdgeId, t: f32) -> NodeId {
        let e = &self.edges[edge.0 as usize];
        let start = e.start;
        let end = e.end;
        let edge_type = e.edge_type;

        let new_pos = match edge_type {
            EdgeType::Line => {
                let start_pos = self.nodes[start.0 as usize].position;
                let end_pos = self.nodes[end.0 as usize].position;
                start_pos.lerp(end_pos, t)
            }
            EdgeType::Cubic => {
                // De Casteljau's algorithm for cubic Bezier
                let p0 = self.nodes[start.0 as usize].position;
                let p1 = e.control1;
                let p2 = e.control2;
                let p3 = self.nodes[end.0 as usize].position;

                let q0 = p0.lerp(p1, t);
                let q1 = p1.lerp(p2, t);
                let q2 = p2.lerp(p3, t);

                let r0 = q0.lerp(q1, t);
                let r1 = q1.lerp(q2, t);

                r0.lerp(r1, t)
            }
        };

        // Add new node
        let new_node = self.add_node(new_pos);

        // Get original control points for cubic
        let (c1, c2) = if edge_type == EdgeType::Cubic {
            let e = &self.edges[edge.0 as usize];
            (e.control1, e.control2)
        } else {
            (Vec2::ZERO, Vec2::ZERO)
        };

        // Remove old edge
        self.remove_edge(edge);

        // Add two new edges
        match edge_type {
            EdgeType::Line => {
                self.add_line(start, new_node);
                self.add_line(new_node, end);
            }
            EdgeType::Cubic => {
                let p0 = self.nodes[start.0 as usize].position;
                let p3 = self.nodes[end.0 as usize].position;

                let q0 = p0.lerp(c1, t);
                let q1 = c1.lerp(c2, t);
                let q2 = c2.lerp(p3, t);

                let r0 = q0.lerp(q1, t);
                let r1 = q1.lerp(q2, t);

                self.add_cubic(start, q0, r0, new_node);
                self.add_cubic(new_node, r1, q2, end);
            }
        }

        new_node
    }

    /// Welds two nodes into one, merging their edges.
    pub fn weld_nodes(&mut self, keep: NodeId, remove: NodeId) {
        // Update all edges pointing to `remove` to point to `keep`
        let edges_to_update: Vec<_> = self.nodes[remove.0 as usize]
            .edges
            .iter()
            .map(|eh| eh.edge)
            .collect();

        for edge_id in edges_to_update {
            let edge = &mut self.edges[edge_id.0 as usize];
            if edge.start == remove {
                edge.start = keep;
            }
            if edge.end == remove {
                edge.end = keep;
            }

            // Add edge to keep's list if not already there
            if !self.nodes[keep.0 as usize]
                .edges
                .iter()
                .any(|eh| eh.edge == edge_id)
            {
                self.nodes[keep.0 as usize].edges.push(EdgeHandle {
                    edge: edge_id,
                    handle: Vec2::ZERO,
                });
            }
        }

        // Clear removed node's edges
        self.nodes[remove.0 as usize].edges.clear();
    }

    /// Computes the bounding box of the network.
    pub fn bounds(&self) -> (Vec2, Vec2) {
        if self.nodes.is_empty() {
            return (Vec2::ZERO, Vec2::ZERO);
        }

        let mut min = Vec2::splat(f32::MAX);
        let mut max = Vec2::splat(f32::MIN);

        for node in &self.nodes {
            min = min.min(node.position);
            max = max.max(node.position);
        }

        // Include control points
        for edge in &self.edges {
            if edge.start.is_null() {
                continue;
            }
            if edge.edge_type == EdgeType::Cubic {
                min = min.min(edge.control1).min(edge.control2);
                max = max.max(edge.control1).max(edge.control2);
            }
        }

        (min, max)
    }
}

/// Trait extension for Vec2 to compute angle (used internally for region detection).
trait Vec2Angle {
    fn angle(self) -> f32;
}

impl Vec2Angle for Vec2 {
    fn angle(self) -> f32 {
        self.y.atan2(self.x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_nodes_and_line() {
        let mut net = VectorNetwork::new();
        let n0 = net.add_node(Vec2::ZERO);
        let n1 = net.add_node(Vec2::new(100.0, 0.0));
        let e = net.add_line(n0, n1);

        assert_eq!(net.node_count(), 2);
        assert_eq!(net.edge_count(), 1);
        assert_eq!(net.node_valence(n0), 1);
        assert_eq!(net.node_valence(n1), 1);
        assert!(net.is_edge_valid(e));
    }

    #[test]
    fn test_add_cubic() {
        let mut net = VectorNetwork::new();
        let n0 = net.add_node(Vec2::ZERO);
        let n1 = net.add_node(Vec2::new(100.0, 0.0));
        let e = net.add_cubic(n0, Vec2::new(30.0, 50.0), Vec2::new(70.0, 50.0), n1);

        assert_eq!(net.edges[e.0 as usize].edge_type, EdgeType::Cubic);
    }

    #[test]
    fn test_triangle_region() {
        let mut net = VectorNetwork::new();
        let n0 = net.add_node(Vec2::new(0.0, 0.0));
        let n1 = net.add_node(Vec2::new(100.0, 0.0));
        let n2 = net.add_node(Vec2::new(50.0, 100.0));

        net.add_line(n0, n1);
        net.add_line(n1, n2);
        net.add_line(n2, n0);

        let regions = net.find_regions();
        // Should find at least one region (the triangle)
        assert!(!regions.is_empty());
    }

    #[test]
    fn test_edges_to_path() {
        let mut net = VectorNetwork::new();
        let n0 = net.add_node(Vec2::ZERO);
        let n1 = net.add_node(Vec2::new(100.0, 0.0));
        let n2 = net.add_node(Vec2::new(100.0, 100.0));

        let e0 = net.add_line(n0, n1);
        let e1 = net.add_line(n1, n2);

        let path = net.edges_to_path(&[e0, e1]);
        assert_eq!(path.len(), 3); // move, line, line
    }

    #[test]
    fn test_region_to_path() {
        let mut net = VectorNetwork::new();
        let n0 = net.add_node(Vec2::new(0.0, 0.0));
        let n1 = net.add_node(Vec2::new(100.0, 0.0));
        let n2 = net.add_node(Vec2::new(50.0, 100.0));

        net.add_line(n0, n1);
        net.add_line(n1, n2);
        net.add_line(n2, n0);

        let regions = net.find_regions();
        if !regions.is_empty() {
            let path = regions[0].to_path(&net);
            assert!(path.len() >= 4); // move, 3 lines, close
        }
    }

    #[test]
    fn test_split_edge() {
        let mut net = VectorNetwork::new();
        let n0 = net.add_node(Vec2::ZERO);
        let n1 = net.add_node(Vec2::new(100.0, 0.0));
        let e = net.add_line(n0, n1);

        let n_mid = net.split_edge(e, 0.5);

        // Original edge should be removed, new edges created
        assert!(!net.is_edge_valid(e));
        assert_eq!(net.node_count(), 3);

        // New node should be at midpoint
        let mid_pos = net.node_position(n_mid);
        assert!((mid_pos.x - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_remove_edge() {
        let mut net = VectorNetwork::new();
        let n0 = net.add_node(Vec2::ZERO);
        let n1 = net.add_node(Vec2::new(100.0, 0.0));
        let e = net.add_line(n0, n1);

        net.remove_edge(e);

        assert!(!net.is_edge_valid(e));
        assert_eq!(net.node_valence(n0), 0);
        assert_eq!(net.node_valence(n1), 0);
    }

    #[test]
    fn test_node_neighbors() {
        let mut net = VectorNetwork::new();
        let n0 = net.add_node(Vec2::ZERO);
        let n1 = net.add_node(Vec2::new(100.0, 0.0));
        let n2 = net.add_node(Vec2::new(0.0, 100.0));

        net.add_line(n0, n1);
        net.add_line(n0, n2);

        let neighbors = net.node_neighbors(n0);
        assert_eq!(neighbors.len(), 2);
    }

    #[test]
    fn test_bounds() {
        let mut net = VectorNetwork::new();
        net.add_node(Vec2::new(10.0, 20.0));
        net.add_node(Vec2::new(100.0, 50.0));
        net.add_node(Vec2::new(50.0, 200.0));

        let (min, max) = net.bounds();
        assert!((min.x - 10.0).abs() < 0.01);
        assert!((min.y - 20.0).abs() < 0.01);
        assert!((max.x - 100.0).abs() < 0.01);
        assert!((max.y - 200.0).abs() < 0.01);
    }

    #[test]
    fn test_to_paths() {
        let mut net = VectorNetwork::new();
        let n0 = net.add_node(Vec2::ZERO);
        let n1 = net.add_node(Vec2::new(100.0, 0.0));
        let n2 = net.add_node(Vec2::new(100.0, 100.0));

        net.add_line(n0, n1);
        net.add_line(n1, n2);

        let paths = net.to_paths();
        assert!(!paths.is_empty());
    }

    #[test]
    fn test_weld_nodes() {
        let mut net = VectorNetwork::new();
        let n0 = net.add_node(Vec2::ZERO);
        let n1 = net.add_node(Vec2::new(100.0, 0.0));
        let n2 = net.add_node(Vec2::new(100.0, 0.0)); // Same position as n1
        let n3 = net.add_node(Vec2::new(200.0, 0.0));

        net.add_line(n0, n1);
        net.add_line(n2, n3);

        net.weld_nodes(n1, n2);

        // n2 should have no edges
        assert_eq!(net.node_valence(n2), 0);
    }
}
