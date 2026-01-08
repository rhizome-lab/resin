//! 2D vector graphics for resin.
//!
//! Provides path primitives, vector networks, and operations for 2D vector art.

mod network;
mod path;

pub use network::{
    Edge, EdgeHandle, EdgeId, EdgeType, HandleStyle, Node, NodeId, Region, VectorNetwork,
};
pub use path::{
    Path,
    PathBuilder,
    PathCommand,
    // Primitives
    circle,
    ellipse,
    line,
    polygon,
    polyline,
    rect,
    rect_centered,
    regular_polygon,
    rounded_rect,
    star,
};
