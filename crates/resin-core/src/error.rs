//! Error types for resin-core.

use crate::value::ValueType;
use thiserror::Error;

/// Error when a value has the wrong type.
#[derive(Debug, Clone, Error)]
#[error("type error: expected {expected}, got {got}")]
pub struct TypeError {
    /// The type that was expected.
    pub expected: ValueType,
    /// The type that was actually provided.
    pub got: ValueType,
}

impl TypeError {
    /// Create a new type error.
    pub fn expected(expected: ValueType, got: ValueType) -> Self {
        Self { expected, got }
    }
}

/// Errors that can occur during graph operations.
#[derive(Debug, Clone, Error)]
pub enum GraphError {
    /// Node with the given ID was not found.
    #[error("node not found: {0}")]
    NodeNotFound(u32),

    /// Port on a node was not found.
    #[error("port not found: node {node}, port {port}")]
    PortNotFound {
        /// Node ID.
        node: u32,
        /// Port index.
        port: usize,
    },

    /// Type mismatch when connecting ports.
    #[error("type mismatch on edge: expected {expected}, got {got}")]
    TypeMismatch {
        /// Expected type.
        expected: ValueType,
        /// Actual type.
        got: ValueType,
    },

    /// Graph contains a cycle.
    #[error("cycle detected in graph")]
    CycleDetected,

    /// Required input port is not connected.
    #[error("unconnected input: node {node}, port {port}")]
    UnconnectedInput {
        /// Node ID.
        node: u32,
        /// Port index.
        port: usize,
    },

    /// Error during node execution.
    #[error("execution error: {0}")]
    ExecutionError(String),
}
