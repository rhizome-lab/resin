# Plugin Architecture

How third-party code extends resin with custom operations.

## Principle

**Resin defines the contract. Host handles loading.**

Resin is a library, not a framework. Different hosts have different needs:
- Game engine: no user plugins, just built-in ops
- DAW: existing Lua scripting system
- Standalone tool: WASM sandbox for security
- Internal tool: native plugins, trusted code

Forcing a plugin system on everyone would conflict with our modular philosophy.

## Layers

```
┌─────────────────────────────────────────────────┐
│  Host Application (DAW, modeler, game, etc.)    │
├─────────────────────────────────────────────────┤
│  Optional: resin-wasm-plugins / resin-lua / ... │  ← adapters
├─────────────────────────────────────────────────┤
│  resin-core: traits + serialization contract    │  ← resin provides
└─────────────────────────────────────────────────┘
```

## What Resin Provides

### 1. Op Traits

```rust
/// Core trait for mesh operations
pub trait MeshOp: Send + Sync {
    fn apply(&self, mesh: &Mesh) -> Mesh;
}

/// Similar for other domains
pub trait AudioOp: Send + Sync { ... }
pub trait TextureOp: Send + Sync { ... }
```

### 2. Serialization Contract

Ops must be serializable to/from a common format (JSON, MessagePack, etc.):

```rust
pub trait SerializableOp {
    /// Unique type identifier, e.g. "resin::mesh::Subdivide"
    fn type_name(&self) -> &'static str;

    /// Serialize parameters to value
    fn to_value(&self) -> Value;

    /// Type-specific deserialization handled by registry
}
```

### 3. Registry Interface

```rust
pub struct OpRegistry<Op: ?Sized> {
    deserializers: HashMap<String, Box<dyn Fn(Value) -> Result<Box<Op>>>>,
}

impl<Op: ?Sized> OpRegistry<Op> {
    pub fn register<T>(&mut self, type_name: &str)
    where
        T: Op + DeserializeOwned + 'static,
    {
        self.deserializers.insert(
            type_name.to_string(),
            Box::new(|v| Ok(Box::new(serde_json::from_value::<T>(v)?))),
        );
    }

    pub fn deserialize(&self, type_name: &str, value: Value) -> Result<Box<Op>> {
        let deserialize = self.deserializers.get(type_name)?;
        deserialize(value)
    }
}
```

## What Host Provides

Plugin loading mechanism. Examples:

### Static Linking (Cargo crates)

Simplest case. User adds plugin crate to `Cargo.toml`, rebuilds app.

```rust
// In host's startup
registry.register::<my_plugin::CustomBevel>("myplugin::CustomBevel");
```

Works with `typetag` for automatic registration if all code is statically linked.

### WASM Plugins (optional adapter)

```rust
// resin-wasm-plugins crate
pub struct WasmPluginHost {
    engine: wasmtime::Engine,
    registry: OpRegistry<dyn MeshOp>,
}

impl WasmPluginHost {
    pub fn load_plugin(&mut self, wasm_bytes: &[u8]) -> Result<()> {
        // Instantiate WASM module
        // Call its register() export
        // Wrap WASM functions as trait impls
    }
}
```

### Lua Scripting (optional adapter)

```rust
// resin-lua-plugins crate
pub struct LuaPluginHost {
    lua: mlua::Lua,
    registry: OpRegistry<dyn MeshOp>,
}

impl LuaPluginHost {
    pub fn load_script(&mut self, source: &str) -> Result<()> {
        // Execute Lua script
        // Script calls register_op(name, apply_fn)
        // Wrap Lua functions as trait impls
    }
}
```

### Native Plugins (C ABI)

For hosts that need maximum performance and trust their plugins:

```rust
// Plugin exposes C ABI
#[no_mangle]
pub extern "C" fn register_ops(registry: *mut OpRegistry) { ... }

// Host loads with dlopen/LoadLibrary
```

## Graph Serialization

Graphs reference ops by type name:

```json
{
  "nodes": [
    { "id": 1, "op": "resin::mesh::Cube", "params": { "size": [1, 1, 1] } },
    { "id": 2, "op": "resin::mesh::Subdivide", "params": { "levels": 2 } },
    { "id": 3, "op": "myplugin::CustomBevel", "params": { "amount": 0.1 } }
  ],
  "edges": [[1, 2], [2, 3]]
}
```

Deserialization uses registry to resolve type names to implementations.

If a type name isn't registered, deserialization fails with clear error ("unknown op: myplugin::CustomBevel").

## Summary

| Component | Responsibility |
|-----------|----------------|
| resin-core | Op traits, serialization format, registry interface |
| resin-*-plugins | Optional adapters for common plugin models |
| Host application | Plugin discovery, loading, sandboxing |

This keeps resin focused and lets hosts make appropriate choices for their context.
