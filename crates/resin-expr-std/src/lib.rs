//! Standard function library for resin expressions.
//!
//! Provides common math functions (sin, cos, sqrt, etc.) that can be registered
//! with an expression `FunctionRegistry`.
//!
//! # Usage
//!
//! ```ignore
//! use resin_core::expr::{Expr, FunctionRegistry};
//! use resin_expr_std::register_std;
//!
//! let mut registry = FunctionRegistry::new();
//! register_std(&mut registry);
//!
//! let expr = Expr::parse("sin(x) + cos(y)").unwrap();
//! ```

use resin_core::expr::{Ast, ExprFn, FunctionRegistry};

// ============================================================================
// Macro for simple functions
// ============================================================================

macro_rules! define_fn {
    ($name:ident, $str_name:literal, $args:literal, |$($arg:ident),*| $body:expr) => {
        pub struct $name;

        impl ExprFn for $name {
            fn name(&self) -> &str { $str_name }
            fn arg_count(&self) -> usize { $args }
            fn interpret(&self, args: &[f32]) -> f32 {
                let [$($arg),*] = args else { return 0.0 };
                $body
            }
        }
    };
}

// ============================================================================
// Trigonometric functions
// ============================================================================

define_fn!(Sin, "sin", 1, |a| a.sin());
define_fn!(Cos, "cos", 1, |a| a.cos());
define_fn!(Tan, "tan", 1, |a| a.tan());
define_fn!(Asin, "asin", 1, |a| a.asin());
define_fn!(Acos, "acos", 1, |a| a.acos());
define_fn!(Atan, "atan", 1, |a| a.atan());
define_fn!(Atan2, "atan2", 2, |y, x| y.atan2(*x));
define_fn!(Sinh, "sinh", 1, |a| a.sinh());
define_fn!(Cosh, "cosh", 1, |a| a.cosh());
define_fn!(Tanh, "tanh", 1, |a| a.tanh());

// ============================================================================
// Exponential / logarithmic
// ============================================================================

define_fn!(Exp, "exp", 1, |a| a.exp());
define_fn!(Exp2, "exp2", 1, |a| a.exp2());
define_fn!(Log, "log", 1, |a| a.ln());
define_fn!(Ln, "ln", 1, |a| a.ln());
define_fn!(Log2, "log2", 1, |a| a.log2());
define_fn!(Log10, "log10", 1, |a| a.log10());
define_fn!(Pow, "pow", 2, |a, b| a.powf(*b));
define_fn!(Sqrt, "sqrt", 1, |a| a.sqrt());
define_fn!(InverseSqrt, "inversesqrt", 1, |a| 1.0 / a.sqrt());

// ============================================================================
// Common math functions
// ============================================================================

define_fn!(Abs, "abs", 1, |a| a.abs());
define_fn!(Sign, "sign", 1, |a| a.signum());
define_fn!(Floor, "floor", 1, |a| a.floor());
define_fn!(Ceil, "ceil", 1, |a| a.ceil());
define_fn!(Round, "round", 1, |a| a.round());
define_fn!(Trunc, "trunc", 1, |a| a.trunc());
define_fn!(Fract, "fract", 1, |a| a.fract());
define_fn!(Min, "min", 2, |a, b| a.min(*b));
define_fn!(Max, "max", 2, |a, b| a.max(*b));
define_fn!(Clamp, "clamp", 3, |x, lo, hi| x.clamp(*lo, *hi));
define_fn!(Saturate, "saturate", 1, |a| a.clamp(0.0, 1.0));

// ============================================================================
// Interpolation
// ============================================================================

/// Linear interpolation: lerp(a, b, t) = a + (b - a) * t
pub struct Lerp;
impl ExprFn for Lerp {
    fn name(&self) -> &str {
        "lerp"
    }
    fn arg_count(&self) -> usize {
        3
    }
    fn interpret(&self, args: &[f32]) -> f32 {
        let [a, b, t] = args else { return 0.0 };
        a + (b - a) * t
    }
}

/// Alias for lerp (GLSL naming)
pub struct Mix;
impl ExprFn for Mix {
    fn name(&self) -> &str {
        "mix"
    }
    fn arg_count(&self) -> usize {
        3
    }
    fn interpret(&self, args: &[f32]) -> f32 {
        let [a, b, t] = args else { return 0.0 };
        a + (b - a) * t
    }
}

/// Step function: step(edge, x) = x < edge ? 0.0 : 1.0
pub struct Step;
impl ExprFn for Step {
    fn name(&self) -> &str {
        "step"
    }
    fn arg_count(&self) -> usize {
        2
    }
    fn interpret(&self, args: &[f32]) -> f32 {
        let [edge, x] = args else { return 0.0 };
        if x < edge { 0.0 } else { 1.0 }
    }
}

/// Smooth Hermite interpolation
pub struct Smoothstep;
impl ExprFn for Smoothstep {
    fn name(&self) -> &str {
        "smoothstep"
    }
    fn arg_count(&self) -> usize {
        3
    }
    fn interpret(&self, args: &[f32]) -> f32 {
        let [edge0, edge1, x] = args else { return 0.0 };
        let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }
}

// ============================================================================
// Decomposition-based functions (work on all backends automatically)
// ============================================================================

/// Inverse lerp: inverse_lerp(a, b, v) = (v - a) / (b - a)
pub struct InverseLerp;
impl ExprFn for InverseLerp {
    fn name(&self) -> &str {
        "inverse_lerp"
    }
    fn arg_count(&self) -> usize {
        3
    }
    fn interpret(&self, args: &[f32]) -> f32 {
        let [a, b, v] = args else { return 0.0 };
        (v - a) / (b - a)
    }
    fn decompose(&self, args: &[Ast]) -> Option<Ast> {
        if args.len() != 3 {
            return None;
        }
        let a = &args[0];
        let b = &args[1];
        let v = &args[2];
        // (v - a) / (b - a)
        Some(Ast::BinOp(
            resin_core::expr::BinOp::Div,
            Box::new(Ast::BinOp(
                resin_core::expr::BinOp::Sub,
                Box::new(v.clone()),
                Box::new(a.clone()),
            )),
            Box::new(Ast::BinOp(
                resin_core::expr::BinOp::Sub,
                Box::new(b.clone()),
                Box::new(a.clone()),
            )),
        ))
    }
}

/// Remap: remap(x, in_lo, in_hi, out_lo, out_hi)
pub struct Remap;
impl ExprFn for Remap {
    fn name(&self) -> &str {
        "remap"
    }
    fn arg_count(&self) -> usize {
        5
    }
    fn interpret(&self, args: &[f32]) -> f32 {
        let [x, in_lo, in_hi, out_lo, out_hi] = args else {
            return 0.0;
        };
        let t = (x - in_lo) / (in_hi - in_lo);
        out_lo + (out_hi - out_lo) * t
    }
}

// ============================================================================
// Noise functions
// ============================================================================

/// 2D Perlin noise
pub struct Noise;
impl ExprFn for Noise {
    fn name(&self) -> &str {
        "noise"
    }
    fn arg_count(&self) -> usize {
        2
    }
    fn interpret(&self, args: &[f32]) -> f32 {
        let [x, y] = args else { return 0.0 };
        resin_core::noise::perlin2(*x, *y)
    }
}

/// Alias for noise
pub struct Perlin;
impl ExprFn for Perlin {
    fn name(&self) -> &str {
        "perlin"
    }
    fn arg_count(&self) -> usize {
        2
    }
    fn interpret(&self, args: &[f32]) -> f32 {
        let [x, y] = args else { return 0.0 };
        resin_core::noise::perlin2(*x, *y)
    }
}

/// 2D Simplex noise
pub struct Simplex;
impl ExprFn for Simplex {
    fn name(&self) -> &str {
        "simplex"
    }
    fn arg_count(&self) -> usize {
        2
    }
    fn interpret(&self, args: &[f32]) -> f32 {
        let [x, y] = args else { return 0.0 };
        resin_core::noise::simplex2(*x, *y)
    }
}

// ============================================================================
// Registry
// ============================================================================

/// Registers all standard functions into the given registry.
pub fn register_std(registry: &mut FunctionRegistry) {
    // Trigonometric
    registry.register(Sin);
    registry.register(Cos);
    registry.register(Tan);
    registry.register(Asin);
    registry.register(Acos);
    registry.register(Atan);
    registry.register(Atan2);
    registry.register(Sinh);
    registry.register(Cosh);
    registry.register(Tanh);

    // Exponential / logarithmic
    registry.register(Exp);
    registry.register(Exp2);
    registry.register(Log);
    registry.register(Ln);
    registry.register(Log2);
    registry.register(Log10);
    registry.register(Pow);
    registry.register(Sqrt);
    registry.register(InverseSqrt);

    // Common math
    registry.register(Abs);
    registry.register(Sign);
    registry.register(Floor);
    registry.register(Ceil);
    registry.register(Round);
    registry.register(Trunc);
    registry.register(Fract);
    registry.register(Min);
    registry.register(Max);
    registry.register(Clamp);
    registry.register(Saturate);

    // Interpolation
    registry.register(Lerp);
    registry.register(Mix);
    registry.register(Step);
    registry.register(Smoothstep);
    registry.register(InverseLerp);
    registry.register(Remap);

    // Noise
    registry.register(Noise);
    registry.register(Perlin);
    registry.register(Simplex);
}

/// Creates a new registry with all standard functions.
pub fn std_registry() -> FunctionRegistry {
    let mut registry = FunctionRegistry::new();
    register_std(&mut registry);
    registry
}

#[cfg(test)]
mod tests {
    use super::*;
    use resin_core::EvalContext;
    use resin_core::expr::Expr;
    use resin_core::glam::Vec2;

    fn eval(expr: &str, x: f32, y: f32) -> f32 {
        let registry = std_registry();
        let expr = Expr::parse(expr).unwrap();
        let ctx = EvalContext::new();
        expr.eval(Vec2::new(x, y), &ctx, &registry).unwrap()
    }

    #[test]
    fn test_trig() {
        assert!(eval("sin(0)", 0.0, 0.0).abs() < 0.001);
        assert!((eval("cos(0)", 0.0, 0.0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_exp_log() {
        assert!((eval("exp(0)", 0.0, 0.0) - 1.0).abs() < 0.001);
        assert!((eval("ln(1)", 0.0, 0.0) - 0.0).abs() < 0.001);
        assert!((eval("sqrt(16)", 0.0, 0.0) - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_common() {
        assert_eq!(eval("abs(-5)", 0.0, 0.0), 5.0);
        assert_eq!(eval("floor(3.7)", 0.0, 0.0), 3.0);
        assert_eq!(eval("ceil(3.2)", 0.0, 0.0), 4.0);
        assert_eq!(eval("min(3, 7)", 0.0, 0.0), 3.0);
        assert_eq!(eval("max(3, 7)", 0.0, 0.0), 7.0);
        assert_eq!(eval("clamp(5, 0, 3)", 0.0, 0.0), 3.0);
        assert_eq!(eval("saturate(1.5)", 0.0, 0.0), 1.0);
    }

    #[test]
    fn test_interpolation() {
        assert_eq!(eval("lerp(0, 10, 0.5)", 0.0, 0.0), 5.0);
        assert_eq!(eval("mix(0, 10, 0.5)", 0.0, 0.0), 5.0);
        assert_eq!(eval("step(0.5, 0.3)", 0.0, 0.0), 0.0);
        assert_eq!(eval("step(0.5, 0.7)", 0.0, 0.0), 1.0);
        assert!((eval("smoothstep(0, 1, 0.5)", 0.0, 0.0) - 0.5).abs() < 0.1);
        assert_eq!(eval("inverse_lerp(0, 10, 5)", 0.0, 0.0), 0.5);
    }

    #[test]
    fn test_noise() {
        let v = eval("noise(x, y)", 0.5, 0.5);
        assert!((0.0..=1.0).contains(&v));

        let v = eval("perlin(x, y)", 0.5, 0.5);
        assert!((0.0..=1.0).contains(&v));

        let v = eval("simplex(x, y)", 0.5, 0.5);
        assert!((0.0..=1.0).contains(&v));
    }

    #[test]
    fn test_remap() {
        // remap 5 from [0,10] to [0,100] = 50
        assert_eq!(eval("remap(5, 0, 10, 0, 100)", 0.0, 0.0), 50.0);
    }
}
