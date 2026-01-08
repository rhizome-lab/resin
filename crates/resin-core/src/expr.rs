//! Expression language for field evaluation.
//!
//! A simple expression parser that compiles string expressions into evaluable fields.
//! All functions are registered via the `ExprFn` trait - no hardcoded function set.
//!
//! # Syntax
//!
//! ```text
//! // Variables
//! x, y, z          // Coordinates
//! time             // Context time
//!
//! // Operators (precedence low to high)
//! a + b, a - b     // Addition, subtraction
//! a * b, a / b     // Multiplication, division
//! a ^ b            // Exponentiation
//! -a               // Negation
//!
//! // Functions (registered via ExprFn trait)
//! sin(x), cos(x), sqrt(x), abs(x)
//! min(a, b), max(a, b)
//! clamp(x, lo, hi), lerp(a, b, t)
//! noise(x, y)      // if registered
//! ```
//!
//! # Example
//!
//! ```ignore
//! use resin_core::expr::{Expr, FunctionRegistry, std_registry};
//! use resin_core::EvalContext;
//! use glam::Vec2;
//!
//! let registry = std_registry();  // sin, cos, etc.
//! let expr = Expr::parse("sin(x * 3.14) + 0.5").unwrap();
//! let ctx = EvalContext::new();
//! let value = expr.eval(Vec2::new(0.5, 0.0), &ctx, &registry);
//! ```

use crate::context::EvalContext;
use glam::Vec2;
use std::collections::HashMap;
use std::f32::consts::PI;
use std::sync::Arc;

// ============================================================================
// ExprFn trait and registry
// ============================================================================

/// A function that can be called from expressions.
///
/// All functions (including sin, cos, etc.) implement this trait.
/// There are no hardcoded functions - everything is registered.
pub trait ExprFn: Send + Sync {
    /// Function name (e.g., "sin", "perlin").
    fn name(&self) -> &str;

    /// Number of arguments this function expects.
    fn arg_count(&self) -> usize;

    /// CPU interpretation (required - universal fallback).
    fn interpret(&self, args: &[f32]) -> f32;

    /// Express as simpler expressions (enables automatic backend support).
    /// If this returns Some, backends can compile without knowing about this function.
    fn decompose(&self, _args: &[Ast]) -> Option<Ast> {
        None
    }
}

/// Registry of expression functions.
#[derive(Clone, Default)]
pub struct FunctionRegistry {
    funcs: HashMap<String, Arc<dyn ExprFn>>,
}

impl FunctionRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a function.
    pub fn register<F: ExprFn + 'static>(&mut self, func: F) {
        self.funcs.insert(func.name().to_string(), Arc::new(func));
    }

    /// Gets a function by name.
    pub fn get(&self, name: &str) -> Option<&Arc<dyn ExprFn>> {
        self.funcs.get(name)
    }
}

// ============================================================================
// Standard library functions
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

define_fn!(FnSin, "sin", 1, |a| a.sin());
define_fn!(FnCos, "cos", 1, |a| a.cos());
define_fn!(FnTan, "tan", 1, |a| a.tan());
define_fn!(FnSqrt, "sqrt", 1, |a| a.sqrt());
define_fn!(FnAbs, "abs", 1, |a| a.abs());
define_fn!(FnFloor, "floor", 1, |a| a.floor());
define_fn!(FnCeil, "ceil", 1, |a| a.ceil());
define_fn!(FnFract, "fract", 1, |a| a.fract());
define_fn!(FnMin, "min", 2, |a, b| a.min(*b));
define_fn!(FnMax, "max", 2, |a, b| a.max(*b));
define_fn!(FnClamp, "clamp", 3, |x, lo, hi| x.clamp(*lo, *hi));
define_fn!(FnLerp, "lerp", 3, |a, b, t| a + (b - a) * t);

// Also register "mix" as alias for lerp
pub struct FnMix;
impl ExprFn for FnMix {
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

// Noise function
pub struct FnNoise;
impl ExprFn for FnNoise {
    fn name(&self) -> &str {
        "noise"
    }
    fn arg_count(&self) -> usize {
        2
    }
    fn interpret(&self, args: &[f32]) -> f32 {
        let [x, y] = args else { return 0.0 };
        crate::noise::perlin2(*x, *y)
    }
}

// Alias for noise
pub struct FnPerlin;
impl ExprFn for FnPerlin {
    fn name(&self) -> &str {
        "perlin"
    }
    fn arg_count(&self) -> usize {
        2
    }
    fn interpret(&self, args: &[f32]) -> f32 {
        let [x, y] = args else { return 0.0 };
        crate::noise::perlin2(*x, *y)
    }
}

/// Creates a registry with standard math functions.
pub fn std_registry() -> FunctionRegistry {
    let mut r = FunctionRegistry::new();
    r.register(FnSin);
    r.register(FnCos);
    r.register(FnTan);
    r.register(FnSqrt);
    r.register(FnAbs);
    r.register(FnFloor);
    r.register(FnCeil);
    r.register(FnFract);
    r.register(FnMin);
    r.register(FnMax);
    r.register(FnClamp);
    r.register(FnLerp);
    r.register(FnMix);
    r.register(FnNoise);
    r.register(FnPerlin);
    r
}

// ============================================================================
// Parse error
// ============================================================================

/// Expression parse error.
#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    UnexpectedChar(char),
    UnexpectedEnd,
    UnexpectedToken(String),
    UnknownVariable(String),
    InvalidNumber(String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::UnexpectedChar(c) => write!(f, "unexpected character: '{}'", c),
            ParseError::UnexpectedEnd => write!(f, "unexpected end of expression"),
            ParseError::UnexpectedToken(t) => write!(f, "unexpected token: '{}'", t),
            ParseError::UnknownVariable(name) => write!(f, "unknown variable: '{}'", name),
            ParseError::InvalidNumber(s) => write!(f, "invalid number: '{}'", s),
        }
    }
}

impl std::error::Error for ParseError {}

/// Expression evaluation error.
#[derive(Debug, Clone, PartialEq)]
pub enum EvalError {
    UnknownFunction(String),
    WrongArgCount {
        func: String,
        expected: usize,
        got: usize,
    },
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvalError::UnknownFunction(name) => write!(f, "unknown function: '{}'", name),
            EvalError::WrongArgCount {
                func,
                expected,
                got,
            } => {
                write!(
                    f,
                    "function '{}' expects {} args, got {}",
                    func, expected, got
                )
            }
        }
    }
}

impl std::error::Error for EvalError {}

// ============================================================================
// Lexer
// ============================================================================

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(f32),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    LParen,
    RParen,
    Comma,
    Eof,
}

struct Lexer<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> Lexer<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, pos: 0 }
    }

    fn peek_char(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    fn next_char(&mut self) -> Option<char> {
        let c = self.peek_char()?;
        self.pos += c.len_utf8();
        Some(c)
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek_char() {
            if c.is_whitespace() {
                self.next_char();
            } else {
                break;
            }
        }
    }

    fn read_number(&mut self) -> Result<f32, ParseError> {
        let start = self.pos;
        while let Some(c) = self.peek_char() {
            if c.is_ascii_digit() || c == '.' {
                self.next_char();
            } else {
                break;
            }
        }
        let s = &self.input[start..self.pos];
        s.parse()
            .map_err(|_| ParseError::InvalidNumber(s.to_string()))
    }

    fn read_ident(&mut self) -> String {
        let start = self.pos;
        while let Some(c) = self.peek_char() {
            if c.is_alphanumeric() || c == '_' {
                self.next_char();
            } else {
                break;
            }
        }
        self.input[start..self.pos].to_string()
    }

    fn next_token(&mut self) -> Result<Token, ParseError> {
        self.skip_whitespace();

        let Some(c) = self.peek_char() else {
            return Ok(Token::Eof);
        };

        match c {
            '+' => {
                self.next_char();
                Ok(Token::Plus)
            }
            '-' => {
                self.next_char();
                Ok(Token::Minus)
            }
            '*' => {
                self.next_char();
                Ok(Token::Star)
            }
            '/' => {
                self.next_char();
                Ok(Token::Slash)
            }
            '^' => {
                self.next_char();
                Ok(Token::Caret)
            }
            '(' => {
                self.next_char();
                Ok(Token::LParen)
            }
            ')' => {
                self.next_char();
                Ok(Token::RParen)
            }
            ',' => {
                self.next_char();
                Ok(Token::Comma)
            }
            '0'..='9' | '.' => Ok(Token::Number(self.read_number()?)),
            'a'..='z' | 'A'..='Z' | '_' => Ok(Token::Ident(self.read_ident())),
            _ => Err(ParseError::UnexpectedChar(c)),
        }
    }
}

// ============================================================================
// AST
// ============================================================================

/// AST node for expressions.
#[derive(Debug, Clone)]
pub enum Ast {
    Num(f32),
    Var(Var),
    BinOp(BinOp, Box<Ast>, Box<Ast>),
    UnaryOp(UnaryOp, Box<Ast>),
    Call(String, Vec<Ast>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Var {
    X,
    Y,
    Z,
    Time,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnaryOp {
    Neg,
}

// ============================================================================
// Parser
// ============================================================================

struct Parser<'a> {
    lexer: Lexer<'a>,
    current: Token,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Result<Self, ParseError> {
        let mut lexer = Lexer::new(input);
        let current = lexer.next_token()?;
        Ok(Self { lexer, current })
    }

    fn advance(&mut self) -> Result<(), ParseError> {
        self.current = self.lexer.next_token()?;
        Ok(())
    }

    fn expect(&mut self, expected: Token) -> Result<(), ParseError> {
        if self.current == expected {
            self.advance()
        } else {
            Err(ParseError::UnexpectedToken(format!("{:?}", self.current)))
        }
    }

    fn parse_expr(&mut self) -> Result<Ast, ParseError> {
        self.parse_add_sub()
    }

    fn parse_add_sub(&mut self) -> Result<Ast, ParseError> {
        let mut left = self.parse_mul_div()?;

        loop {
            match &self.current {
                Token::Plus => {
                    self.advance()?;
                    let right = self.parse_mul_div()?;
                    left = Ast::BinOp(BinOp::Add, Box::new(left), Box::new(right));
                }
                Token::Minus => {
                    self.advance()?;
                    let right = self.parse_mul_div()?;
                    left = Ast::BinOp(BinOp::Sub, Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }

        Ok(left)
    }

    fn parse_mul_div(&mut self) -> Result<Ast, ParseError> {
        let mut left = self.parse_power()?;

        loop {
            match &self.current {
                Token::Star => {
                    self.advance()?;
                    let right = self.parse_power()?;
                    left = Ast::BinOp(BinOp::Mul, Box::new(left), Box::new(right));
                }
                Token::Slash => {
                    self.advance()?;
                    let right = self.parse_power()?;
                    left = Ast::BinOp(BinOp::Div, Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }

        Ok(left)
    }

    fn parse_power(&mut self) -> Result<Ast, ParseError> {
        let base = self.parse_unary()?;

        if self.current == Token::Caret {
            self.advance()?;
            let exp = self.parse_power()?; // Right associative
            Ok(Ast::BinOp(BinOp::Pow, Box::new(base), Box::new(exp)))
        } else {
            Ok(base)
        }
    }

    fn parse_unary(&mut self) -> Result<Ast, ParseError> {
        if self.current == Token::Minus {
            self.advance()?;
            let inner = self.parse_unary()?;
            Ok(Ast::UnaryOp(UnaryOp::Neg, Box::new(inner)))
        } else {
            self.parse_primary()
        }
    }

    fn parse_primary(&mut self) -> Result<Ast, ParseError> {
        match &self.current {
            Token::Number(n) => {
                let n = *n;
                self.advance()?;
                Ok(Ast::Num(n))
            }
            Token::Ident(name) => {
                let name = name.clone();
                self.advance()?;

                // Check if it's a function call
                if self.current == Token::LParen {
                    self.advance()?;
                    let mut args = Vec::new();
                    if self.current != Token::RParen {
                        args.push(self.parse_expr()?);
                        while self.current == Token::Comma {
                            self.advance()?;
                            args.push(self.parse_expr()?);
                        }
                    }
                    self.expect(Token::RParen)?;
                    Ok(Ast::Call(name, args))
                } else {
                    // It's a variable or constant
                    match name.as_str() {
                        "x" => Ok(Ast::Var(Var::X)),
                        "y" => Ok(Ast::Var(Var::Y)),
                        "z" => Ok(Ast::Var(Var::Z)),
                        "time" | "t" => Ok(Ast::Var(Var::Time)),
                        "pi" | "PI" => Ok(Ast::Num(PI)),
                        "e" | "E" => Ok(Ast::Num(std::f32::consts::E)),
                        _ => Err(ParseError::UnknownVariable(name)),
                    }
                }
            }
            Token::LParen => {
                self.advance()?;
                let inner = self.parse_expr()?;
                self.expect(Token::RParen)?;
                Ok(inner)
            }
            Token::Eof => Err(ParseError::UnexpectedEnd),
            _ => Err(ParseError::UnexpectedToken(format!("{:?}", self.current))),
        }
    }
}

// ============================================================================
// Expression
// ============================================================================

/// A compiled expression that can be evaluated.
#[derive(Debug, Clone)]
pub struct Expr {
    ast: Ast,
}

impl Expr {
    /// Parses an expression from a string.
    pub fn parse(input: &str) -> Result<Self, ParseError> {
        let mut parser = Parser::new(input)?;
        let ast = parser.parse_expr()?;
        if parser.current != Token::Eof {
            return Err(ParseError::UnexpectedToken(format!("{:?}", parser.current)));
        }
        Ok(Self { ast })
    }

    /// Evaluates the expression at a 2D point.
    pub fn eval(
        &self,
        pos: Vec2,
        ctx: &EvalContext,
        registry: &FunctionRegistry,
    ) -> Result<f32, EvalError> {
        Self::eval_ast(&self.ast, pos.x, pos.y, 0.0, ctx, registry)
    }

    /// Evaluates the expression at a 3D point.
    pub fn eval3(
        &self,
        x: f32,
        y: f32,
        z: f32,
        ctx: &EvalContext,
        registry: &FunctionRegistry,
    ) -> Result<f32, EvalError> {
        Self::eval_ast(&self.ast, x, y, z, ctx, registry)
    }

    fn eval_ast(
        ast: &Ast,
        x: f32,
        y: f32,
        z: f32,
        ctx: &EvalContext,
        registry: &FunctionRegistry,
    ) -> Result<f32, EvalError> {
        match ast {
            Ast::Num(n) => Ok(*n),
            Ast::Var(v) => Ok(match v {
                Var::X => x,
                Var::Y => y,
                Var::Z => z,
                Var::Time => ctx.time,
            }),
            Ast::BinOp(op, l, r) => {
                let l = Self::eval_ast(l, x, y, z, ctx, registry)?;
                let r = Self::eval_ast(r, x, y, z, ctx, registry)?;
                Ok(match op {
                    BinOp::Add => l + r,
                    BinOp::Sub => l - r,
                    BinOp::Mul => l * r,
                    BinOp::Div => l / r,
                    BinOp::Pow => l.powf(r),
                })
            }
            Ast::UnaryOp(op, inner) => {
                let v = Self::eval_ast(inner, x, y, z, ctx, registry)?;
                Ok(match op {
                    UnaryOp::Neg => -v,
                })
            }
            Ast::Call(name, args) => {
                let func = registry
                    .get(name)
                    .ok_or_else(|| EvalError::UnknownFunction(name.clone()))?;

                if args.len() != func.arg_count() {
                    return Err(EvalError::WrongArgCount {
                        func: name.clone(),
                        expected: func.arg_count(),
                        got: args.len(),
                    });
                }

                let arg_values: Vec<f32> = args
                    .iter()
                    .map(|a| Self::eval_ast(a, x, y, z, ctx, registry))
                    .collect::<Result<_, _>>()?;

                Ok(func.interpret(&arg_values))
            }
        }
    }
}

// ============================================================================
// Field implementation
// ============================================================================

/// An expression bundled with its function registry for use as a Field.
pub struct ExprField {
    expr: Expr,
    registry: FunctionRegistry,
}

impl ExprField {
    /// Creates a new ExprField with the standard function registry.
    pub fn new(expr: Expr) -> Self {
        Self {
            expr,
            registry: std_registry(),
        }
    }

    /// Creates a new ExprField with a custom registry.
    pub fn with_registry(expr: Expr, registry: FunctionRegistry) -> Self {
        Self { expr, registry }
    }

    /// Parses and creates an ExprField with standard functions.
    pub fn parse(input: &str) -> Result<Self, ParseError> {
        Ok(Self::new(Expr::parse(input)?))
    }
}

impl crate::field::Field<Vec2, f32> for ExprField {
    fn sample(&self, input: Vec2, ctx: &EvalContext) -> f32 {
        self.expr.eval(input, ctx, &self.registry).unwrap_or(0.0)
    }
}

impl crate::field::Field<glam::Vec3, f32> for ExprField {
    fn sample(&self, input: glam::Vec3, ctx: &EvalContext) -> f32 {
        self.expr
            .eval3(input.x, input.y, input.z, ctx, &self.registry)
            .unwrap_or(0.0)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn eval(expr: &str, x: f32, y: f32) -> f32 {
        let registry = std_registry();
        let expr = Expr::parse(expr).unwrap();
        let ctx = EvalContext::new();
        expr.eval(Vec2::new(x, y), &ctx, &registry).unwrap()
    }

    #[test]
    fn test_parse_number() {
        assert_eq!(eval("42", 0.0, 0.0), 42.0);
    }

    #[test]
    fn test_parse_float() {
        assert!((eval("1.234", 0.0, 0.0) - 1.234).abs() < 0.001);
    }

    #[test]
    fn test_parse_variable() {
        assert_eq!(eval("x", 5.0, 0.0), 5.0);
        assert_eq!(eval("y", 0.0, 3.0), 3.0);
    }

    #[test]
    fn test_parse_add() {
        assert_eq!(eval("1 + 2", 0.0, 0.0), 3.0);
    }

    #[test]
    fn test_parse_mul() {
        assert_eq!(eval("3 * 4", 0.0, 0.0), 12.0);
    }

    #[test]
    fn test_precedence() {
        assert_eq!(eval("2 + 3 * 4", 0.0, 0.0), 14.0);
    }

    #[test]
    fn test_parentheses() {
        assert_eq!(eval("(2 + 3) * 4", 0.0, 0.0), 20.0);
    }

    #[test]
    fn test_negation() {
        assert_eq!(eval("-5", 0.0, 0.0), -5.0);
    }

    #[test]
    fn test_power() {
        assert_eq!(eval("2 ^ 3", 0.0, 0.0), 8.0);
    }

    #[test]
    fn test_function_sin() {
        assert!(eval("sin(0)", 0.0, 0.0).abs() < 0.001);
    }

    #[test]
    fn test_function_sqrt() {
        assert_eq!(eval("sqrt(16)", 0.0, 0.0), 4.0);
    }

    #[test]
    fn test_function_min_max() {
        assert_eq!(eval("min(3, 7)", 0.0, 0.0), 3.0);
        assert_eq!(eval("max(3, 7)", 0.0, 0.0), 7.0);
    }

    #[test]
    fn test_function_clamp() {
        assert_eq!(eval("clamp(5, 0, 3)", 0.0, 0.0), 3.0);
    }

    #[test]
    fn test_function_lerp() {
        assert_eq!(eval("lerp(0, 10, 0.5)", 0.0, 0.0), 5.0);
    }

    #[test]
    fn test_function_mix() {
        assert_eq!(eval("mix(0, 10, 0.5)", 0.0, 0.0), 5.0);
    }

    #[test]
    fn test_complex_expression() {
        let v = eval("sin(x * 3.14) + y / 2", 0.5, 4.0);
        // sin(0.5 * 3.14) + 4.0 / 2 = sin(1.57) + 2 ≈ 1 + 2 = 3
        assert!((v - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_time_variable() {
        let registry = std_registry();
        let expr = Expr::parse("time").unwrap();
        let ctx = EvalContext::new().with_time(5.0);
        assert_eq!(expr.eval(Vec2::ZERO, &ctx, &registry).unwrap(), 5.0);
    }

    #[test]
    fn test_pi_constant() {
        assert!((eval("pi", 0.0, 0.0) - PI).abs() < 0.001);
    }

    #[test]
    fn test_noise_function() {
        let v = eval("noise(x, y)", 0.5, 0.5);
        assert!((0.0..=1.0).contains(&v));
    }

    #[test]
    fn test_perlin_alias() {
        let v = eval("perlin(x, y)", 0.5, 0.5);
        assert!((0.0..=1.0).contains(&v));
    }

    #[test]
    fn test_unknown_function() {
        let registry = std_registry();
        let expr = Expr::parse("unknown(1)").unwrap();
        let ctx = EvalContext::new();
        let result = expr.eval(Vec2::ZERO, &ctx, &registry);
        assert!(matches!(result, Err(EvalError::UnknownFunction(_))));
    }

    #[test]
    fn test_wrong_arg_count() {
        let registry = std_registry();
        let expr = Expr::parse("sin(1, 2)").unwrap();
        let ctx = EvalContext::new();
        let result = expr.eval(Vec2::ZERO, &ctx, &registry);
        assert!(matches!(result, Err(EvalError::WrongArgCount { .. })));
    }

    #[test]
    fn test_custom_function() {
        // Register a custom function
        struct Double;
        impl ExprFn for Double {
            fn name(&self) -> &str {
                "double"
            }
            fn arg_count(&self) -> usize {
                1
            }
            fn interpret(&self, args: &[f32]) -> f32 {
                args[0] * 2.0
            }
        }

        let mut registry = std_registry();
        registry.register(Double);

        let expr = Expr::parse("double(5)").unwrap();
        let ctx = EvalContext::new();
        assert_eq!(expr.eval(Vec2::ZERO, &ctx, &registry).unwrap(), 10.0);
    }

    #[test]
    fn test_expr_field() {
        use crate::field::Field;

        let field = ExprField::parse("x + y").unwrap();
        let ctx = EvalContext::new();
        let v: f32 = field.sample(Vec2::new(3.0, 4.0), &ctx);
        assert_eq!(v, 7.0);
    }
}
