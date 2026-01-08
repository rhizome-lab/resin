//! SVG export for 2D vector graphics.
//!
//! Provides functionality to export paths and shapes to SVG format.
//!
//! # Example
//!
//! ```
//! use resin_vector::{Path, PathBuilder, circle};
//! use resin_vector::svg::{SvgDocument, SvgStyle};
//!
//! let path = circle(glam::Vec2::new(50.0, 50.0), 40.0);
//! let mut doc = SvgDocument::new(100.0, 100.0);
//! doc.add_path(&path, SvgStyle::stroke("#ff0000", 2.0));
//! let svg_string = doc.to_string();
//! ```

use crate::{Path, PathCommand};
use glam::Vec2;
use std::fmt::Write;

/// Style for SVG elements.
#[derive(Debug, Clone)]
pub struct SvgStyle {
    /// Fill color (None for no fill).
    pub fill: Option<String>,
    /// Stroke color (None for no stroke).
    pub stroke: Option<String>,
    /// Stroke width.
    pub stroke_width: f32,
    /// Stroke line cap style.
    pub stroke_linecap: StrokeLinecap,
    /// Stroke line join style.
    pub stroke_linejoin: StrokeLinejoin,
    /// Fill opacity (0.0 to 1.0).
    pub fill_opacity: f32,
    /// Stroke opacity (0.0 to 1.0).
    pub stroke_opacity: f32,
    /// Stroke dash array.
    pub stroke_dasharray: Option<Vec<f32>>,
    /// Stroke dash offset.
    pub stroke_dashoffset: f32,
}

/// Stroke line cap style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StrokeLinecap {
    /// Flat cap at the end of the line.
    #[default]
    Butt,
    /// Rounded cap.
    Round,
    /// Square cap extending past the end.
    Square,
}

/// Stroke line join style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StrokeLinejoin {
    /// Sharp corner.
    #[default]
    Miter,
    /// Rounded corner.
    Round,
    /// Beveled corner.
    Bevel,
}

impl Default for SvgStyle {
    fn default() -> Self {
        Self {
            fill: Some("black".to_string()),
            stroke: None,
            stroke_width: 1.0,
            stroke_linecap: StrokeLinecap::default(),
            stroke_linejoin: StrokeLinejoin::default(),
            fill_opacity: 1.0,
            stroke_opacity: 1.0,
            stroke_dasharray: None,
            stroke_dashoffset: 0.0,
        }
    }
}

impl SvgStyle {
    /// Creates a style with only fill.
    pub fn fill(color: impl Into<String>) -> Self {
        Self {
            fill: Some(color.into()),
            stroke: None,
            ..Default::default()
        }
    }

    /// Creates a style with only stroke.
    pub fn stroke(color: impl Into<String>, width: f32) -> Self {
        Self {
            fill: None,
            stroke: Some(color.into()),
            stroke_width: width,
            ..Default::default()
        }
    }

    /// Creates a style with both fill and stroke.
    pub fn fill_stroke(
        fill_color: impl Into<String>,
        stroke_color: impl Into<String>,
        stroke_width: f32,
    ) -> Self {
        Self {
            fill: Some(fill_color.into()),
            stroke: Some(stroke_color.into()),
            stroke_width,
            ..Default::default()
        }
    }

    /// Sets the stroke linecap.
    pub fn with_linecap(mut self, linecap: StrokeLinecap) -> Self {
        self.stroke_linecap = linecap;
        self
    }

    /// Sets the stroke linejoin.
    pub fn with_linejoin(mut self, linejoin: StrokeLinejoin) -> Self {
        self.stroke_linejoin = linejoin;
        self
    }

    /// Sets the fill opacity.
    pub fn with_fill_opacity(mut self, opacity: f32) -> Self {
        self.fill_opacity = opacity;
        self
    }

    /// Sets the stroke opacity.
    pub fn with_stroke_opacity(mut self, opacity: f32) -> Self {
        self.stroke_opacity = opacity;
        self
    }

    /// Sets the stroke dash array.
    pub fn with_dash(mut self, dasharray: Vec<f32>) -> Self {
        self.stroke_dasharray = Some(dasharray);
        self
    }

    /// Sets the stroke dash offset.
    pub fn with_dash_offset(mut self, offset: f32) -> Self {
        self.stroke_dashoffset = offset;
        self
    }

    /// Converts the style to SVG attribute string.
    fn to_attributes(&self) -> String {
        let mut attrs = String::new();

        match &self.fill {
            Some(color) => write!(&mut attrs, "fill=\"{}\" ", color).unwrap(),
            None => attrs.push_str("fill=\"none\" "),
        }

        if self.fill_opacity < 1.0 {
            write!(&mut attrs, "fill-opacity=\"{:.3}\" ", self.fill_opacity).unwrap();
        }

        if let Some(color) = &self.stroke {
            write!(&mut attrs, "stroke=\"{}\" ", color).unwrap();
            write!(&mut attrs, "stroke-width=\"{:.3}\" ", self.stroke_width).unwrap();

            if self.stroke_opacity < 1.0 {
                write!(&mut attrs, "stroke-opacity=\"{:.3}\" ", self.stroke_opacity).unwrap();
            }

            match self.stroke_linecap {
                StrokeLinecap::Round => attrs.push_str("stroke-linecap=\"round\" "),
                StrokeLinecap::Square => attrs.push_str("stroke-linecap=\"square\" "),
                StrokeLinecap::Butt => {}
            }

            match self.stroke_linejoin {
                StrokeLinejoin::Round => attrs.push_str("stroke-linejoin=\"round\" "),
                StrokeLinejoin::Bevel => attrs.push_str("stroke-linejoin=\"bevel\" "),
                StrokeLinejoin::Miter => {}
            }

            if let Some(dasharray) = &self.stroke_dasharray {
                let dashes: Vec<String> = dasharray.iter().map(|d| format!("{:.3}", d)).collect();
                write!(&mut attrs, "stroke-dasharray=\"{}\" ", dashes.join(",")).unwrap();
            }

            if self.stroke_dashoffset != 0.0 {
                write!(
                    &mut attrs,
                    "stroke-dashoffset=\"{:.3}\" ",
                    self.stroke_dashoffset
                )
                .unwrap();
            }
        }

        attrs.trim_end().to_string()
    }
}

/// An SVG document containing paths and shapes.
#[derive(Debug, Clone)]
pub struct SvgDocument {
    /// Document width.
    pub width: f32,
    /// Document height.
    pub height: f32,
    /// ViewBox (min_x, min_y, width, height).
    pub viewbox: Option<ViewBox>,
    /// Elements in the document.
    elements: Vec<SvgElement>,
}

/// SVG viewBox definition.
#[derive(Debug, Clone, Copy)]
pub struct ViewBox {
    /// Minimum X coordinate.
    pub min_x: f32,
    /// Minimum Y coordinate.
    pub min_y: f32,
    /// Width of the viewBox.
    pub width: f32,
    /// Height of the viewBox.
    pub height: f32,
}

/// An element in an SVG document.
#[derive(Debug, Clone)]
enum SvgElement {
    Path {
        data: String,
        style: SvgStyle,
    },
    Circle {
        cx: f32,
        cy: f32,
        r: f32,
        style: SvgStyle,
    },
    Rect {
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        rx: f32,
        ry: f32,
        style: SvgStyle,
    },
    Line {
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        style: SvgStyle,
    },
    Group {
        elements: Vec<SvgElement>,
        transform: Option<String>,
    },
}

impl SvgDocument {
    /// Creates a new SVG document with the given dimensions.
    pub fn new(width: f32, height: f32) -> Self {
        Self {
            width,
            height,
            viewbox: None,
            elements: Vec::new(),
        }
    }

    /// Creates a new SVG document with a viewBox.
    pub fn with_viewbox(width: f32, height: f32, viewbox: ViewBox) -> Self {
        Self {
            width,
            height,
            viewbox: Some(viewbox),
            elements: Vec::new(),
        }
    }

    /// Adds a path to the document.
    pub fn add_path(&mut self, path: &Path, style: SvgStyle) {
        let data = path_to_svg_data(path);
        self.elements.push(SvgElement::Path { data, style });
    }

    /// Adds a circle to the document.
    pub fn add_circle(&mut self, center: Vec2, radius: f32, style: SvgStyle) {
        self.elements.push(SvgElement::Circle {
            cx: center.x,
            cy: center.y,
            r: radius,
            style,
        });
    }

    /// Adds a rectangle to the document.
    pub fn add_rect(&mut self, pos: Vec2, size: Vec2, style: SvgStyle) {
        self.elements.push(SvgElement::Rect {
            x: pos.x,
            y: pos.y,
            width: size.x,
            height: size.y,
            rx: 0.0,
            ry: 0.0,
            style,
        });
    }

    /// Adds a rounded rectangle to the document.
    pub fn add_rounded_rect(&mut self, pos: Vec2, size: Vec2, radius: f32, style: SvgStyle) {
        self.elements.push(SvgElement::Rect {
            x: pos.x,
            y: pos.y,
            width: size.x,
            height: size.y,
            rx: radius,
            ry: radius,
            style,
        });
    }

    /// Adds a line to the document.
    pub fn add_line(&mut self, from: Vec2, to: Vec2, style: SvgStyle) {
        self.elements.push(SvgElement::Line {
            x1: from.x,
            y1: from.y,
            x2: to.x,
            y2: to.y,
            style,
        });
    }

    /// Converts the document to an SVG string.
    pub fn to_svg_string(&self) -> String {
        let mut svg = String::new();

        // XML declaration
        svg.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");

        // SVG element
        write!(
            &mut svg,
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{:.3}\" height=\"{:.3}\"",
            self.width, self.height
        )
        .unwrap();

        if let Some(vb) = &self.viewbox {
            write!(
                &mut svg,
                " viewBox=\"{:.3} {:.3} {:.3} {:.3}\"",
                vb.min_x, vb.min_y, vb.width, vb.height
            )
            .unwrap();
        }

        svg.push_str(">\n");

        // Elements
        for element in &self.elements {
            write_element(&mut svg, element, 1);
        }

        svg.push_str("</svg>\n");
        svg
    }
}

impl std::fmt::Display for SvgDocument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_svg_string())
    }
}

/// Writes an SVG element to a string with indentation.
fn write_element(svg: &mut String, element: &SvgElement, indent: usize) {
    let indent_str = "  ".repeat(indent);

    match element {
        SvgElement::Path { data, style } => {
            write!(
                svg,
                "{}<path d=\"{}\" {}/>\n",
                indent_str,
                data,
                style.to_attributes()
            )
            .unwrap();
        }
        SvgElement::Circle { cx, cy, r, style } => {
            write!(
                svg,
                "{}<circle cx=\"{:.3}\" cy=\"{:.3}\" r=\"{:.3}\" {}/>\n",
                indent_str,
                cx,
                cy,
                r,
                style.to_attributes()
            )
            .unwrap();
        }
        SvgElement::Rect {
            x,
            y,
            width,
            height,
            rx,
            ry,
            style,
        } => {
            write!(
                svg,
                "{}<rect x=\"{:.3}\" y=\"{:.3}\" width=\"{:.3}\" height=\"{:.3}\"",
                indent_str, x, y, width, height
            )
            .unwrap();
            if *rx > 0.0 {
                write!(svg, " rx=\"{:.3}\"", rx).unwrap();
            }
            if *ry > 0.0 {
                write!(svg, " ry=\"{:.3}\"", ry).unwrap();
            }
            write!(svg, " {}/>\n", style.to_attributes()).unwrap();
        }
        SvgElement::Line {
            x1,
            y1,
            x2,
            y2,
            style,
        } => {
            write!(
                svg,
                "{}<line x1=\"{:.3}\" y1=\"{:.3}\" x2=\"{:.3}\" y2=\"{:.3}\" {}/>\n",
                indent_str,
                x1,
                y1,
                x2,
                y2,
                style.to_attributes()
            )
            .unwrap();
        }
        SvgElement::Group {
            elements,
            transform,
        } => {
            if let Some(t) = transform {
                write!(svg, "{}<g transform=\"{}\">\n", indent_str, t).unwrap();
            } else {
                write!(svg, "{}<g>\n", indent_str).unwrap();
            }
            for child in elements {
                write_element(svg, child, indent + 1);
            }
            write!(svg, "{}</g>\n", indent_str).unwrap();
        }
    }
}

/// Converts a Path to SVG path data string.
pub fn path_to_svg_data(path: &Path) -> String {
    let mut data = String::new();

    for cmd in path.commands() {
        match cmd {
            PathCommand::MoveTo(p) => {
                write!(&mut data, "M{:.3},{:.3} ", p.x, p.y).unwrap();
            }
            PathCommand::LineTo(p) => {
                write!(&mut data, "L{:.3},{:.3} ", p.x, p.y).unwrap();
            }
            PathCommand::QuadTo { control, to } => {
                write!(
                    &mut data,
                    "Q{:.3},{:.3} {:.3},{:.3} ",
                    control.x, control.y, to.x, to.y
                )
                .unwrap();
            }
            PathCommand::CubicTo {
                control1,
                control2,
                to,
            } => {
                write!(
                    &mut data,
                    "C{:.3},{:.3} {:.3},{:.3} {:.3},{:.3} ",
                    control1.x, control1.y, control2.x, control2.y, to.x, to.y
                )
                .unwrap();
            }
            PathCommand::Close => {
                data.push_str("Z ");
            }
        }
    }

    data.trim_end().to_string()
}

/// Converts a Path to a complete SVG string (simple helper).
pub fn path_to_svg(path: &Path, width: f32, height: f32, style: SvgStyle) -> String {
    let mut doc = SvgDocument::new(width, height);
    doc.add_path(path, style);
    doc.to_svg_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{circle, line, rect};

    #[test]
    fn test_path_to_svg_data() {
        let path = line(Vec2::ZERO, Vec2::new(100.0, 100.0));
        let data = path_to_svg_data(&path);
        assert!(data.contains("M0.000,0.000"));
        assert!(data.contains("L100.000,100.000"));
    }

    #[test]
    fn test_svg_document_basic() {
        let mut doc = SvgDocument::new(200.0, 200.0);
        doc.add_circle(Vec2::new(100.0, 100.0), 50.0, SvgStyle::fill("red"));

        let svg = doc.to_svg_string();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("circle"));
        assert!(svg.contains("fill=\"red\""));
    }

    #[test]
    fn test_svg_style_stroke() {
        let style = SvgStyle::stroke("#0000ff", 2.0);
        let attrs = style.to_attributes();
        assert!(attrs.contains("stroke=\"#0000ff\""));
        assert!(attrs.contains("stroke-width=\"2.000\""));
        assert!(attrs.contains("fill=\"none\""));
    }

    #[test]
    fn test_svg_style_fill_stroke() {
        let style = SvgStyle::fill_stroke("white", "black", 1.0);
        let attrs = style.to_attributes();
        assert!(attrs.contains("fill=\"white\""));
        assert!(attrs.contains("stroke=\"black\""));
    }

    #[test]
    fn test_svg_path() {
        let path = circle(Vec2::new(50.0, 50.0), 40.0);
        let mut doc = SvgDocument::new(100.0, 100.0);
        doc.add_path(&path, SvgStyle::stroke("blue", 2.0));

        let svg = doc.to_svg_string();
        assert!(svg.contains("<path"));
        assert!(svg.contains("d=\""));
    }

    #[test]
    fn test_svg_rect() {
        let mut doc = SvgDocument::new(100.0, 100.0);
        doc.add_rect(
            Vec2::new(10.0, 10.0),
            Vec2::new(80.0, 80.0),
            SvgStyle::fill("green"),
        );

        let svg = doc.to_svg_string();
        assert!(svg.contains("<rect"));
        assert!(svg.contains("x=\"10.000\""));
        assert!(svg.contains("width=\"80.000\""));
    }

    #[test]
    fn test_svg_rounded_rect() {
        let mut doc = SvgDocument::new(100.0, 100.0);
        doc.add_rounded_rect(
            Vec2::new(10.0, 10.0),
            Vec2::new(80.0, 80.0),
            5.0,
            SvgStyle::fill("blue"),
        );

        let svg = doc.to_svg_string();
        assert!(svg.contains("rx=\"5.000\""));
    }

    #[test]
    fn test_svg_line() {
        let mut doc = SvgDocument::new(100.0, 100.0);
        doc.add_line(
            Vec2::ZERO,
            Vec2::new(100.0, 100.0),
            SvgStyle::stroke("red", 1.0),
        );

        let svg = doc.to_svg_string();
        assert!(svg.contains("<line"));
        assert!(svg.contains("x1=\"0.000\""));
        assert!(svg.contains("x2=\"100.000\""));
    }

    #[test]
    fn test_svg_dashed_stroke() {
        let style = SvgStyle::stroke("black", 1.0).with_dash(vec![5.0, 3.0]);
        let attrs = style.to_attributes();
        assert!(attrs.contains("stroke-dasharray=\"5.000,3.000\""));
    }

    #[test]
    fn test_svg_viewbox() {
        let doc = SvgDocument::with_viewbox(
            100.0,
            100.0,
            ViewBox {
                min_x: -50.0,
                min_y: -50.0,
                width: 100.0,
                height: 100.0,
            },
        );

        let svg = doc.to_svg_string();
        assert!(svg.contains("viewBox=\"-50.000 -50.000 100.000 100.000\""));
    }

    #[test]
    fn test_path_to_svg_helper() {
        let path = rect(Vec2::new(10.0, 10.0), Vec2::new(80.0, 80.0));
        let svg = path_to_svg(&path, 100.0, 100.0, SvgStyle::fill("red"));

        assert!(svg.contains("<svg"));
        assert!(svg.contains("<path"));
        assert!(svg.contains("</svg>"));
    }
}
