use std::env;
use std::f32::consts::PI;
use std::fs::create_dir_all;

use anyhow::{bail, Result};
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{NdArray, Wgpu};
use burn::module::Module;
use burn::prelude::Backend;
use burn::record::{DefaultFileRecorder, FullPrecisionSettings};

use plotpy::{Curve, Plot};
use resvg;
use resvg::usvg::Tree;
use svg::node::element::path::{Command, Data, Parameters, Position};
use svg::node::element::{Circle, Definitions, Group, Line, LinearGradient, Path, Rectangle, Stop};
use svg::{Document, Node};
use train_quadcopter::data::State;
use train_quadcopter::model::ModelConfig;
use train_quadcopter::simulate::{simulate, SimSample};

fn main() -> Result<()> {
    println!("Trained Model Visualization");

    // Load the model weights from the trained file.
    if env::var("INFERENCE_USE_NDARRAY").is_ok() {
        let device = NdArrayDevice::default();
        run_inference::<NdArray>(device)
    } else {
        let device = WgpuDevice::default();
        run_inference::<Wgpu>(device)
    }
}

fn run_inference<B: Backend>(device: B::Device) -> Result<()> {
    let file_path = "./burn-training-artifacts/model.mpk";
    let model = ModelConfig::new().init::<B>(&device);
    let model = model.load_file(
        file_path,
        &DefaultFileRecorder::<FullPrecisionSettings>::new(),
        &device,
    )?;
    println!("Loaded model");

    // Try simulating the model.
    let start = State::new(-10.0, 20.0, 0.0, 0.0, 0.0);
    // let start = State::new(5.0, 20.0, 0.0, 0.0, 0.0);
    // let start = State::new(-0.5, 20.0, 0.0, 0.0, 0.0);
    // let start = State::new(10.0, 10.0, 0.0, 0.0, 0.0);
    // let start = State::new(-5.0, 10.0, 0.0, 0.0, 0.0);
    let samples = simulate(device, &model, start, 8.0, 1.0 / 60.0);
    println!("Simulated trajectory");

    animate(&samples, "animation")?;
    println!("Animated trajectory");

    Ok(())
}

pub fn animate(samples: &Vec<SimSample>, out_dir: &str) -> Result<()> {
    let px_width = 1080;
    let px_height = 1080;
    let min_y = -2.0;
    let height = 25.0;
    let margin = 0.6;
    let width = (px_width as f32) / (px_height as f32) * height;
    let min_x = -width / 2.0;
    let view_box = (min_x, min_y, width, height);

    create_dir_all(out_dir)?;

    let mut trajectory: Vec<(f32, f32)> = Vec::new();
    let mut frame_number: usize = 0;
    for sample in samples {
        println!("Rendering frame: {}", frame_number);

        trajectory.push((sample.state.x, sample.state.z));

        let mut traj_data = Data::new();
        let mut traj_iter = trajectory.iter();
        let start = traj_iter.next().unwrap();
        append_move_to(&mut traj_data, start);
        for position in trajectory.iter() {
            append_line_to(&mut traj_data, position);
        }
        let quadcopter_path = Path::new()
            .set("d", traj_data)
            .set("fill", "none")
            .set("stroke", "rgb(56,155,193)")
            .set("stroke-width", 0.08)
            .set("stroke-dasharray", "0.2 0.1");

        let quadcopter = draw_quadcopter(sample.state.x, sample.state.z, sample.state.theta)
            .set("stroke", "rgb(95%,95%,95%)")
            .set("stroke-width", 8.0)
            .set("fill", "rgb(65%,65%,65%)");

        let t_angle = -sample.state.theta + (PI / 2.0);
        let thrust_scale = 0.15;
        let arrow_width = 0.2;
        let arrow_length = 0.2;
        let thrust_x = thrust_scale * sample.controls.thrust;
        let thrust_vec_data = Data::new()
            .move_to((0, 0))
            .line_to((thrust_x, 0))
            .move_to((thrust_x - arrow_length, arrow_width / 2.0))
            .line_to((thrust_x, 0))
            .line_to((thrust_x - arrow_length, -arrow_width / 2.0));
        let thrust_vec_path = Path::new()
            .set("d", thrust_vec_data)
            .set("fill", "none")
            .set("stroke", "rgb(85%,10%,10%)")
            .set("stroke-width", 0.1)
            .set(
                "transform",
                format!(
                    "translate({},{}) rotate({})",
                    sample.state.x,
                    sample.state.z,
                    t_angle * 180.0 / PI,
                ),
            );

        let viewport_group = create_viewport_group(view_box)
            .add(draw_background(view_box, margin))
            .add(quadcopter_path)
            .add(quadcopter)
            .add(thrust_vec_path);

        let document = Document::new()
            .set("viewBox", view_box)
            .set("width", px_width)
            .set("height", px_height)
            .add(viewport_group);

        let out_file_name = format!("{}/{:05}.png", out_dir, frame_number);
        save_svg(&out_file_name, &document)?;

        // Uncomment this section below if you need to inspect the raw SVG
        // file. (Useful for debugging.)
        /*
        if frame_number == 0 {
            svg::save("animate-test.svg", &document)?;
        }
        */

        frame_number += 1;
    }
    Ok(())
}

fn append_move_to(data: &mut Data, xy: &(f32, f32)) {
    data.append(Command::Move(Position::Absolute, Parameters::from(*xy)));
}

fn append_line_to(data: &mut Data, xy: &(f32, f32)) {
    data.append(Command::Line(Position::Absolute, Parameters::from(*xy)));
}

fn create_viewport_group(view_box: (f32, f32, f32, f32)) -> Group {
    let (min_x, min_y, _width, height) = view_box;
    Group::new().set(
        "transform",
        format!(
            "translate({},{}) scale(1,-1) translate({},{})",
            min_x,
            min_y,
            -min_x,
            -height - min_y
        ),
    )
}

/// Draw the background of the animation.
fn draw_background(view_box: (f32, f32, f32, f32), margin: f32) -> impl Node {
    let major_line_spacing: f32 = 10.0;
    let minor_line_spacing: f32 = 1.0;
    let rounding_radius: f32 = 0.2;
    let (min_x, min_y, width, height) = view_box;
    let minor_line_color = "rgb(15%,15%,15%)";
    let line_color = "rgb(25%,25%,25%)";

    // Background solid color
    let bg_rect = Rectangle::new()
        .set("x", min_x)
        .set("y", min_y)
        .set("width", width)
        .set("height", height)
        .set("fill", "rgb(9,12,30)")
        .set("stroke", "none");

    // Background rectangle with gradient
    let bg_grad = LinearGradient::new()
        .set("id", "bg_grad")
        .set("x1", "0")
        .set("x2", "0")
        .set("y1", "0")
        .set("y2", "1")
        .add(
            Stop::new()
                .set("offset", "0%")
                .set("stop-color", "rgb(7,11,24)"),
        )
        .add(
            Stop::new()
                .set("offset", "100%")
                .set("stop-color", "rgb(7,38,62)"),
        );
    let bg_grad_defs = Definitions::new().add(bg_grad);
    let fg_rect = Rectangle::new()
        .set("x", min_x + margin)
        .set("y", min_y + margin)
        .set("width", width - 2.0 * margin)
        .set("height", height - 2.0 * margin)
        .set("rx", rounding_radius)
        .set("ry", rounding_radius)
        .set("stroke", line_color)
        .set("stroke-width", "0.05px")
        .set("fill", "url('#bg_grad')");

    let mut group = Group::new().add(bg_grad_defs).add(bg_rect).add(fg_rect);

    // Lines
    let yi_start = (((min_y + margin) as f32) / minor_line_spacing).floor() as i32;
    let yi_end = (((min_y + height - margin) as f32) / minor_line_spacing).ceil() as i32;
    for yi in yi_start..=yi_end {
        let y = yi as f32 * minor_line_spacing;
        group.append(
            Line::new()
                .set("x1", min_x + margin)
                .set("y1", y)
                .set("x2", min_x + width - margin)
                .set("y2", y)
                .set("stroke", minor_line_color)
                .set("stroke-width", "0.02px"),
        )
    }
    let xi_start = (((min_x + margin) as f32) / minor_line_spacing).floor() as i32;
    let xi_end = (((min_x + width - margin) as f32) / minor_line_spacing).ceil() as i32;
    for xi in xi_start..=xi_end {
        let x = xi as f32 * minor_line_spacing;
        group.append(
            Line::new()
                .set("x1", x)
                .set("y1", min_y + margin)
                .set("x2", x)
                .set("y2", min_y + width - margin)
                .set("stroke", minor_line_color)
                .set("stroke-width", "0.02px"),
        )
    }
    let yi_start = (((min_y + margin) as f32) / major_line_spacing).floor() as i32;
    let yi_end = (((min_y + height - margin) as f32) / major_line_spacing).ceil() as i32;
    for yi in yi_start..=yi_end {
        let y = yi as f32 * major_line_spacing;
        group.append(
            Line::new()
                .set("x1", min_x + margin)
                .set("y1", y)
                .set("x2", min_x + width - margin)
                .set("y2", y)
                .set("stroke", line_color)
                .set("stroke-width", "0.04px"),
        )
    }
    let xi_start = (((min_x + margin) as f32) / major_line_spacing).floor() as i32;
    let xi_end = (((min_x + width - margin) as f32) / major_line_spacing).ceil() as i32;
    for xi in xi_start..=xi_end {
        let x = xi as f32 * major_line_spacing;
        group.append(
            Line::new()
                .set("x1", x)
                .set("y1", min_y + margin)
                .set("x2", x)
                .set("y2", min_y + width - margin)
                .set("stroke", line_color)
                .set("stroke-width", "0.04px"),
        )
    }
    // Bright line for the ground
    group.append(
        Line::new()
            .set("x1", min_x + margin)
            .set("y1", 0)
            .set("x2", min_x + width - margin)
            .set("y2", 0)
            .set("stroke", "rgb(90%,90%,90%)")
            .set("stroke-width", "0.06px"),
    );

    // Target
    group.append(
        Circle::new()
            .set("cx", 0)
            .set("cy", 0.1)
            .set("r", 0.15)
            .set("stroke", "none")
            .set("fill", "rgb(90%,90%,90%)"),
    );
    group.append(
        Circle::new()
            .set("cx", 0)
            .set("cy", 0.1)
            .set("r", 0.4)
            .set("fill", "none")
            .set("stroke", "rgb(184,12,20)")
            .set("stroke-width", "0.05px"),
    );
    group.append(
        Circle::new()
            .set("cx", 0)
            .set("cy", 0.1)
            .set("r", 0.6)
            .set("fill", "none")
            .set("stroke", "rgb(184,12,20)")
            .set("stroke-width", "0.04px"),
    );

    group
}

/// Draw the quadcopter at the given coordinates with the specified rotation.
fn draw_quadcopter(x: f32, y: f32, theta: f32) -> Path {
    let path_data = Data::parse(QUADCOPTER_OUTLINE_SVG).unwrap();
    let path = Path::new()
        .set("d", path_data)
        .set("fill", "black")
        .set("stroke", "black")
        .set("stroke-width", 1)
        .set(
            "transform",
            format!(
                "translate({},{}) scale(0.005, 0.005) rotate({}) rotate(180)",
                x,
                y,
                -theta * 180.0 / PI
            ),
        );
    path
}

/// Path to draw a quadcopter outline. This was created in Inkscape and then
/// the values rounded. The units are mm. The quadcopter it draws is about
/// 560mm wide and centred at the origin.
const QUADCOPTER_OUTLINE_SVG: &str = r#"
    M -160 20 
    L -222 20 
    C -228 20 -230 17 -230 12 
    L -230 -2 
    C -230 -7 -228 -10 -222 -10 
    L -207 -10 L -207 -27 
    C -207 -32 -203 -36 -198 -37 
    C -200 -39 -201 -41 -202 -44 
    C -248 -38 -281 -36 -281 -39 
    C -282 -41 -250 -49 -204 -56 
    C -203 -66 -194 -74 -183 -74 
    C -174 -74 -167 -69 -163 -61 
    C -118 -67 -85 -69 -84 -66 
    C -84 -63 -116 -56 -162 -49 
    C -163 -45 -165 -41 -168 -37 
    C -163 -36 -159 -32 -159 -27 
    L -159 -10 
    C -105 -10 -105 -40 0 -40 
    C 105 -40 105 -10 159 -10 
    L 159 -27 
    C 159 -32 163 -36 168 -37 
    C 165 -41 163 -45 162 -49 
    C 116 -56 84 -63 84 -66 
    C 85 -69 118 -67 163 -61 
    C 167 -69 174 -74 183 -74 
    C 194 -74 203 -66 204 -56 
    C 250 -49 282 -41 281 -39 
    C 281 -36 248 -38 202 -44 
    C 201 -41 200 -39 198 -37 
    C 203 -36 207 -32 207 -27 
    L 207 -10 
    L 222 -10 
    C 228 -10 230 -7 230 -2 
    L 230 12 
    C 230 17 228 20 222 20 
    L 160 20 
    C 105 20 105 50 0 50 
    C -105 50 -105 20 -160 20 
    Z
    "#;

/// Save an SVG `Document` to the provided file name.
///
/// This uses `resvg` under the hood.
pub fn save_svg(out_file_name: &str, document: &Document) -> Result<()> {
    let svg_string = document.to_string();
    let svg_options = resvg::usvg::Options::default();
    let svg_tree = Tree::from_str(&svg_string, &svg_options)?;

    let pixmap_size = svg_tree.size().to_int_size();
    let width = pixmap_size.width();
    let height = pixmap_size.height();

    let mut pixmap = tiny_skia::Pixmap::new(width, height).unwrap();
    let transform = tiny_skia::Transform::default();
    resvg::render(&svg_tree, transform, &mut pixmap.as_mut());

    pixmap.save_png(out_file_name)?;

    Ok(())
}

pub fn plot_xz(samples: &Vec<SimSample>) -> Result<()> {
    let xs: Vec<f32> = samples.iter().map(|s| s.state.x).collect();
    let zs: Vec<f32> = samples.iter().map(|s| s.state.z).collect();

    let mut plot = Plot::new();
    plot.set_label_x("Horizontal (m)")
        .set_label_y("Vertical (m)")
        .set_figure_size_inches(8.0, 6.0)
        .set_equal_axes(true);

    let mut curve = Curve::new();
    curve
        .set_line_color("#000000")
        .set_line_style("-")
        .set_line_width(1.0);
    curve.draw(&xs, &zs);

    plot.add(&curve);

    let out_file_name = "sim-test.svg";
    match plot.save(out_file_name) {
        Ok(()) => Ok(()),
        Err(message) => bail!(message),
    }
}
