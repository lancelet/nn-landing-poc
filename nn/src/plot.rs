use anyhow::{bail, Result};
use ndarray::Array2;
use palette::{LinSrgb, Mix, Srgb};
use plotpy::{Curve, Plot};
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Plot multiple trajectories.
///
/// The trajectories are all colored slightly differently.
///
/// The (x,z) coordinates of the trajectory are assumed to be in (zero-based)
/// columns at indices 1 and 2 respectively.
pub fn plot_trajectories<'a, I>(out_file_name: &str, trajectories: I) -> Result<()>
where
    I: IntoIterator<Item = &'a Array2<f32>>,
{
    let mut plot = Plot::new();
    plot.set_label_x("Horizontal (m)")
        .set_label_y("Vertical (m)")
        .set_figure_size_inches(8.0, 6.0)
        .set_equal_axes(true);

    let mut rng = StdRng::from_seed([42; 32]);
    let purple = LinSrgb::new(0.3, 0.0, 0.7);
    let black = LinSrgb::new(0.0, 0.0, 0.0);

    for trajectory in trajectories {
        let xs: Vec<f32> = trajectory.column(1).iter().map(|&x| x).collect();
        let zs: Vec<f32> = trajectory.column(2).iter().map(|&z| z).collect();

        // Generate a random color in the purple-to-black range
        let mix_amount = rng.gen_range(0.0f32..0.7f32);
        let curve_color: Srgb = purple.mix(black, mix_amount).into();
        let hex_code = color_to_hex(&curve_color);

        let mut curve = Curve::new();
        curve
            .set_line_color(&hex_code)
            .set_line_style("-")
            .set_line_alpha(0.4)
            .set_line_width(1.2);
        curve.draw(&xs, &zs);

        plot.add(&curve);
    }

    match plot.save(out_file_name) {
        Ok(()) => Ok(()),
        Err(message) => bail!(message),
    }
}

/// Convert an SRGB color to a hex string.
fn color_to_hex(color: &Srgb) -> String {
    let u8_color = color.into_format::<u8>();
    let (r, g, b) = (u8_color.red, u8_color.green, u8_color.blue);
    format!("#{:02X}{:02X}{:02X}", r, g, b)
}
