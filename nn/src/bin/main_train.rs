use train_quadcopter::training;

use anyhow::Result;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::backend::Autodiff;

fn main() -> Result<()> {
    println!("Quadcopter Training");

    let device = WgpuDevice::default();
    training::run::<Autodiff<Wgpu>>(device)
}
