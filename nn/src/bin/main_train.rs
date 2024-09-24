use std::env;

use train_quadcopter::training;

use anyhow::Result;
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::backend::Autodiff;

fn main() -> Result<()> {
    println!("Quadcopter Training");

    if env::var("TRAIN_USE_NDARRAY").is_ok() {
        let device = NdArrayDevice::default();
        training::run::<Autodiff<NdArray>>(device)
    } else {
        let device = WgpuDevice::default();
        training::run::<Autodiff<Wgpu>>(device)
    }
}
