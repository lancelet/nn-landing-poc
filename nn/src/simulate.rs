use std::f32::consts::PI;

use crate::{
    data::{Controls, State},
    model::Model,
};
use burn::prelude::Backend;
use ode_solvers::{Rk4, System, Vector5};

#[derive(Debug)]
pub struct SimSample {
    pub time: f32,
    pub state: State,
    pub controls: Controls,
}
impl SimSample {
    fn new(time: f32, state: State, controls: Controls) -> Self {
        Self {
            time,
            state,
            controls,
        }
    }
}

pub struct ModelSystem<B: Backend> {
    pub model: Model<B>,
    pub device: B::Device,
}
impl<B: Backend> ModelSystem<B> {
    fn new(model: Model<B>, device: B::Device) -> Self {
        Self { model, device }
    }
}

impl<B: Backend> System<f32, Vector5<f32>> for ModelSystem<B> {
    // System to integrate: ODE dynamics with controls from the neural-network
    // model.
    fn system(&self, _x: f32, y: &Vector5<f32>, dy: &mut Vector5<f32>) {
        let state = State::from_svector(y);
        let controls = self.model.inference(&self.device, &state);

        let g = 9.81;
        let x_dot = state.vx;
        let z_dot = state.vz;
        let vx_dot = controls.thrust * state.theta.sin();
        let vz_dot = controls.thrust * state.theta.cos() - g;
        let theta_dot = controls.omega;

        dy.set_column(
            0,
            &State::new(x_dot, z_dot, theta_dot, vx_dot, vz_dot).to_svector(),
        );
    }

    fn solout(&mut self, _x: f32, y: &Vector5<f32>, _dy: &Vector5<f32>) -> bool {
        let state = State::from_svector(y);
        let target_x = 0.0;
        let target_z = 0.1;
        let target_angle = 0.0;
        let distance = ((state.x - target_x).powf(2.0) + (state.z - target_z).powf(2.0)).sqrt();
        let d_angle = (state.theta - target_angle).abs();

        // println!("distance = {}, d_angle = {}", distance, d_angle);

        (distance < 0.35) && (d_angle < (0.5 * PI / 180.0))
    }
}

// Simulate a forward integration of the model.
pub fn simulate<B: Backend>(
    device: B::Device,
    model: &Model<B>,
    start: State,
    t_final: f32,
    dt: f32,
) -> Vec<SimSample> {
    let model_system = ModelSystem::new(model.clone(), device.clone());

    let mut stepper = Rk4::new(model_system, 0.0, start.to_svector(), t_final, dt);
    /*
    let mut stepper = Dop853::new(
        model_system,
        0.0,
        t_final,
        dt,
        start.to_svector(),
        rtol,
        atol,
    );
    */

    let _res = stepper.integrate();

    let ts = stepper.x_out();
    let ys = stepper.y_out();
    assert!(ts.len() == ys.len());

    let states: Vec<State> = ys.iter().map(State::from_svector).collect();
    let controls: Vec<Controls> = states
        .iter()
        .map(|state| model.inference(&device, state))
        .collect();

    states
        .into_iter()
        .zip(controls.into_iter())
        .zip(ts.into_iter())
        .map(|((state, control), &t)| SimSample::new(t, state, control))
        .collect()
}
