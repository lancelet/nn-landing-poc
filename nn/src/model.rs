use burn::nn::loss::MseLoss;
use burn::nn::loss::Reduction::Mean;
use burn::tensor::backend::AutodiffBackend;
use burn::train::{RegressionOutput, TrainOutput, TrainStep, ValidStep};
use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    prelude::Backend,
    tensor::Tensor,
};

use crate::data::{Controls, ExampleBatch, MeanStdev, State, StateMeanStdev};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    linear3: Linear<B>,
    linear4: Linear<B>,
    linear5: Linear<B>,
}
impl<B: Backend> Model<B> {
    /// Run the model forward in inference mode.
    ///
    /// Some normalization values are hard-coded here which really shouldn't
    /// be.
    pub fn inference(&self, device: &B::Device, input: &State) -> Controls {
        // NB: These must match training values.
        let state_norm_params = StateMeanStdev::new(
            MeanStdev::new(0.0, 10.0),
            MeanStdev::new(6.0, 22.0),
            MeanStdev::new(0.0, 1.0),
            MeanStdev::new(0.0, 10.0),
            MeanStdev::new(0.0, 10.0),
        );
        let normalized_state = state_norm_params.normalize(input);

        // Run the model forward
        let normalized_controls = self.forward(normalized_state.to_tensor(device).unsqueeze());

        // basic outputs are clamped to (-1, 1)
        // scale:
        //   - thrust to [0, 20]
        //   - omega to [-2, 2]
        let offset: Tensor<B, 1> = Tensor::from_floats([1.0, 0.0], &normalized_controls.device());
        let scaling: Tensor<B, 1> = Tensor::from_floats([10.0, 2.0], &normalized_controls.device());
        let output = (normalized_controls + offset.unsqueeze()) * scaling.unsqueeze();

        // convert the output tensor back to controls
        Controls::from_tensor(&output.squeeze(0))
    }

    pub fn forward(&self, state: Tensor<B, 2>) -> Tensor<B, 2> {
        let relu = Relu::new();
        let x = relu.forward(self.linear1.forward(state));
        let x = relu.forward(self.linear2.forward(x));
        let x = relu.forward(self.linear3.forward(x));
        let x = relu.forward(self.linear4.forward(x));
        let x = self.linear5.forward(x);
        let x = x.clamp(-1.0, 1.0);
        x
    }

    pub fn forward_regression(&self, example_batch: ExampleBatch<B>) -> RegressionOutput<B> {
        let states = example_batch.states;
        let targets = example_batch.control_targets;

        let output = self.forward(states);

        // basic outputs are clamped to (-1, 1)
        // scale:
        //   - thrust to [0, 20]
        //   - omega to [-2, 2]
        let offset: Tensor<B, 1> = Tensor::from_floats([1.0, 0.0], &output.device());
        let scaling: Tensor<B, 1> = Tensor::from_floats([10.0, 2.0], &output.device());
        let output = (output + offset.unsqueeze()) * scaling.unsqueeze();

        let loss = MseLoss::new().forward(output.clone(), targets.clone(), Mean);

        RegressionOutput {
            loss,
            output,
            targets,
        }
    }
}
impl<B: AutodiffBackend> TrainStep<ExampleBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, example_batch: ExampleBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let regression_output = self.forward_regression(example_batch);
        TrainOutput::new(self, regression_output.loss.backward(), regression_output)
    }
}
impl<B: Backend> ValidStep<ExampleBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, example_batch: ExampleBatch<B>) -> RegressionOutput<B> {
        self.forward_regression(example_batch)
    }
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = "32")]
    linear_size: usize,
}
impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        let sz = self.linear_size;
        Model {
            linear1: LinearConfig::new(5, sz).init(device),
            linear2: LinearConfig::new(sz, sz).init(device),
            linear3: LinearConfig::new(sz, sz).init(device),
            linear4: LinearConfig::new(sz, sz).init(device),
            linear5: LinearConfig::new(sz, 2).init(device),
        }
    }
}
