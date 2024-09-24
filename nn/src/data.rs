use anyhow::Result;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::{DataLoader, DataLoaderBuilder};
use burn::data::dataset::Dataset;
use burn::prelude::Backend;
use burn::tensor::Tensor;
use ndarray::{s, Array2};
use ndarray_npy::NpzReader;
use ode_solvers::Vector2;
use ode_solvers::Vector5;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use regex::Regex;
use std::fs;
use std::path::PathBuf;
use std::{fs::File, path::Path};

/// Quadcopter datasets.
#[derive(Debug)]
pub struct QuadcopterDatasets {
    pub train: QuadcopterDataset,
    pub test: QuadcopterDataset,
}

/// Quadcopter dataset.
#[derive(Debug)]
pub struct QuadcopterDataset {
    files: Vec<PathBuf>,
    n_samples_per_file: usize,
}
impl QuadcopterDataset {
    pub fn new(files: Vec<PathBuf>, n_samples_per_file: usize) -> Self {
        Self {
            files,
            n_samples_per_file,
        }
    }
}
impl Dataset<Example> for QuadcopterDataset {
    fn len(&self) -> usize {
        self.files.len() * self.n_samples_per_file
    }
    /// Get an example from the dataset.
    ///
    /// There's some logic in here which basically:
    ///   1. Figures out which file the sample is inside.
    ///   2. Shuffles all the indices in the file (for repeatability),
    ///      using the file index as the seed.
    ///   3. Selects from the first `n_samples_per_file` shuffled indices
    ///      for the specific training sample to use.
    ///
    /// This requires re-loading files for every example, so it could
    /// perhaps be made more efficient.
    fn get(&self, index: usize) -> Option<Example> {
        let file_index = index / self.n_samples_per_file;
        let index_in_file = index % self.n_samples_per_file;
        if let Some(str_path) = self.files[file_index].to_str() {
            if let Ok(array) = read_npz_trajectory(str_path) {
                let mut prng = StdRng::seed_from_u64(file_index as u64);
                let mut indices: Vec<usize> = (0..array.shape()[0]).collect();
                indices.shuffle(&mut prng);
                let row_index = indices[index_in_file % indices.len()];
                let single_row = array
                    .slice(s![row_index, ..])
                    .to_shape((1, array.ncols()))
                    .expect("Array size should be correct")
                    .to_owned();
                let mut examples_vec = trajectory_to_examples(&single_row);
                return Some(examples_vec.pop().expect("Must be an item"));
            }
        }
        None
    }
}

/// Example for training or testing.
#[derive(Debug, Clone)]
pub struct Example {
    pub state: State,
    pub controls: Controls,
}
impl Example {
    pub fn new(state: State, controls: Controls) -> Self {
        Self { state, controls }
    }
}

/// Example batch for training
#[derive(Clone, Debug)]
pub struct ExampleBatch<B: Backend> {
    pub states: Tensor<B, 2>,
    pub control_targets: Tensor<B, 2>,
}

/// Batcher to create example batches for learning examples.
#[derive(Clone)]
pub struct ExampleBatcher<B: Backend> {
    device: B::Device,
    state_norm_params: StateMeanStdev,
}
impl<B: Backend> ExampleBatcher<B> {
    pub fn new(device: B::Device, state_norm_params: StateMeanStdev) -> ExampleBatcher<B> {
        ExampleBatcher {
            device,
            state_norm_params,
        }
    }
}
impl<B: Backend> Batcher<Example, ExampleBatch<B>> for ExampleBatcher<B> {
    fn batch(&self, items: Vec<Example>) -> ExampleBatch<B> {
        let states: Vec<Tensor<B, 2>> = items
            .iter()
            .map(|item| self.state_norm_params.normalize(&item.state))
            .map(|item| item.to_tensor(&self.device))
            .map(|tensor| tensor.unsqueeze())
            .collect();
        let controls: Vec<Tensor<B, 2>> = items
            .iter()
            .map(|item| item.controls.to_tensor(&self.device))
            .map(|tensor| tensor.unsqueeze())
            .collect();

        let states: Tensor<B, 2> = Tensor::cat(states, 0).to_device(&self.device);
        let control_targets: Tensor<B, 2> = Tensor::cat(controls, 0).to_device(&self.device);

        ExampleBatch {
            states,
            control_targets,
        }
    }
}

/// State of the quadcopter.
#[derive(Debug, Clone)]
pub struct State {
    pub x: f32,
    pub z: f32,
    pub theta: f32,
    pub vx: f32,
    pub vz: f32,
}
impl State {
    pub fn new(x: f32, z: f32, theta: f32, vx: f32, vz: f32) -> Self {
        Self {
            x,
            z,
            theta,
            vx,
            vz,
        }
    }

    pub fn from_svector(v: &Vector5<f32>) -> Self {
        Self::new(v[0], v[1], v[2], v[3], v[4])
    }

    pub fn to_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 1> {
        let floats = [self.x, self.z, self.theta, self.vx, self.vz];
        Tensor::<B, 1>::from_floats(floats, device)
    }

    pub fn to_svector(&self) -> Vector5<f32> {
        Vector5::new(self.x, self.z, self.theta, self.vx, self.vz)
    }
}

/// Normalized state of the quadcopter.
pub struct NormalizedState {
    pub state: State,
}
impl NormalizedState {
    pub fn new(x: f32, z: f32, theta: f32, vx: f32, vz: f32) -> Self {
        NormalizedState {
            state: State::new(x, z, theta, vx, vz),
        }
    }

    pub fn to_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 1> {
        self.state.to_tensor(device)
    }
}

/// Mean and standard deviation of the state.
#[derive(Debug, Clone)]
pub struct StateMeanStdev {
    pub x: MeanStdev,
    pub z: MeanStdev,
    pub theta: MeanStdev,
    pub vx: MeanStdev,
    pub vz: MeanStdev,
}
impl StateMeanStdev {
    pub fn new(x: MeanStdev, z: MeanStdev, theta: MeanStdev, vx: MeanStdev, vz: MeanStdev) -> Self {
        Self {
            x,
            z,
            theta,
            vx,
            vz,
        }
    }
    pub fn identity() -> Self {
        let id = MeanStdev::new(0.0, 1.0);
        Self::new(id.clone(), id.clone(), id.clone(), id.clone(), id)
    }
    pub fn normalize(&self, state: &State) -> NormalizedState {
        NormalizedState::new(
            self.x.normalize(state.x),
            self.z.normalize(state.z),
            self.theta.normalize(state.theta),
            self.vx.normalize(state.vx),
            self.vz.normalize(state.vz),
        )
    }
}

/// Controls for the quadcopter.
#[derive(Debug, Clone)]
pub struct Controls {
    pub thrust: f32,
    pub omega: f32,
}
impl Controls {
    pub fn new(thrust: f32, omega: f32) -> Self {
        Self { thrust, omega }
    }

    pub fn from_svector(v: &Vector2<f32>) -> Self {
        Self::new(v[0], v[1])
    }

    pub fn from_tensor<B: Backend>(tensor: &Tensor<B, 1>) -> Controls {
        let data = tensor.to_data().to_vec().unwrap();
        Self::new(data[0], data[1])
    }

    pub fn to_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 1> {
        let floats = [self.thrust, self.omega];
        Tensor::<B, 1>::from_floats(floats, device)
    }

    pub fn to_svector(&self) -> Vector2<f32> {
        Vector2::new(self.thrust, self.omega)
    }
}

/// Load datasets from a parent directoryi.
///
/// This lists all the files, splits them into training and test, and returns
/// the datasets.
pub fn load_datasets(parent_dir: &str, n_samples_per_file: usize) -> Result<QuadcopterDatasets> {
    let training_fraction: f32 = 0.9;

    let mut training_files = list_trajectory_files(parent_dir)?;
    let n_training = (training_files.len() as f32 * training_fraction) as usize;
    let testing_files = training_files.split_off(n_training);

    let train = QuadcopterDataset::new(training_files, n_samples_per_file);
    let test = QuadcopterDataset::new(testing_files, n_samples_per_file);
    let datasets = QuadcopterDatasets { train, test };

    Ok(datasets)
}

/// Convert a trajectory (array from numpy) into a vector of examples for
/// training.
pub fn trajectory_to_examples(trajectory: &Array2<f32>) -> Vec<Example> {
    let mut examples = Vec::with_capacity(trajectory.shape()[0]);
    for row in trajectory.rows() {
        let state = State::new(row[1], row[2], row[3], row[4], row[5]);
        let controls = Controls::new(row[6], row[7]);
        let example = Example::new(state, controls);
        examples.push(example);
    }
    examples
}

/// Read a trajectory in an NPZ file.
///
/// The NPZ file must have an array named `trajectory`.
pub fn read_npz_trajectory(file_name: &str) -> Result<Array2<f32>> {
    let mut npz = NpzReader::new(File::open(file_name)?)?;
    let trajectory: Array2<f32> = npz.by_name("trajectory")?;
    Ok(trajectory)
}

/// List all the trajectory file paths in a parent directory.
///
/// This lists the files sorted by their trajectory number.
pub fn list_trajectory_files(parent_dir: &str) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let re = Regex::new(r"^\d{7}.npz").unwrap();
    let dir = Path::new(parent_dir);
    for entry in fs::read_dir(dir)? {
        if let Ok(entry) = entry {
            let path = entry.path().canonicalize()?;
            if let Some(file_name) = path.file_name().and_then(|s| s.to_str()) {
                if re.is_match(file_name) {
                    files.push(path.to_path_buf());
                }
            }
        }
    }
    files.sort();
    Ok(files)
}

/// Mean and standard deviation, for scaling data.
#[derive(Clone, Debug)]
pub struct MeanStdev {
    mean: f32,
    stdev: f32,
}
impl MeanStdev {
    pub fn new(mean: f32, stdev: f32) -> Self {
        MeanStdev { mean, stdev }
    }
    pub fn normalize(&self, value: f32) -> f32 {
        (value - self.mean) / self.stdev
    }
    pub fn denormalize(&self, normalized_value: f32) -> f32 {
        self.stdev * normalized_value + self.mean
    }
}

fn compute_mean_and_stdev<B: Backend>(
    dataloader: &dyn DataLoader<ExampleBatch<B>>,
) -> StateMeanStdev {
    let mut running_mean = None;
    let mut running_variance = None;
    let mut count = 0;
    let mut batch_count = 0;
    let report_every_n_batches = 100;

    for batch in dataloader.iter() {
        let states = batch.states;
        let batch_size = states.shape().dims[0];

        // compute batch mean and variance
        let batch_mean = states.clone().mean_dim(0);
        let batch_var = states.clone().var(0);

        // update the running mean
        if running_mean.is_none() {
            running_mean = Some(batch_mean.clone());
            running_variance = Some(batch_var.clone());
        } else {
            let old_mean = running_mean.unwrap();
            let old_var = running_variance.unwrap();

            let new_count = count + batch_size;
            let delta = batch_mean.clone() - old_mean.clone();

            let new_mean = old_mean + delta.clone() * ((batch_size as f32) / (new_count as f32));
            let new_var = old_var
                + (batch_var.clone()
                    + delta.powf_scalar(2.0)
                        * ((count as f32) * (batch_size as f32) / (new_count as f32)))
                    * ((batch_size as f32) / (new_count as f32));

            running_mean = Some(new_mean);
            running_variance = Some(new_var);
        }

        if batch_count % report_every_n_batches == 0 {
            let smsd = tensors_to_state_mean_stdev(
                &running_mean.clone().unwrap(),
                &running_variance.clone().unwrap(),
            );
            println!("count: {}, smsd: {:?}", count, smsd);
        }

        count += batch_size;
        batch_count += 1;
    }

    tensors_to_state_mean_stdev(&running_mean.unwrap(), &running_variance.unwrap())
}

fn tensors_to_state_mean_stdev<B: Backend>(
    running_mean: &Tensor<B, 2>,
    running_variance: &Tensor<B, 2>,
) -> StateMeanStdev {
    let means: Vec<f32> = running_mean.to_data().to_vec().unwrap();
    let vars: Vec<f32> = running_variance.to_data().to_vec().unwrap();

    StateMeanStdev::new(
        MeanStdev::new(means[0], vars[0].sqrt()),
        MeanStdev::new(means[1], vars[1].sqrt()),
        MeanStdev::new(means[2], vars[2].sqrt()),
        MeanStdev::new(means[3], vars[3].sqrt()),
        MeanStdev::new(means[4], vars[4].sqrt()),
    )
}

pub fn compute_training_mean_and_stdev<B: Backend>(device: B::Device) {
    let n_samples_per_file = 60;
    let batch_size = 8;
    let num_workers = 4;
    let trajectories_dir = "../trajectories/quadcopter";

    let datasets = load_datasets(trajectories_dir, n_samples_per_file).unwrap();
    let train_dataset = datasets.train;
    let batcher_train = ExampleBatcher::<B>::new(device.clone(), StateMeanStdev::identity());
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(batch_size)
        .num_workers(num_workers)
        .build(train_dataset);

    let smsd = compute_mean_and_stdev(dataloader_train.as_ref());

    println!("final: smsd: {:?}", smsd);
}
