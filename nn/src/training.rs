use crate::{
    data::{load_datasets, ExampleBatcher, MeanStdev, StateMeanStdev},
    model::ModelConfig,
};
use anyhow::Result;
use burn::{
    data::dataloader::DataLoaderBuilder,
    module::Module,
    optim::AdamConfig,
    record::{DefaultFileRecorder, FullPrecisionSettings},
    tensor::backend::AutodiffBackend,
    train::{
        metric::{
            store::{Aggregate, Direction, Split},
            LossMetric,
        },
        LearnerBuilder, MetricEarlyStoppingStrategy, StoppingCondition,
    },
};

static ARTIFACT_DIR: &str = "./burn-training-artifacts";
static TRAJECTORIES_DIR: &str = "../trajectories/quadcopter";

pub fn run<B: AutodiffBackend>(device: B::Device) -> Result<()> {
    // General config
    let n_samples_per_file = 60;
    let batch_size = 8;
    let seed = 42;
    let num_workers = 6;

    //
    let optimizer = AdamConfig::new();
    let config = ModelConfig::new();
    let model = config.init::<B>(&device);
    let num_epochs = 6;
    let lr = 1e-4;

    // Define train/test datasets
    let datasets = load_datasets(TRAJECTORIES_DIR, n_samples_per_file)?;
    let train_dataset = datasets.train;
    let test_dataset = datasets.test;

    // TODO: Obtain better values by actually surveying the data
    let state_norm_params = StateMeanStdev::new(
        MeanStdev::new(0.0, 10.0),
        MeanStdev::new(6.0, 22.0),
        MeanStdev::new(0.0, 1.0),
        MeanStdev::new(0.0, 10.0),
        MeanStdev::new(0.0, 10.0),
    );

    let batcher_train = ExampleBatcher::<B>::new(device.clone(), state_norm_params.clone());
    let batcher_test =
        ExampleBatcher::<B::InnerBackend>::new(device.clone(), state_norm_params.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(batch_size)
        .shuffle(seed)
        .num_workers(num_workers)
        .build(train_dataset);
    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(batch_size)
        .shuffle(seed)
        .num_workers(num_workers)
        .build(test_dataset);

    // Model
    let learner = LearnerBuilder::new(ARTIFACT_DIR)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(DefaultFileRecorder::<FullPrecisionSettings>::new())
        .early_stopping(MetricEarlyStoppingStrategy::new::<LossMetric<B>>(
            Aggregate::Mean,
            Direction::Lowest,
            Split::Valid,
            StoppingCondition::NoImprovementSince { n_epochs: 1 },
        ))
        .devices(vec![device.clone()])
        .num_epochs(num_epochs)
        .summary()
        .build(model, optimizer.init(), lr);

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    model_trained
        .save_file(
            format!("{ARTIFACT_DIR}/model"),
            &DefaultFileRecorder::<FullPrecisionSettings>::new(),
        )
        .expect("Failed to save trained model");

    Ok(())
}
