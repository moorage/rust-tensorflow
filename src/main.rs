#![cfg_attr(feature = "nightly", feature(alloc_system))]
#[cfg(feature = "nightly")]
extern crate alloc_system;
extern crate tensorflow;

use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::process::exit;
use std::result::Result;
use tensorflow::Code;
use tensorflow::Graph;
use tensorflow::ImportGraphDefOptions;
use tensorflow::Session;
use tensorflow::SessionOptions;
use tensorflow::Status;
use tensorflow::StepWithGraph;
use tensorflow::Tensor;

static MASK_RCNN_PB: &'static str = "/Users/mattmoore/Projects/rust-tensorflow/data/mask_rcnn.pb";

fn main() {
    // Putting the main code in another function serves two purposes:
    // 1. We can use the `?` operator.
    // 2. We can call exit safely, which does not run any destructors.
    exit(match run() {
        Ok(_) => 0,
        Err(e) => {
            println!("{}", e);
            1
        }
    })
}

fn run() -> Result<(), Box<Error>> {
    if !Path::new(MASK_RCNN_PB).exists() {
        return Err(Box::new(
            Status::new_set(
                Code::NotFound,
                &format!(
                    "Couldn't find model at {}.",
                    MASK_RCNN_PB
                ),
            ).unwrap(),
        ));
    }

    // Create input variables for our addition
    let mut x = Tensor::new(&[1]);
    x[0] = 2i32;
    let mut y = Tensor::new(&[1]);
    y[0] = 40i32;

    // Load the computation graph defined by regression.py.
    let mut graph = Graph::new();
    let mut proto = Vec::new();
    File::open(MASK_RCNN_PB)?.read_to_end(&mut proto)?;
    graph.import_graph_def(&proto, &ImportGraphDefOptions::new())?;
    let mut session = Session::new(&SessionOptions::new(), &graph)?;

    // Run the Step
    let mut step = StepWithGraph::new();
    step.add_input(&graph.operation_by_name_required("input_image")?, 0, &x);
    step.add_input(&graph.operation_by_name_required("input_image_meta")?, 0, &y);
    let detections = step.request_output(&graph.operation_by_name_required("output_detections")?, 0);
    let masks = step.request_output(&graph.operation_by_name_required("output_mrcnn_mask")?, 0);
    session.run(&mut step)?;

    // Check our results.
    let detections_res: i32 = step.take_output(detections)?[0];
    println!("detections {:?}", detections_res);
    let masks_res: i32 = step.take_output(masks)?[0];
    println!("masks {:?}", masks_res);

    Ok(())
}
