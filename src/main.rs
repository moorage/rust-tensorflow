#![cfg_attr(feature = "nightly", feature(alloc_system))]
#[cfg(feature = "nightly")]
extern crate alloc_system;
extern crate image;
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

const MASK_RCNN_PB: &str = "/Users/mattmoore/Projects/rust-tensorflow/data/mask_rcnn.pb";
const IMG_256SQ: &str = "/Users/mattmoore/Projects/rust-tensorflow/data/example.png";
// const TF_MASKRCNN_TOILET_CLASSNUM: i32 = 1;
const TF_MASKRCNN_IMG_WIDTHHEIGHT: f32 = 256f32;
const TF_MASKRCNN_IMAGE_METADATA_LENGTH: u64 = 10;
const TF_MASKRCNN_IMAGE_METADATA: [f32; 10] = [
    0f32,
    TF_MASKRCNN_IMG_WIDTHHEIGHT,
    TF_MASKRCNN_IMG_WIDTHHEIGHT,
    3f32,
    0f32,
    0f32,
    TF_MASKRCNN_IMG_WIDTHHEIGHT,
    TF_MASKRCNN_IMG_WIDTHHEIGHT,
    0f32,
    0f32,
];

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
                &format!("Couldn't find model at {}.", MASK_RCNN_PB),
            ).unwrap(),
        ));
    }

    // Create input variables for our addition
    let opened_img = image::open(&IMG_256SQ)?;
    // let img =
    // let (img_w, img_h) = img.dimensions();
    let img_float: Vec<f32> = opened_img.as_rgb8().unwrap().into_raw().iter().map(|&p| p as f32).collect();
    let input_img = <Tensor<f32>>::new(&[
        1,
        TF_MASKRCNN_IMG_WIDTHHEIGHT as u64,
        TF_MASKRCNN_IMG_WIDTHHEIGHT as u64,
        3,
    ]).with_values(&img_float)
        .unwrap();
    let input_meta = <Tensor<f32>>::new(&[1, TF_MASKRCNN_IMAGE_METADATA_LENGTH])
        .with_values(&TF_MASKRCNN_IMAGE_METADATA)
        .unwrap();

    // Load the computation graph defined by regression.py.
    let mut graph = Graph::new();
    let mut proto = Vec::new();
    File::open(MASK_RCNN_PB)?.read_to_end(&mut proto)?;
    graph.import_graph_def(&proto, &ImportGraphDefOptions::new())?;
    let mut session = Session::new(&SessionOptions::new(), &graph)?;

    // Run the Step
    let mut step = StepWithGraph::new();
    step.add_input(
        &graph.operation_by_name_required("input_image")?,
        0,
        &input_img,
    );
    step.add_input(
        &graph.operation_by_name_required("input_image_meta")?,
        0,
        &input_meta,
    );
    let detections =
        step.request_output(&graph.operation_by_name_required("output_detections")?, 0);
    let masks = step.request_output(&graph.operation_by_name_required("output_mrcnn_mask")?, 0);
    session.run(&mut step)?;

    // Check our results.
    let detections_res: i32 = step.take_output(detections)?[0];
    println!("detections {:?}", detections_res);
    let masks_res: i32 = step.take_output(masks)?[0];
    println!("masks {:?}", masks_res);

    Ok(())
}
