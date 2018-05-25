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
use std::time::SystemTime;
use tensorflow::Code;
use tensorflow::Graph;
use tensorflow::ImportGraphDefOptions;
use tensorflow::Session;
use tensorflow::SessionOptions;
use tensorflow::Status;
use tensorflow::StepWithGraph;
use tensorflow::Tensor;

const MASK_RCNN_PB: &str = "/Users/thrivesmart1/Projects/rust-tensorflow/data/mask_rcnn.pb";
const IMG_256SQ: &str = "/Users/thrivesmart1/Projects/rust-tensorflow/data/example.png";
const TF_MASKRCNN_TOILET_CLASSNUM: f32 = 1f32;
const TF_MASKRCNN_PREDICTION_THRESSHOLD: f32 = 0.9;
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

macro_rules! row_major_3d_idx {
    ($i:expr, $j:expr, $k:expr; $i_len:expr, $j_len:expr) => (($i + $i_len*$j + $i_len*$j_len*$k) as usize);
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

    let img_pixels = {
        let opened: image::DynamicImage = image::open(&IMG_256SQ)?;
        opened.raw_pixels() // FIXME raw_pixels creates a clone, which is probably not the most efficient
    };

    let img_float: Vec<f32> = img_pixels.iter().map(|&p| p as f32).collect();
    // let (img_w, img_h) = img.dimensions();
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

    let wallclock = SystemTime::now();

    println!("[+{}s] reading graph from disk...", SystemTime::now().duration_since(wallclock).unwrap().as_secs());
    File::open(MASK_RCNN_PB)?.read_to_end(&mut proto)?;
    println!("[+{}s] done reading graph from disk.", SystemTime::now().duration_since(wallclock).unwrap().as_secs());
    println!("[+{}s] parsing graph...", SystemTime::now().duration_since(wallclock).unwrap().as_secs());
    graph.import_graph_def(&proto, &ImportGraphDefOptions::new())?;
    let mut session = Session::new(&SessionOptions::new(), &graph)?;
    println!("[+{}s] done parsing graph.", SystemTime::now().duration_since(wallclock).unwrap().as_secs());

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
    println!("[+{}s] running inference session...", SystemTime::now().duration_since(wallclock).unwrap().as_secs());
    session.run(&mut step)?;
    println!("[+{}s] session complete.", SystemTime::now().duration_since(wallclock).unwrap().as_secs());

    // Check our results.

    let masks_tensor: Tensor<f32> = step.take_output(masks)?;
    if masks_tensor.dims().len() != 5 || masks_tensor.dims()[4] != 2 {
        return Err(Box::new(
            Status::new_set(
                Code::OutOfRange,
                &format!("Expected mask dimensions to be [1,100,28,28,2] but got: {:?}", masks_tensor),
            ).unwrap(),
        ));
    }


    let mut highest_score_index: i32 = -1;
    let mut highest_score_value: f32 = -1f32;
    let detections_tensor: Tensor<f32> = step.take_output(detections)?;
    let detections_dim0 = detections_tensor.dims()[0];
    let detections_dim1 = detections_tensor.dims()[1];

    for i in 0..masks_tensor.dims()[1] {
        let score_at_i = detections_tensor[row_major_3d_idx!(0, i, 5; detections_dim0, detections_dim1)];
        let detected_class = detections_tensor[row_major_3d_idx!(0, i, 4; detections_dim0, detections_dim1)];
        println!("      {}: got detected class {} with score {} ", i, detected_class, score_at_i);
        if detected_class == TF_MASKRCNN_TOILET_CLASSNUM && score_at_i > TF_MASKRCNN_PREDICTION_THRESSHOLD && score_at_i > highest_score_value {
            // check that it's a greater than zero area
            let y1 = detections_tensor[row_major_3d_idx!(0, i, 0; detections_dim0, detections_dim1)];
            let x1 = detections_tensor[row_major_3d_idx!(0, i, 1; detections_dim0, detections_dim1)];
            let y2 = detections_tensor[row_major_3d_idx!(0, i, 2; detections_dim0, detections_dim1)];
            let x2 = detections_tensor[row_major_3d_idx!(0, i, 3; detections_dim0, detections_dim1)];
            let mask_height = y2 - y1;
            let mask_width = x2 - x1;

            if mask_height != 0f32 && mask_width != 0f32 {
                highest_score_value = score_at_i;
                highest_score_index = i as i32;
                println!("Detected new, better toilet seat at {} with {}% confidence with w:{} h:{}", i, (score_at_i*100.0), mask_width, mask_height);
            }
        }
    }

    if highest_score_index > -1 {
        println!("[+{}s] Final detection at {} with {}% confidence.", SystemTime::now().duration_since(wallclock).unwrap().as_secs(), highest_score_index, detections_tensor[row_major_3d_idx!(0, highest_score_index as u64, 5; detections_dim0, detections_dim1)]);
        // let detected_class = detections_tensor[row_major_3d_idx!(0, highest_score_index as u64, 4; detections_dim0, detections_dim1)];

        // Pointer arithmetic
        // let i0 = 0i64, /* size0 = (int)outputs[3].shape().dim_size(1), */ i1 = highestScoreIndex, size1 = (int)outputs[3].shape().dim_size(1), size2 = (int)outputs[3].shape().dim_size(2), size3 = (int)outputs[3].shape().dim_size(3), i4 = (int)detectedClass /*, size4 = 2 */;
        // int pointerLocationOfI = (i0*size1 + i1)*size2;
        // float_t *maskPointer = outputs[3].flat<float_t>().data();
        //
        // // The shape of the detection is [28,28,2], where the last index is the class of interest.
        // // We'll extract index 1 because it's the toilet seat.
        // cv::Mat initialMask(cv::Size(size2, size3), CV_32FC2, &maskPointer[pointerLocationOfI]); // CV_32FC2 because I know size4 is 2
        // cv::Mat detectedMask(initialMask.size(), CV_32FC1);
        // cv::extractChannel(initialMask, detectedMask, i4);
        //
        // // Convert to B&W
        // cv::Mat binaryMask(detectedMask.size(), CV_8UC1);
        // cv::threshold(detectedMask, binaryMask, 0.5, 255, cv::THRESH_BINARY);
        //
        // // First scale and offset in relation to TF_MASKRCNN_IMG_WIDTHHEIGHT
        // cv::Mat scaledDetectionMat(maskHeight, maskWidth, CV_8UC1);
        // cv::resize(binaryMask, scaledDetectionMat, scaledDetectionMat.size(), 0, 0);
        // cv::Mat scaledOffsetMat(moldedInput.size(), CV_8UC1, cv::Scalar(0));
        // scaledDetectionMat.copyTo(scaledOffsetMat(cv::Rect(x1, y1, maskWidth, maskHeight)));
        //
        // // Second, scale and offset in relation to our original color image
        // cv::Mat detectionScaledToSquare(squareInputMat.size(), CV_8UC1);
        // cv::resize(scaledOffsetMat, detectionScaledToSquare, detectionScaledToSquare.size(), 0, 0);
        //
        // detectionScaledToSquare(cv::Rect(leftBorder, topBorder, colorImg.size().width, colorImg.size().height)).copyTo(dest);
    }


    Ok(())
}
