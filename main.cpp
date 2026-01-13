// ------------------------------------------------------------------------
// RF-DETR ONNX Runtime C++ Implementation - Main Test Harness
// Copyright (c) 2025. All Rights Reserved.
// Licensed under the Apache License, Version 2.0
// ------------------------------------------------------------------------

#include "engine.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

// COCO class names (91 classes - standard COCO IDs with gaps)
// Index 0 is background, classes 1-90 follow original COCO dataset IDs
const std::vector<std::string> COCO_CLASSES = {
    "__background__",  // 0
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",  // 1-9
    "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog",  // 10-18
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella",  // 19-27
    "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",  // 28-36
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",  // 37-43
    "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",  // 44-53
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",  // 54-62
    "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop",  // 63-71
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",  // 72-79
    "refrigerator", "N/A", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",  // 80-87
    "toothbrush"  // 88-88 (indices 89-90 unused)
};

// Generate random colors for visualization
std::vector<cv::Scalar> generate_colors(int num_classes) {
    std::vector<cv::Scalar> colors;
    std::srand(42);  // Fixed seed for reproducibility

    for (int i = 0; i < num_classes; ++i) {
        colors.push_back(cv::Scalar(
            std::rand() % 256,
            std::rand() % 256,
            std::rand() % 256
        ));
    }

    return colors;
}

// Draw detections on image
void draw_detections(cv::Mat& image,
                     const std::vector<Detection>& detections,
                     const std::vector<cv::Scalar>& colors) {
    for (const auto& det : detections) {
        // Get color for this class
        cv::Scalar color = colors[det.class_id % colors.size()];

        // Draw bounding box
        cv::rectangle(image, det.box, color, 2);

        // Prepare label text
        std::string label;
        if (det.class_id < static_cast<int>(COCO_CLASSES.size())) {
            label = COCO_CLASSES[det.class_id];
        } else {
            label = "class_" + std::to_string(det.class_id);
        }
        label += " " + std::to_string(static_cast<int>(det.confidence * 100)) + "%";

        // Draw label background
        int baseline = 0;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                               0.5, 1, &baseline);
        int top = std::max(det.box.y, label_size.height);
        cv::rectangle(image,
                      cv::Point(det.box.x, top - label_size.height),
                      cv::Point(det.box.x + label_size.width, top + baseline),
                      color, cv::FILLED);

        // Draw label text
        cv::putText(image, label,
                    cv::Point(det.box.x, top),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 255), 1);
    }
}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "RF-DETR ONNXRuntime C++ Inference" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    // Show help if requested
    if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        std::cout << "Usage: " << argv[0] << " [model] [image] [conf] [provider] [opt] [cpu_mode] [fp16]" << std::endl;
        std::cout << std::endl;
        std::cout << "Arguments:" << std::endl;
        std::cout << "  model       Path to ONNX model file (default: rf-detr-nano.onnx)" << std::endl;
        std::cout << "  image       Path to input image (default: test.jpg)" << std::endl;
        std::cout << "  conf        Confidence threshold 0.0-1.0 (default: 0.5)" << std::endl;
        std::cout << "  provider    Execution provider (default: CPUExecutionProvider)" << std::endl;
        std::cout << "  opt         Optimization level (default: extended)" << std::endl;
        std::cout << "  cpu_mode    CPU threading mode (default: high-thread-count)" << std::endl;
        std::cout << "  fp16        Enable FP16 for TensorRT: 0=FP32, 1=FP16 (default: 0)" << std::endl;
        std::cout << std::endl;
        std::cout << "Execution Providers:" << std::endl;
        std::cout << "  CPUExecutionProvider           - CPU inference with SIMD optimizations" << std::endl;
        std::cout << "  CUDAExecutionProvider          - NVIDIA CUDA GPU inference" << std::endl;
        std::cout << "  TensorrtExecutionProvider      - NVIDIA TensorRT with engine caching" << std::endl;
        std::cout << "                                   Use fp16=1 for FP16 mode" << std::endl;
        std::cout << std::endl;
        std::cout << "Optimization Levels:" << std::endl;
        std::cout << "  disable     - No graph optimizations" << std::endl;
        std::cout << "  basic       - Basic optimizations" << std::endl;
        std::cout << "  extended    - Extended optimizations (recommended)" << std::endl;
        std::cout << "  all         - All hardware-specific optimizations" << std::endl;
        std::cout << std::endl;
        std::cout << "CPU Threading Modes:" << std::endl;
        std::cout << "  auto                - Auto-detect optimal settings" << std::endl;
        std::cout << "  high-thread-count   - Optimized for >16 CPU cores" << std::endl;
        std::cout << std::endl;
        std::cout << "Examples:" << std::endl;
        std::cout << "  # TensorRT FP32" << std::endl;
        std::cout << "  " << argv[0] << " model.onnx image.jpg 0.5 TensorrtExecutionProvider extended auto 0" << std::endl;
        std::cout << "  # TensorRT FP16" << std::endl;
        std::cout << "  " << argv[0] << " model.onnx image.jpg 0.5 TensorrtExecutionProvider extended auto 1" << std::endl;
        return 0;
    }

    // Parse command line arguments
    std::string model_path = "rf-detr-nano.onnx";
    std::string image_path = "test.jpg";
    float conf_threshold = 0.5f;
    std::string provider = "CPUExecutionProvider";
    std::string opt_level = "extended";
    std::string cpu_mode = "high-thread-count";
    bool use_fp16 = false;

    if (argc > 1) model_path = argv[1];
    if (argc > 2) image_path = argv[2];
    if (argc > 3) conf_threshold = std::stof(argv[3]);
    if (argc > 4) {
        provider = argv[4];
        // Validate provider
        if (provider != "CPUExecutionProvider" &&
            provider != "CUDAExecutionProvider" &&
            provider != "TensorrtExecutionProvider") {
            std::cerr << "Error: Unknown provider '" << provider << "'" << std::endl;
            std::cerr << "Valid providers: CPUExecutionProvider, CUDAExecutionProvider, "
                      << "TensorrtExecutionProvider" << std::endl;
            std::cerr << "Use --help for more information" << std::endl;
            return 1;
        }
    }
    if (argc > 5) opt_level = argv[5];
    if (argc > 6) cpu_mode = argv[6];
    if (argc > 7) use_fp16 = (std::stoi(argv[7]) != 0);

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Model: " << model_path << std::endl;
    std::cout << "  Image: " << image_path << std::endl;
    std::cout << "  Confidence Threshold: " << conf_threshold << std::endl;
    std::cout << "  Execution Provider: " << provider << std::endl;
    std::cout << "  Optimization Level: " << opt_level << std::endl;
    std::cout << "  CPU Mode: " << cpu_mode << std::endl;
    std::cout << "  FP16 Mode: " << (use_fp16 ? "ENABLED" : "DISABLED") << std::endl;
    std::cout << std::endl;

    try {
        // Initialize engine
        std::wstring model_path_w(model_path.begin(), model_path.end());
        RFDETREngine engine(model_path_w, "RF-DETR", provider.c_str(), opt_level.c_str(), cpu_mode.c_str(), use_fp16);

        // Load image
        std::cout << "Loading image: " << image_path << std::endl;
        cv::Mat image = cv::imread(image_path);

        if (image.empty()) {
            std::cerr << "Error: Could not load image: " << image_path << std::endl;
            return 1;
        }

        std::cout << "Image loaded: " << image.cols << "x" << image.rows << std::endl;
        std::cout << std::endl;

        // Warm-up run (first inference is often slower due to initialization)
        std::cout << "Performing warm-up inference..." << std::endl;
        engine.infer(image, conf_threshold);
        std::cout << "Warm-up complete" << std::endl;
        std::cout << std::endl;

        // Benchmark inference (100 runs for stable results)
        std::cout << "Benchmarking inference (100 runs)..." << std::endl;
        std::vector<double> times;
        std::vector<Detection> detections;

        for (int i = 0; i < 100; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            detections = engine.infer(image, conf_threshold);
            auto end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double, std::milli> elapsed = end - start;
            times.push_back(elapsed.count());

            // Print progress every 10 runs
            if ((i + 1) % 10 == 0 || i == 0) {
                std::cout << "  Run " << (i + 1) << ": " << std::fixed << std::setprecision(2)
                          << elapsed.count() << " ms (" << (1000.0 / elapsed.count())
                          << " FPS)" << std::endl;
            }
        }

        // Calculate statistics
        double total_time = 0.0;
        for (double t : times) total_time += t;
        double avg_time = total_time / times.size();
        double avg_fps = 1000.0 / avg_time;

        std::cout << std::endl;
        std::cout << "Performance Statistics:" << std::endl;
        std::cout << "  Average inference time: " << std::fixed << std::setprecision(2)
                  << avg_time << " ms" << std::endl;
        std::cout << "  Average FPS: " << std::fixed << std::setprecision(2)
                  << avg_fps << std::endl;
        std::cout << std::endl;

        // Display detection results
        std::cout << "Detection Results:" << std::endl;
        std::cout << "  Total detections: " << detections.size() << std::endl;
        std::cout << std::endl;

        if (!detections.empty()) {
            std::cout << "Detections:" << std::endl;
            for (size_t i = 0; i < detections.size(); ++i) {
                const auto& det = detections[i];
                std::string class_name = (det.class_id < static_cast<int>(COCO_CLASSES.size()))
                                        ? COCO_CLASSES[det.class_id]
                                        : "class_" + std::to_string(det.class_id);

                std::cout << "  [" << (i + 1) << "] "
                          << class_name << " - "
                          << std::fixed << std::setprecision(1)
                          << (det.confidence * 100.0f) << "% - "
                          << "Box: (" << det.box.x << ", " << det.box.y << ", "
                          << det.box.width << ", " << det.box.height << ")"
                          << std::endl;
            }
            std::cout << std::endl;

            // Visualize results
            std::cout << "Generating visualization..." << std::endl;
            cv::Mat vis_image = image.clone();
            auto colors = generate_colors(91);  // 91 classes
            draw_detections(vis_image, detections, colors);

            // Save result
            std::string output_path = "output.jpg";
            cv::imwrite(output_path, vis_image);
            std::cout << "Visualization saved to: " << output_path << std::endl;
        } else {
            std::cout << "No detections above confidence threshold " << conf_threshold << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Inference completed successfully!" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
