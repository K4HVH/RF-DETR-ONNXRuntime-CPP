// ------------------------------------------------------------------------
// RF-DETR demo CLI (image / video / benchmark modes)
// ------------------------------------------------------------------------
#include "rfdetr/rfdetr.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace rfdetr;

namespace {

struct Args {
    std::string model       = "models/inference_model.onnx";
    std::string input       = "test2.png";
    std::string output      = "output.jpg";
    std::string mode        = "image";   // image | video | benchmark
    std::string device      = "auto";    // auto | cpu | cuda | tensorrt | tensorrt-rtx
    std::string precision   = "fp16";    // fp32 | fp16 | int8
    std::string int8_mode   = "nocal";   // cal | qdq | nocal
    std::string int8_table;
    std::string cache_dir   = "trt_cache";
    std::string classes_csv = "body,head";
    int         device_id   = 0;
    int         batch       = 1;
    int         iters       = 100;
    int         warmup      = 10;
    float       conf        = 0.5f;
    bool        verbose     = false;
    bool        no_graph    = false;
    bool        no_gpu_pre  = false;
    bool        show        = false;
};

void print_help() {
    std::cout <<
R"(RF-DETR demo
Usage: rfdetr_demo [options]

  --model <path>         Path to ONNX model              (default: models/inference_model.onnx)
  --input <path>         Input image or video            (default: test2.png)
  --output <path>        Output file                     (default: output.jpg)
  --mode <image|video|benchmark>
  --device <auto|cpu|cuda|tensorrt|tensorrt-rtx>         (default: auto)
  --precision <fp32|fp16|int8>                           (default: fp16)
  --int8-mode <cal|qdq|nocal>                            (default: nocal)
  --int8-table <path>    INT8 calibration table (when --int8-mode=cal)
  --cache-dir <path>     TensorRT engine/timing cache dir (default: trt_cache)
  --classes  "a,b,c"     Class-name CSV (default: body,head)
  --device-id <int>      GPU device id                   (default: 0)
  --batch <int>          Max batch size                  (default: 1)
  --iters <int>          Benchmark iterations            (default: 100)
  --warmup <int>         Benchmark warmup iterations     (default: 10)
  --conf <float>         Confidence threshold            (default: 0.5)
  --no-graph             Disable CUDA graph capture
  --no-gpu-pre           Disable GPU preprocessing
  --show                 Show result window (image/video mode)
  --verbose, -v          Verbose logging
  --help, -h             This help
)";
}

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out; std::string cur;
    for (char c : s) { if (c == sep) { if (!cur.empty()) out.push_back(cur); cur.clear(); } else cur.push_back(c); }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

bool parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { std::cerr << "Missing value for " << name << "\n"; std::exit(2); }
            return argv[++i];
        };
        if (k == "--help" || k == "-h") { print_help(); std::exit(0); }
        else if (k == "--model")        a.model     = need("--model");
        else if (k == "--input")        a.input     = need("--input");
        else if (k == "--output")       a.output    = need("--output");
        else if (k == "--mode")         a.mode      = need("--mode");
        else if (k == "--device")       a.device    = need("--device");
        else if (k == "--precision")    a.precision = need("--precision");
        else if (k == "--int8-mode")    a.int8_mode = need("--int8-mode");
        else if (k == "--int8-table")   a.int8_table= need("--int8-table");
        else if (k == "--cache-dir")    a.cache_dir = need("--cache-dir");
        else if (k == "--classes")      a.classes_csv = need("--classes");
        else if (k == "--device-id")    a.device_id = std::atoi(need("--device-id"));
        else if (k == "--batch")        a.batch     = std::atoi(need("--batch"));
        else if (k == "--iters")        a.iters     = std::atoi(need("--iters"));
        else if (k == "--warmup")       a.warmup    = std::atoi(need("--warmup"));
        else if (k == "--conf")         a.conf      = static_cast<float>(std::atof(need("--conf")));
        else if (k == "--no-graph")     a.no_graph  = true;
        else if (k == "--no-gpu-pre")   a.no_gpu_pre= true;
        else if (k == "--show")         a.show      = true;
        else if (k == "--verbose" || k == "-v") a.verbose = true;
        else { std::cerr << "Unknown arg: " << k << "\n"; return false; }
    }
    return true;
}

Device parse_device(const std::string& s) {
    if (s == "cpu")         return Device::CPU;
    if (s == "cuda")        return Device::CUDA;
    if (s == "tensorrt")    return Device::TensorRT;
    if (s == "tensorrt-rtx")return Device::TensorRTRTX;
    return Device::Auto;
}
Precision parse_prec(const std::string& s) {
    if (s == "fp32") return Precision::FP32;
    if (s == "int8") return Precision::INT8;
    return Precision::FP16;
}
Int8Mode parse_int8(const std::string& s) {
    if (s == "cal")   return Int8Mode::Calibrated;
    if (s == "qdq")   return Int8Mode::EmbeddedQDQ;
    return Int8Mode::NoCalibration;
}
const char* name_of(Device d) {
    switch (d) {
        case Device::CPU:         return "CPU";
        case Device::CUDA:        return "CUDA";
        case Device::TensorRT:    return "TensorRT";
        case Device::TensorRTRTX: return "TensorRT-RTX";
        case Device::Auto:        return "Auto";
    }
    return "?";
}
const char* name_of(Precision p) {
    switch (p) { case Precision::FP32: return "FP32"; case Precision::FP16: return "FP16"; case Precision::INT8: return "INT8"; }
    return "?";
}

EngineConfig make_config(const Args& a) {
    EngineConfig cfg;
    cfg.model_path = a.model;
    cfg.device     = parse_device(a.device);
    cfg.precision  = parse_prec(a.precision);
    cfg.int8_mode  = parse_int8(a.int8_mode);
    cfg.int8_calibration_table = a.int8_table;
    cfg.device_id  = a.device_id;
    cfg.max_batch_size = a.batch;
    cfg.enable_cuda_graph = !a.no_graph;
    cfg.enable_gpu_preprocess = !a.no_gpu_pre;
    cfg.trt_cache_dir = a.cache_dir;
    cfg.verbose = a.verbose;
    cfg.log_level = a.verbose ? LogLevel::Info : LogLevel::Warning;
    return cfg;
}

struct Stats {
    double mean = 0, stdev = 0, min_v = 0, max_v = 0, p50 = 0, p90 = 0, p99 = 0;
};
Stats compute(std::vector<double> v) {
    Stats s{};
    if (v.empty()) return s;
    double sum = 0; for (double x : v) sum += x;
    s.mean = sum / v.size();
    double sq = 0; for (double x : v) sq += (x - s.mean) * (x - s.mean);
    s.stdev = std::sqrt(sq / v.size());
    std::sort(v.begin(), v.end());
    s.min_v = v.front();
    s.max_v = v.back();
    auto pct = [&](double p){ return v[std::min(v.size() - 1, size_t(p * v.size())) ]; };
    s.p50 = pct(0.50);
    s.p90 = pct(0.90);
    s.p99 = pct(0.99);
    return s;
}

int run_image(const Args& a) {
    auto cfg = make_config(a);
    Engine engine(cfg);
    const auto names = split(a.classes_csv, ',');

    cv::Mat img = cv::imread(a.input);
    if (img.empty()) { std::cerr << "Could not read image: " << a.input << "\n"; return 1; }

    // Warmup
    for (int i = 0; i < std::max(1, a.warmup); ++i) (void)engine.infer(img, a.conf);

    auto dets = engine.infer(img, a.conf);
    auto t = engine.last_timings();

    std::cout << "Device   : " << name_of(engine.active_device()) << "\n";
    std::cout << "Precision: " << name_of(engine.active_precision()) << "\n";
    std::cout << "Timing   : " << std::fixed << std::setprecision(2)
              << t.preprocess_ms << " + " << t.inference_ms << " + "
              << t.postprocess_ms << " = " << t.total_ms() << " ms\n";
    std::cout << "Detections: " << dets.size() << "\n";
    for (size_t i = 0; i < dets.size(); ++i) {
        const auto& d = dets[i];
        std::string cn = (d.class_id < (int)names.size()) ? names[d.class_id] : ("cls_" + std::to_string(d.class_id));
        std::cout << "  [" << (i+1) << "] " << cn
                  << " (" << std::fixed << std::setprecision(1) << d.confidence * 100.f << "%) "
                  << "box=" << d.box.x << "," << d.box.y << "," << d.box.width << "x" << d.box.height << "\n";
    }

    cv::Mat vis = img.clone();
    draw_detections(vis, dets, names);
    if (!a.output.empty()) cv::imwrite(a.output, vis);
    if (a.show) { cv::imshow("RF-DETR", vis); cv::waitKey(0); }
    return 0;
}

int run_benchmark(const Args& a) {
    auto cfg = make_config(a);
    Engine engine(cfg);

    cv::Mat img = cv::imread(a.input);
    if (img.empty()) { std::cerr << "Could not read image: " << a.input << "\n"; return 1; }

    for (int i = 0; i < a.warmup; ++i) (void)engine.infer(img, a.conf);

    std::vector<double> total, pre, inf, post;
    total.reserve(a.iters); pre.reserve(a.iters); inf.reserve(a.iters); post.reserve(a.iters);

    for (int i = 0; i < a.iters; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        (void)engine.infer(img, a.conf);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto t  = engine.last_timings();
        total.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        pre.push_back(t.preprocess_ms);
        inf.push_back(t.inference_ms);
        post.push_back(t.postprocess_ms);
    }

    auto ts = compute(total), ps = compute(pre), is_ = compute(inf), os = compute(post);

    std::cout << "\n=== RF-DETR Benchmark ===\n";
    std::cout << "Device     : " << name_of(engine.active_device())
              << " / " << name_of(engine.active_precision()) << "\n";
    std::cout << "Iterations : " << a.iters << " (warmup " << a.warmup << ")\n";
    std::cout << "Image      : " << a.input << " (" << img.cols << "x" << img.rows << ")\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Stage        mean     stdev     p50      p90      p99      min      max  [ms]\n";
    auto row = [](const char* n, const Stats& s){
        std::cout << std::left << std::setw(12) << n
                  << std::right << std::setw(7) << s.mean
                  << std::setw(10) << s.stdev
                  << std::setw(9) << s.p50
                  << std::setw(9) << s.p90
                  << std::setw(9) << s.p99
                  << std::setw(9) << s.min_v
                  << std::setw(9) << s.max_v << "\n";
    };
    row("preprocess",  ps);
    row("inference",   is_);
    row("postprocess", os);
    row("total",       ts);
    std::cout << "FPS (mean) : " << (1000.0 / ts.mean) << "\n";
    return 0;
}

int run_video(const Args& a) {
    auto cfg = make_config(a);
    const auto names = split(a.classes_csv, ',');

    cv::VideoCapture cap(a.input);
    if (!cap.isOpened()) { std::cerr << "Could not open video: " << a.input << "\n"; return 1; }

    const int w   = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    const int h   = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    const double fps = cap.get(cv::CAP_PROP_FPS);

    cv::VideoWriter writer;
    if (!a.output.empty()) {
        int fourcc = cv::VideoWriter::fourcc('m','p','4','v');
        writer.open(a.output, fourcc, fps > 0 ? fps : 30.0, cv::Size(w, h));
    }

    PipelinedEngine pipe(cfg, /*queue_depth*/ 4);

    // Feeder thread
    std::thread feeder([&]{
        uint64_t id = 0;
        cv::Mat frame;
        while (cap.read(frame)) {
            pipe.submit(id++, frame.clone(), a.conf);
        }
        pipe.stop();
    });

    uint64_t received = 0;
    auto t_start = std::chrono::high_resolution_clock::now();
    while (auto r = pipe.next_result()) {
        ++received;
        // In the simple demo we re-read the source to draw on original frames we would need to
        // buffer them. For now the demo only reports counts + writes a placeholder draw on a black
        // frame so the pipeline is visibly working. For production use, buffer frames per-id.
        if (writer.isOpened() || a.show) {
            cv::Mat blank(h, w, CV_8UC3, cv::Scalar(0,0,0));
            draw_detections(blank, r->detections, names);
            if (writer.isOpened()) writer.write(blank);
            if (a.show) { cv::imshow("RF-DETR (pipeline)", blank); if (cv::waitKey(1) == 27) break; }
        }
    }
    feeder.join();

    auto dt = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_start).count();
    std::cout << "Processed " << received << " frames in " << dt << "s = "
              << (received / std::max(dt, 1e-9)) << " FPS\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse_args(argc, argv, a)) { print_help(); return 2; }

    try {
        if (a.mode == "image")     return run_image(a);
        if (a.mode == "benchmark") return run_benchmark(a);
        if (a.mode == "video")     return run_video(a);
        std::cerr << "Unknown mode: " << a.mode << "\n";
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "[rfdetr] Error: " << e.what() << "\n";
        return 1;
    }
}
