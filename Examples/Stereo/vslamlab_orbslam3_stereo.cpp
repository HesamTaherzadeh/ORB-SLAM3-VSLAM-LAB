#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<opencv2/core/core.hpp>

#include "sys/types.h"
#include "sys/sysinfo.h"

#include<System.h>
#include "Optimizer.h"
#include "autotune/Autotune.h"
#include "cov_auto_tune/config_yaml.h"
#include <glog/logging.h>
#include <yaml-cpp/yaml.h>

#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <stdexcept>

// #define PRINT_CONFIG_ONLY  

using namespace std;
namespace ORB_SLAM3{
    using Seconds = double;
}

void LoadImages(const string &pathToSequence, const string &rgb_csv,
                vector<string> &imageFilenames_l, vector<ORB_SLAM3::Seconds> &timestamps,
                vector<string> &imageFilenames_r,
                const string cam0_name = "rgb_0", const string cam1_name = "rgb_1");
std::string paddingZeros(const std::string& number, const size_t numberOfZeros = 5);

void removeSubstring(std::string& str, const std::string& substring) {
    size_t pos;
    while ((pos = str.find(substring)) != std::string::npos) {
        str.erase(pos, substring.length());
    }
}



int main(int argc, char **argv)
{

    // VSLAM-LAB inputs
    string sequence_path;
    string calibration_yaml;
    string rgb_csv;
    string exp_folder;
    string exp_id{"0"};
    string settings_yaml{"orbslam2_settings.yaml"};
    bool verbose{true};
    string run_config{"-"};       // one yaml: cov: (mode of every stage), vanilla:, autotune:, solver:
    bool loop_closing{true};
    double playback_speed{1.0};

    string vocabulary{"Vocabulary/ORBvoc.txt"};
    for (int i = 0; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("sequence_path:") != std::string::npos) {
            removeSubstring(arg, "sequence_path:");
            sequence_path =  arg;
            LOG(INFO) << "[vslamlab_orbslam3_stereo.cpp] Path to sequence = " << sequence_path;
            continue;
        }
        if (arg.find("calibration_yaml:") != std::string::npos) {
            removeSubstring(arg, "calibration_yaml:");
            calibration_yaml =  arg;
            LOG(INFO) << "[vslamlab_orbslam3_stereo.cpp] Path to calibration.yaml = " << calibration_yaml;
            continue;
        }
        if (arg.find("rgb_csv:") != std::string::npos) {
            removeSubstring(arg, "rgb_csv:");
            rgb_csv =  arg;
            LOG(INFO) << "[vslamlab_orbslam3_stereo.cpp] Path to rgb_csv = " << rgb_csv;
            continue;
        }
        if (arg.find("exp_folder:") != std::string::npos) {
            removeSubstring(arg, "exp_folder:");
            exp_folder =  arg;
            LOG(INFO) << "[vslamlab_orbslam3_stereo.cpp] Path to exp_folder = " << exp_folder;
            continue;
        }
        if (arg.find("exp_id:") != std::string::npos) {
            removeSubstring(arg, "exp_id:");
            exp_id =  arg;
            LOG(INFO) << "[vslamlab_orbslam3_stereo.cpp] Exp id = " << exp_id;
            continue;
        }
        if (arg.find("settings_yaml:") != std::string::npos) {
            removeSubstring(arg, "settings_yaml:");
            settings_yaml =  arg;
            LOG(INFO) << "[vslamlab_orbslam3_stereo.cpp] Path to settings_yaml = " << settings_yaml;
            continue;
        }
        if (arg.find("verbose:") != std::string::npos) {
            removeSubstring(arg, "verbose:");
            verbose = bool(std::stoi(arg));
            LOG(INFO) << "[vslamlab_orbslam3_stereo.cpp] Activate Visualization = " << verbose;
            continue;
        }
        if (arg.find("vocabulary:") != std::string::npos) {
            removeSubstring(arg, "vocabulary:");
            vocabulary = arg;
            LOG(INFO) << "[vslamlab_orbslam3_stereo.cpp] Path to vocabulary = " << vocabulary;
            continue;
        }
        if (arg.find("run_config:") != std::string::npos) {
            removeSubstring(arg, "run_config:");
            run_config = arg;
            LOG(INFO) << "[vslamlab_orbslam3_stereo.cpp] Path to run_config = " << run_config;
            continue;
        }
        if (arg.find("loop_closing:") != std::string::npos) {
            removeSubstring(arg, "loop_closing:");
            loop_closing = std::stoi(arg) != 0;
            LOG(INFO) << "[vslamlab_orbslam3_stereo.cpp] loop_closing = " << loop_closing;
            continue;
        }
        if (arg.find("playback_speed:") != std::string::npos) {
            removeSubstring(arg, "playback_speed:");
            playback_speed = std::stod(arg);
            LOG(INFO) << "[vslamlab_orbslam3_stereo.cpp] playback_speed = " << playback_speed;
            continue;
        }
    }

    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 0;                       
    FLAGS_alsologtostderr = 0;
    FLAGS_timestamp_in_logfile_name = false;      
    const std::string glogPrefix = exp_folder + "/" + paddingZeros(exp_id) + "_glog_";
    for (const auto& [severity, name] : {std::pair{google::GLOG_INFO, "INFO"}, std::pair{google::GLOG_WARNING, "WARNING"},
                                         std::pair{google::GLOG_ERROR, "ERROR"}, std::pair{google::GLOG_FATAL, "FATAL"}}) {
        google::SetLogDestination(severity, (glogPrefix + name + ".txt").c_str());   
        google::SetLogSymlink(severity, "");                                        
    }

    ORB_SLAM3::System::mbLoopClosingEnabled = loop_closing;

    if (run_config == "-")
        LOG(FATAL) << "run_config must be supplied";

    try {
        // Path to sequence = /home/hesam/Autotune-studio/VSLAM-LAB-Benchmark/KITTI/04 the last will become '04'
        const std::string sequence = sequence_path.substr(sequence_path.find_last_of('/') + 1);
        ORB_SLAM3::Optimizer::msVanillaConfig = cov_auto_tune::vanillaConfigFromYamlFile(run_config, sequence);
        ORB_SLAM3::LocalMapping::msVanillaConfig = ORB_SLAM3::Optimizer::msVanillaConfig;
        const ORB_SLAM3::VanillaConfig& v = ORB_SLAM3::Optimizer::msVanillaConfig;
        ORB_SLAM3::Optimizer::mbLiveBAAutotune = (v.cov_local_ba == ORB_SLAM3::CovMode::Autotune);
        ORB_SLAM3::Optimizer::mbFinalBAAutotune = (v.cov_final_ba == ORB_SLAM3::CovMode::Autotune);
        if (v.anyAutotune())
            ORB_SLAM3::Optimizer::msLocalBAConfig = cov_auto_tune::configFromYamlFile(run_config);
    } catch (const std::exception& e) {
        LOG(FATAL) << "Failed to load run_config '" << run_config << "': " << e.what();
    }

    {   // what every stage uses: printed to stdout and the glog
        const ORB_SLAM3::VanillaConfig& v = ORB_SLAM3::Optimizer::msVanillaConfig;
        std::ostringstream modes;
        modes << "[MODES] run_config=" << run_config << " mbLiveBAAutotune=" << ORB_SLAM3::Optimizer::mbLiveBAAutotune
              << " mbFinalBAAutotune=" << ORB_SLAM3::Optimizer::mbFinalBAAutotune << (loop_closing ? "" : " (loop closing off: no gba)") << "\n";
        v.print(modes);
        const auto huberLine = [&](const char* level, const std::array<double, 2>& h, bool autotuned) {
            const std::array<float, 2> d = ORB_SLAM3::Optimizer::HuberDelta(h, autotuned);
            modes << "[HUBER] " << level << " mono=" << d[0] << " stereo=" << d[1]
                  << " (from " << (h[0] > 0.0 ? "huber:" : autotuned ? "autotune:" : "vanilla:") << ")\n";
        };
        huberLine("pose_only", v.huber_pose_only, false);
        huberLine("local_ba ", v.huber_local_ba, ORB_SLAM3::Optimizer::mbLiveBAAutotune);
        huberLine("gba      ", v.huber_gba, v.cov_gba == ORB_SLAM3::CovMode::Autotune);
        huberLine("final_ba ", v.huber_final_ba, ORB_SLAM3::Optimizer::mbFinalBAAutotune);
        if (v.anyAutotune()) {
            modes << "[MODES] autotune settings (autotuned BAs):\n";
            ORB_SLAM3::Optimizer::msLocalBAConfig.print(modes);
        }
        for (int o = 0; o < 8 && v.anyFixed(); ++o)
            modes << "[FIXED COV] octave=" << o << "\nmono cov=\n" << v.fixed_cov_mono[o] << "\nstereo cov=\n" << v.fixed_cov_stereo[o] << "\n";
        LOG(INFO) << modes.str();
        std::cout << modes.str() << std::flush;
    }
#ifdef PRINT_CONFIG_ONLY
    return 1;
#endif

    // Retrieve paths to images
    vector<string> imageFilenames_l{}, imageFilenames_r{};
    vector<ORB_SLAM3::Seconds> timestamps{};
    YAML::Node settings = YAML::LoadFile(settings_yaml);
    std::string cam0_name = settings["cam_stereo"].as<std::vector<std::string>>()[0];
    std::string cam1_name = settings["cam_stereo"].as<std::vector<std::string>>()[1];
    LoadImages(sequence_path, rgb_csv, imageFilenames_l, timestamps, imageFilenames_r, cam0_name, cam1_name);

    size_t nImages = imageFilenames_l.size();
    if (nImages == 0) {
        LOG(FATAL) << "No images loaded from rgb_csv '" << rgb_csv;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(vocabulary, calibration_yaml, settings_yaml,
                           ORB_SLAM3::System::STEREO,
                           verbose);

    // Vector for tracking time statistics
    vector<ORB_SLAM3::Seconds> vTimesTrack;
    vTimesTrack.resize(nImages);

    LOG(INFO) << "-------";
    LOG(INFO) << "Start processing sequence ...";
    LOG(INFO) << "Images in the sequence: " << nImages;

    // Main loop
    cv::Mat imLeft, imRight;
    for(size_t ni = 0; ni < nImages; ni++)
    {
        // Read image from file
        imLeft = cv::imread(imageFilenames_l[ni], cv::IMREAD_UNCHANGED);
        imRight = cv::imread(imageFilenames_r[ni], cv::IMREAD_UNCHANGED);
        ORB_SLAM3::Seconds tframe = timestamps[ni];

        // Pass the image to the SLAM system
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        SLAM.TrackStereo(imLeft,imRight,tframe);
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        ORB_SLAM3::Seconds ttrack = std::chrono::duration_cast<std::chrono::duration<ORB_SLAM3::Seconds> >(t2 - t1).count();
        vTimesTrack[ni] = ttrack;

        // Wait to load the next frame
        ORB_SLAM3::Seconds T = 0.0;
        if(ni < nImages-1)
            T = timestamps[ni+1] - tframe;
        else if(ni > 0)
            T = tframe - timestamps[ni-1];

        if(playback_speed > 0.0)
            T /= playback_speed;

        if(ttrack < T)
            usleep((T-ttrack)  * 1e6);

    }

    // Stop all threads
    SLAM.Shutdown();

    if (ORB_SLAM3::Optimizer::msVanillaConfig.cov_pose_only == ORB_SLAM3::CovMode::Autotune)
        ORB_SLAM3::SavePoseOnlyStats(exp_folder + "/" + paddingZeros(exp_id) + "_json_fullba.json");
    if (ORB_SLAM3::Optimizer::msVanillaConfig.anyAutotune()) {
        ORB_SLAM3::SaveAutotuneStats(exp_folder + "/" + paddingZeros(exp_id) + "_autotune.json");
    } else {
        ORB_SLAM3::SaveAutotuneStats(exp_folder + "/" + paddingZeros(exp_id) + "_vanilla.json");
    }

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    ORB_SLAM3::Seconds totaltime = 0.0;
    for(int ni = 0; ni < nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    LOG(INFO) << "-------";
    LOG(INFO) << "median tracking time: " << vTimesTrack[nImages/2];
    LOG(INFO) << "mean tracking time: " << totaltime/nImages;

    // Save camera trajectory
    string resultsPath_expId = exp_folder + "/" + paddingZeros(exp_id);
    SLAM.SaveKeyFrameTrajectoryVSLAMLAB(resultsPath_expId + "_" + "KeyFrameTrajectory.csv");

    return 0;
}

std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        // Simple trim for leading/trailing whitespace, often needed in real-world CSVs
        token.erase(0, token.find_first_not_of(" \t\n\r"));
        token.erase(token.find_last_not_of(" \t\n\r") + 1);
        tokens.push_back(token);
    }
    return tokens;
}

void LoadImages(const string &pathToSequence, const string &rgb_csv,
                vector<string> &imageFilenames_l, vector<ORB_SLAM3::Seconds> &timestamps,
                vector<string> &imageFilenames_r,
                const string cam0_name, const string cam1_name)
{

    imageFilenames_l.clear();
    timestamps.clear();
    imageFilenames_r.clear();

    std::ifstream in(rgb_csv);
    std::string line;

    // Read and map the header row to find indices
    if (!std::getline(in, line)) return; 
    if (!line.empty() && line.back() == '\r') line.pop_back();

    std::vector<std::string> headers = split(line, ',');
    std::map<std::string, int> col_map;
    for (size_t i = 0; i < headers.size(); ++i) {
        col_map[headers[i]] = i;
    }

    // Required headers
    const std::string header_ts = "ts_" + cam0_name + " (ns)";
    const std::string header_rgb0 = "path_" + cam0_name;
    const std::string header_rgb1 = "path_" + cam1_name;

    // Safely get indices
    auto get_index = [&](const std::string& key) -> int {
        return col_map[key];
    };

    int ts_idx = get_index(header_ts);
    int rgb0_idx = get_index(header_rgb0);
    int rgb1_idx = get_index(header_rgb1);
   

    // Read and process data lines using fixed indices
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (!line.empty() && line.back() == '\r') line.pop_back();

        std::vector<std::string> tokens = split(line, ',');
        
        // Assign variables using indices, regardless of column order
        std::string t_str = tokens[ts_idx];
        std::string rel_rgb0_path = tokens[rgb0_idx];
        std::string rel_rgb1_path = tokens[rgb1_idx];

        ORB_SLAM3::Seconds t = static_cast<double>(std::stoll(t_str)) * 1e-9; //Convert to ns to seconds

        timestamps.push_back(t);
        imageFilenames_l.push_back(pathToSequence + "/" + rel_rgb0_path);
        imageFilenames_r.push_back(pathToSequence + "/" + rel_rgb1_path);
    }
}

std::string paddingZeros(const std::string& number, const size_t numberOfZeros){
    std::string zeros{};
    for(size_t iZero{}; iZero < numberOfZeros - number.size(); ++iZero)
        zeros += "0";
    return (zeros + number);
}
