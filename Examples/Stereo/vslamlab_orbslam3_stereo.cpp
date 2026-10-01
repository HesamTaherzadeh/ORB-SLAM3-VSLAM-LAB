#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<opencv2/core/core.hpp>

#include "sys/types.h"
#include "sys/sysinfo.h"

#include<System.h>
/**
 * Patched By Hesam
 */
#include "Optimizer.h"
#include "autotune/Autotune.h"
#include <glog/logging.h>

#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <stdexcept>

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
    /**
     * Patched By Hesam
     */
    string autotune_config{"-"};
    string vanilla_config{"-"};

    string vocabulary{"Vocabulary/ORBvoc.txt"};
    cout << endl;
    for (int i = 0; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("sequence_path:") != std::string::npos) {
            removeSubstring(arg, "sequence_path:");
            sequence_path =  arg;
            std::cout << "[vslamlab_orbslam3_stereo.cpp] Path to sequence = " << sequence_path << std::endl;
            continue;
        }
        if (arg.find("calibration_yaml:") != std::string::npos) {
            removeSubstring(arg, "calibration_yaml:");
            calibration_yaml =  arg;
            std::cout << "[vslamlab_orbslam3_stereo.cpp] Path to calibration.yaml = " << calibration_yaml << std::endl;
            continue;
        }
        if (arg.find("rgb_csv:") != std::string::npos) {
            removeSubstring(arg, "rgb_csv:");
            rgb_csv =  arg;
            std::cout << "[vslamlab_orbslam3_stereo.cpp] Path to rgb_csv = " << rgb_csv << std::endl;
            continue;
        }
        if (arg.find("exp_folder:") != std::string::npos) {
            removeSubstring(arg, "exp_folder:");
            exp_folder =  arg;
            std::cout << "[vslamlab_orbslam3_stereo.cpp] Path to exp_folder = " << exp_folder << std::endl;
            continue;
        }
        if (arg.find("exp_id:") != std::string::npos) {
            removeSubstring(arg, "exp_id:");
            exp_id =  arg;
            std::cout << "[vslamlab_orbslam3_stereo.cpp] Exp id = " << exp_id << std::endl;
            continue;
        }
        if (arg.find("settings_yaml:") != std::string::npos) {
            removeSubstring(arg, "settings_yaml:");
            settings_yaml =  arg;
            std::cout << "[vslamlab_orbslam3_stereo.cpp] Path to settings_yaml = " << settings_yaml << std::endl;
            continue;
        }
        if (arg.find("verbose:") != std::string::npos) {
            removeSubstring(arg, "verbose:");
            verbose = bool(std::stoi(arg));
            std::cout << "[vslamlab_orbslam3_stereo.cpp] Activate Visualization = " << verbose << std::endl;
            continue;
        }
        if (arg.find("vocabulary:") != std::string::npos) {
            removeSubstring(arg, "vocabulary:");
            vocabulary = arg;
            std::cout << "[vslamlab_orbslam3_stereo.cpp] Path to vocabulary = " << vocabulary << std::endl;
            continue;
        }
        /**
         * Patched By Hesam
         */
        if (arg.find("autotune_config:") != std::string::npos) {
            removeSubstring(arg, "autotune_config:");
            autotune_config = arg;
            std::cout << "[vslamlab_orbslam3_stereo.cpp] Path to autotune_config = " << autotune_config << std::endl;
            continue;
        }
        if (arg.find("vanilla_config:") != std::string::npos) {
            removeSubstring(arg, "vanilla_config:");
            vanilla_config = arg;
            std::cout << "[vslamlab_orbslam3_stereo.cpp] Path to vanilla_config = " << vanilla_config << std::endl;
            continue;
        }
    }

    /**
     * Patched By Hesam
     */
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;

    const bool autotune_given = (autotune_config != "-");
    const bool vanilla_given = (vanilla_config != "-");
    if (autotune_given == vanilla_given) {
        LOG(FATAL) << "Exactly one of autotune_config or vanilla_config must be supplied ("
                   << "autotune_config=" << autotune_config
                   << ", vanilla_config=" << vanilla_config << ")";
    }

    if (autotune_given) {
        ORB_SLAM3::Optimizer::mbLiveBAAutotune = true;
        try {
            ORB_SLAM3::Optimizer::msLiveBAConfig = ORB_SLAM3::LoadAutotuneConfig(autotune_config);
        } catch (const std::exception& e) {
            LOG(FATAL) << "Failed to load autotune config '" << autotune_config << "': " << e.what();
        }
    } else {
        try {
            ORB_SLAM3::Optimizer::msVanillaConfig = ORB_SLAM3::LoadVanillaConfig(vanilla_config);
        } catch (const std::exception& e) {
            LOG(FATAL) << "Failed to load vanilla config '" << vanilla_config << "': " << e.what();
        }
    }

    // Retrieve paths to images
    vector<string> imageFilenames_l{}, imageFilenames_r{};
    vector<ORB_SLAM3::Seconds> timestamps{};
    YAML::Node settings = YAML::LoadFile(settings_yaml);
    std::string cam0_name = settings["cam_stereo"].as<std::vector<std::string>>()[0];
    std::string cam1_name = settings["cam_stereo"].as<std::vector<std::string>>()[1];
    LoadImages(sequence_path, rgb_csv, imageFilenames_l, timestamps, imageFilenames_r, cam0_name, cam1_name);

    size_t nImages = imageFilenames_l.size();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(vocabulary, calibration_yaml, settings_yaml,
                           ORB_SLAM3::System::STEREO,
                           verbose);

    // Vector for tracking time statistics
    vector<ORB_SLAM3::Seconds> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

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

        if(ttrack < T)
            usleep((T-ttrack)  * 1e6);

    }

    // Stop all threads
    SLAM.Shutdown();

    /**
     * Patched By Hesam
     */
    if (ORB_SLAM3::Optimizer::mbLiveBAAutotune) {
        string clampStatsPath = exp_folder + "/" + paddingZeros(exp_id) + "_autotune_clamp_stats.csv";
        ORB_SLAM3::SaveAutotuneClampStatsCSV(clampStatsPath);
    }

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    ORB_SLAM3::Seconds totaltime = 0.0;
    for(int ni = 0; ni < nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

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

        ORB_SLAM3::Seconds t = static_cast<double>(std::stoll(t_str)) * 1e-9;

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
