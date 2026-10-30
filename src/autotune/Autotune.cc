#include "autotune/Autotune.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <glog/logging.h>
#include <nlohmann/json.hpp>
#include <opencv2/core/persistence.hpp>

#include "cov_auto_tune/autotuner_config.h"
#include "cov_auto_tune/g2o/autotuner.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"

namespace ORB_SLAM3
{
namespace
{

// One record per local BA, kept in memory during the run and written once at shutdown by SaveAutotuneStats:
// {"ba", "time_s",
//  "optimizations": [{stage (pre / inner / post), requested, iterations (done; 0 = solver failed),
//                     reached_stop_criterion (g2o stopped before the requested iterations on its own),
//                     tracking_stop_flag (Tracking's abort flag, pbStopFlag, was set after this optimize())}],
//  "chi2": [{stage (initial / final), type (mono / stereo), edges, mean_chi2, huber_delta, frac_above_huber}],
//  "groups": [{group, edges, after the last tune():
//              information (matrix written to the edges), cov_eigenvalues (of its inverse, ascending),
//              cov_eigenvalues_before_clamp, flags (per eigenvalue: -1 min_eig_cov, 0 free, +1 max_eig_cov)}],
//  "erased": {mono, stereo (observations erased after the BA), chi2_mono, chi2_stereo (the erase thresholds used)}}
nlohmann::json g_autotuneBAs = nlohmann::json::array();
std::map<std::string, Eigen::MatrixXd> map_of_label_to_covariance;  
bool first_ba_ {true};

template<typename T>
void ReadIfPresent(const cv::FileNode& node, const std::string& name, T& value)
{
    const cv::FileNode child = node[name];
    if(!child.empty())
        child >> value;
}

} // namespace

VanillaConfig LoadVanillaConfig(const std::string& path)
{
    if(!std::ifstream(path).good())
        throw std::runtime_error("LoadVanillaConfig: could not open " + path);

    cv::FileStorage fs(path, cv::FileStorage::READ);
    if(!fs.isOpened())
        throw std::runtime_error("LoadVanillaConfig: could not parse " + path);

    VanillaConfig config;

    cv::FileNode node = fs["vanilla"];
    if(node.empty())
        node = fs.root();

    ReadIfPresent(node, "stop_criteria_threshold", config.stop_criteria_threshold);
    ReadIfPresent(node, "num_iterations", config.num_iterations);
    ReadIfPresent(node, "correlationcoeff_uLuR", config.correlationcoeff_uLuR);
    ReadIfPresent(node, "identity_information", config.identity_information);
    ReadIfPresent(node, "remove_mono", config.remove_mono);
    if(config.correlationcoeff_uLuR <= -1.0 || config.correlationcoeff_uLuR >= 1.0)
        throw std::runtime_error("LoadVanillaConfig: correlationcoeff_uLuR must be in (-1, 1)");

    LOG(INFO) << "[VANILLA CONFIG] loaded from " << path;
    LOG(INFO) << "[VANILLA CONFIG] stop_criteria_threshold=" << config.stop_criteria_threshold
              << " num_iterations=" << config.num_iterations
              << " correlationcoeff_uLuR=" << config.correlationcoeff_uLuR
              << " identity_information=" << config.identity_information
              << " remove_mono=" << config.remove_mono;

    return config;
}

namespace
{

int OctaveBandIndex(int octave, const std::vector<std::vector<int>>& octaveBands)
{
    if(octaveBands.empty())
        return octave;
    for(size_t band = 0; band < octaveBands.size(); ++band)
        for(const int bandOctave : octaveBands[band])
            if(bandOctave == octave)
                return static_cast<int>(band);
    return -1;
}

void RegisterGroups(
    const std::map<int, std::vector<g2o::OptimizableGraph::Edge*>>& groups,
    int dimension,
    const cov_auto_tune::AutotunerConfig& config,
    cov_auto_tune::g2o::Groups& tunerGroups)
{
    int groupIndex = 0;

    for(const auto& entry : groups)
    {
        const int octave = entry.first;
        const std::vector<g2o::OptimizableGraph::Edge*>& groupEdges = entry.second;
        if(groupEdges.empty())
            continue;

        Eigen::MatrixXd meanCovariance, meanInformation;
        if(config.use_identity_prior)
        {
            meanInformation = Eigen::MatrixXd::Identity(dimension, dimension);
            meanCovariance = meanInformation;
        }
        else
        {
            Eigen::MatrixXd covarianceSum = Eigen::MatrixXd::Zero(dimension, dimension);
            for(const auto* edge : groupEdges)
            {
                const Eigen::Map<const Eigen::MatrixXd> information(edge->informationData(), dimension, dimension);
                covarianceSum += information.inverse();
            }
            meanCovariance = covarianceSum / static_cast<double>(groupEdges.size());
            meanCovariance = 0.5 * (meanCovariance + meanCovariance.transpose());
            meanInformation = meanCovariance.inverse();
        }

        std::string label = "residuals_dim_" + std::to_string(dimension) + "_octave_" + std::to_string(octave); 
        cov_auto_tune::g2o::Group& group = tunerGroups[label];


        cov_auto_tune::core::GroupCovarianceConfig& groupConfig = group.config;
        groupConfig.min_eig_cov = config.default_group_covariance.min_eig_cov;
        groupConfig.max_eig_cov = config.default_group_covariance.max_eig_cov;
        
        auto it = map_of_label_to_covariance.find(label);
        if (config.use_last_covariance_as_prior && it != map_of_label_to_covariance.end()){
            groupConfig.prior_covariance_matrix = it->second;
        }else{
            groupConfig.prior_covariance_matrix = meanCovariance;
        }
        LOG(INFO) << "[Autotune.cc]prior " << label << " from=" << (it != map_of_label_to_covariance.end() ? "map" : "mean")
                  << " rows=" << groupConfig.prior_covariance_matrix.rows() << " edges=" << groupEdges.size();

        groupConfig.prior_strength = config.prior_strength;

        group.edges = groupEdges;

        for(auto* edge : groupEdges)
        {
            Eigen::Map<Eigen::MatrixXd> information(edge->informationData(), dimension, dimension);
            information = meanInformation;
        }
        ++groupIndex;
    }
    first_ba_ = false;


    if(groupIndex == 0 && !groups.empty())
        LOG(ERROR) << "[AUTOTUNE] RegisterGroups registered nothing for dimension " << dimension
                   << " -- " << groups.size() << " octave bucket(s), all with zero edges";
}

struct EdgeGroups
{
    std::map<int, std::vector<g2o::OptimizableGraph::Edge*>> mono;
    std::map<int, std::vector<g2o::OptimizableGraph::Edge*>> stereo;
};

// Depth-quantile grouping (batch5), disabled: octave grouping is active again in GroupByOctaveBand below.
// // Sort by depth and cut into nBands groups of equal size; key 0 = nearest.
// std::map<int, std::vector<g2o::OptimizableGraph::Edge*>> GroupByDepthQuantile(
//     const std::vector<g2o::OptimizableGraph::Edge*>& edges, size_t nBands)
// {
//     std::vector<std::pair<double, g2o::OptimizableGraph::Edge*>> byDepth;
//     byDepth.reserve(edges.size());
//     for(auto* e : edges)
//     {
//         const auto* point = static_cast<const g2o::VertexSBAPointXYZ*>(e->vertex(0));
//         const auto* pose = static_cast<const g2o::VertexSE3Expmap*>(e->vertex(1));
//         byDepth.emplace_back(pose->estimate().map(point->estimate()).z(), e);
//     }
//     std::sort(byDepth.begin(), byDepth.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

//     std::map<int, std::vector<g2o::OptimizableGraph::Edge*>> groups;
//     for(size_t i = 0; i < byDepth.size(); ++i)
//         groups[i * nBands / byDepth.size()].push_back(byDepth[i].second);
//     return groups;
// }

EdgeGroups GroupByOctaveBand(
    const std::vector<g2o::OptimizableGraph::Edge*>& vpEdgesMono,
    const std::vector<int>& vnOctaveMono,
    const std::vector<g2o::OptimizableGraph::Edge*>& vpEdgesStereo,
    const std::vector<int>& vnOctaveStereo,
    const std::vector<std::vector<int>>& octaveBands)
{
    EdgeGroups groups;
    // batch16: one group per octave band of solver.octave_bands (one band with octaves 0-7 = one mono and one stereo group).
    // The default bands {0}, ..., {7} give one group per octave, as the per-octave grouping below did.
    for(size_t i = 0; i < vpEdgesMono.size(); ++i)
        groups.mono[OctaveBandIndex(vnOctaveMono[i], octaveBands)].push_back(vpEdgesMono[i]);
    for(size_t i = 0; i < vpEdgesStereo.size(); ++i)
        groups.stereo[OctaveBandIndex(vnOctaveStereo[i], octaveBands)].push_back(vpEdgesStereo[i]);

    // groups.mono = GroupByDepthQuantile(vpEdgesMono, octaveBands.size());
    // groups.stereo = GroupByDepthQuantile(vpEdgesStereo, octaveBands.size());

    // Per-octave grouping (batches 7-15), replaced by the octave bands above:
    // for(size_t i = 0; i < vpEdgesMono.size(); ++i)
    //     groups.mono[vnOctaveMono[i]].push_back(vpEdgesMono[i]);
    // for(size_t i = 0; i < vpEdgesStereo.size(); ++i)
    //     groups.stereo[vnOctaveStereo[i]].push_back(vpEdgesStereo[i]);
    return groups;
}

void RecordChi2ForType(
    const char* stage,
    const char* type,
    const std::vector<g2o::OptimizableGraph::Edge*>& edges,
    double huberDelta)
{
    if(edges.empty())
        return;
    double chi2Sum = 0.0;
    size_t aboveThreshold = 0;
    for(const auto* edge : edges)
    {
        const double chi2 = edge->chi2();
        chi2Sum += chi2;
        if(chi2 > huberDelta * huberDelta)
            ++aboveThreshold;
    }
    g_autotuneBAs.back()["chi2"].push_back({{"stage", stage}, {"type", type}, {"edges", edges.size()},
                                            {"mean_chi2", chi2Sum / static_cast<double>(edges.size())}, {"huber_delta", huberDelta},
                                            {"frac_above_huber", static_cast<double>(aboveThreshold) / static_cast<double>(edges.size())}});
}

void RecordChi2ByType(
    const char* stage,
    const std::vector<g2o::OptimizableGraph::Edge*>& vpEdgesMono,
    const std::vector<g2o::OptimizableGraph::Edge*>& vpEdgesStereo,
    double huberDeltaMono,
    double huberDeltaStereo)
{
    RecordChi2ForType(stage, "mono", vpEdgesMono, huberDeltaMono);
    RecordChi2ForType(stage, "stereo", vpEdgesStereo, huberDeltaStereo);
}

} // namespace

void RunLiveOctaveAutotune(
    g2o::SparseOptimizer& optimizer,
    const std::vector<g2o::OptimizableGraph::Edge*>& vpEdgesMono,
    const std::vector<int>& vnOctaveMono,
    const std::vector<g2o::OptimizableGraph::Edge*>& vpEdgesStereo,
    const std::vector<int>& vnOctaveStereo,
    const cov_auto_tune::AutotunerConfig& config,
    bool* pbStopFlag)
{
    const auto start = std::chrono::steady_clock::now();
    g_autotuneBAs.push_back({{"ba", g_autotuneBAs.size()}, {"chi2", nlohmann::json::array()}, {"groups", nlohmann::json::array()}});
    auto& record = g_autotuneBAs.back();
    record["optimizations"] = nlohmann::json::array();
    const auto optimize = [&](const char* stage, int requested)
    {
        const int iterations = optimizer.optimize(requested);
        const bool optimizationStop = pbStopFlag && *pbStopFlag;
        record["optimizations"].push_back({{"stage", stage}, {"requested", requested}, {"iterations", iterations},
                                           {"reached_stop_criterion", iterations > 0 && iterations < requested && !optimizationStop},
                                           {"tracking_stop_flag", optimizationStop}});
    };
    const EdgeGroups groups =
        GroupByOctaveBand(vpEdgesMono, vnOctaveMono, vpEdgesStereo, vnOctaveStereo, config.octave_bands);

    cov_auto_tune::AutotunerConfig tunerConfig = config;
    tunerConfig.default_group_covariance.prior_strength = tunerConfig.prior_strength;

    cov_auto_tune::g2o::Groups tunerGroups;
    RegisterGroups(groups.mono, 2, config, tunerGroups);
    RegisterGroups(groups.stereo, 3, config, tunerGroups);

    tunerConfig.grouping_method = cov_auto_tune::AutotunerConfig::GroupingMethod::ByCell;
    tunerConfig.allow_unregistered_edges = true;
    tunerConfig.verbose = false;               // nothing printed during the run
    tunerConfig.verbose_diagnostics = false;

    cov_auto_tune::g2o::Autotuner autotuner(&optimizer, tunerConfig, std::move(tunerGroups));

    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    RecordChi2ByType("initial", vpEdgesMono, vpEdgesStereo, config.huber_delta_mono, config.huber_delta_stereo);

    LOG(INFO) << "[Autotune.cc]BA start stop=" << (pbStopFlag && *pbStopFlag);
    if(config.pre_iterations > 0 && !(pbStopFlag && *pbStopFlag))
        optimize("pre", config.pre_iterations);
    LOG(INFO) << "[Autotune.cc]after pre stop=" << (pbStopFlag && *pbStopFlag);

    for(int outer = 0; outer < config.outer_iterations && !(pbStopFlag && *pbStopFlag); ++outer)
    {
        for(int t = 0; t < config.tune_iterations && !(pbStopFlag && *pbStopFlag); ++t)
        {
            autotuner.tune();
        }

        if(!(pbStopFlag && *pbStopFlag))
            optimize("inner", config.inner_iterations);
    }

    if(config.post_iterations > 0 && !(pbStopFlag && *pbStopFlag))
        optimize("post", config.post_iterations);

    optimizer.computeActiveErrors();
    RecordChi2ByType("final", vpEdgesMono, vpEdgesStereo, config.huber_delta_mono, config.huber_delta_stereo);

    for(const auto& [label, group] : autotuner.groups())
    {
        nlohmann::json information = nlohmann::json::array();
        for(int r = 0; r < group.information.rows(); ++r)
        {
            nlohmann::json row = nlohmann::json::array();
            for(int c = 0; c < group.information.cols(); ++c)
                row.push_back(group.information(r, c));
            information.push_back(row);
        }
        // string label = "residuals_dim_" + std::to_string(dimension) + "_octave_" + std::to_string(octave);

        LOG(INFO) << "[Autotune.cc]save " << label << " info_rows=" << group.information.rows() << " stop=" << (pbStopFlag && *pbStopFlag);

        if(group.information.size() > 0)
            map_of_label_to_covariance[label] = group.information.inverse();

        nlohmann::json covEigenvalues = nlohmann::json::array();
        for(size_t i = 0; i < group.covariance_before_clamp.size(); ++i)
        {
            if(group.clamp_flags[i] < 0)
                covEigenvalues.push_back(group.config.min_eig_cov);
            else if(group.clamp_flags[i] > 0)
                covEigenvalues.push_back(group.config.max_eig_cov);
            else
                covEigenvalues.push_back(group.covariance_before_clamp[i]);
        }
        record["groups"].push_back({{"group", label}, {"edges", group.edges.size()}, {"information", information},
                                    {"cov_eigenvalues", covEigenvalues}, {"cov_eigenvalues_before_clamp", group.covariance_before_clamp},
                                    {"flags", group.clamp_flags}});
    }
    record["time_s"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

void RunVanillaWithStats(
    g2o::SparseOptimizer& optimizer,
    const std::vector<g2o::OptimizableGraph::Edge*>& vpEdgesMono,
    const std::vector<g2o::OptimizableGraph::Edge*>& vpEdgesStereo,
    int iterations,
    bool* pbStopFlag)
{
    const auto start = std::chrono::steady_clock::now();
    const double huberDeltaMono = std::sqrt(5.991), huberDeltaStereo = std::sqrt(7.815);
    g_autotuneBAs.push_back({{"ba", g_autotuneBAs.size()}, {"chi2", nlohmann::json::array()},
                             {"groups", nlohmann::json::array()}, {"optimizations", nlohmann::json::array()}});
    auto& record = g_autotuneBAs.back();

    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    RecordChi2ByType("initial", vpEdgesMono, vpEdgesStereo, huberDeltaMono, huberDeltaStereo);

    const int done = optimizer.optimize(iterations);
    const bool optimizationStop = pbStopFlag && *pbStopFlag;
    record["optimizations"].push_back({{"stage", "vanilla"}, {"requested", iterations}, {"iterations", done},
                                       {"reached_stop_criterion", done > 0 && done < iterations && !optimizationStop},
                                       {"tracking_stop_flag", optimizationStop}});

    optimizer.computeActiveErrors();
    RecordChi2ByType("final", vpEdgesMono, vpEdgesStereo, huberDeltaMono, huberDeltaStereo);
    record["time_s"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    std::ostringstream line;
    line << "[VANILLA BA] ba=" << record["ba"] << " iterations=" << done << "/" << iterations << " stop_flag=" << optimizationStop;
    for(const char* type : {"mono", "stereo"})
    {
        nlohmann::json before, after;
        for(const auto& chi2 : record["chi2"])
            if(chi2["type"] == type)
                (chi2["stage"] == "initial" ? before : after) = chi2;
        if(before.is_null())
            continue;
        line << std::fixed << std::setprecision(2) << " | " << type << " edges=" << before["edges"]
             << " outliers " << 100.0 * before["frac_above_huber"].get<double>() << "% -> "
             << 100.0 * after["frac_above_huber"].get<double>() << "%";
    }
    LOG(INFO) << line.str();
}

void RecordErased(size_t erasedMono, size_t erasedStereo, double chi2EraseMono, double chi2EraseStereo)
{
    if(g_autotuneBAs.empty())
        return;
    g_autotuneBAs.back()["erased"] = {{"mono", erasedMono}, {"stereo", erasedStereo},
                                      {"chi2_mono", chi2EraseMono}, {"chi2_stereo", chi2EraseStereo}};
}

void SaveAutotuneStats(const std::string& path)
{
    std::ofstream f(path);
    if(!f.is_open())
    {
        LOG(ERROR) << "[AUTOTUNE] SaveAutotuneStats: could not open " << path;
        return;
    }
    f << g_autotuneBAs.dump(2) << '\n';
}

} // namespace ORB_SLAM3
