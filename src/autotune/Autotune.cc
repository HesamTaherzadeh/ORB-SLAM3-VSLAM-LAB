/**
 * Patched By Hesam
 */

#include "autotune/Autotune.h"

#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>

#include <glog/logging.h>
#include <opencv2/core/persistence.hpp>

#include "cov_auto_tune/autotuner_config.h"
#include "cov_auto_tune/g2o/autotuner.h"

namespace ORB_SLAM3
{
namespace
{

// {outer_iteration, tune_iteration, pct_low, pct_high} per Autotuner::tune() call.
std::vector<std::tuple<int, int, double, double>> g_autotuneClampStats;

template<typename T>
void ReadIfPresent(const cv::FileNode& node, const std::string& name, T& value)
{
    const cv::FileNode child = node[name];
    if(!child.empty())
        child >> value;
}

} // namespace

/**
 * Patched By Hesam
 */
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

    LOG(INFO) << "[VANILLA CONFIG] loaded from " << path;
    LOG(INFO) << "[VANILLA CONFIG] stop_criteria_threshold=" << config.stop_criteria_threshold
              << " num_iterations=" << config.num_iterations;

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
    cov_auto_tune::core::GroupRegistry& registry)
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

        cov_auto_tune::core::GroupCovarianceConfig groupConfig;
        groupConfig.min_eig_cov = config.default_group_covariance.min_eig_cov;
        groupConfig.max_eig_cov = config.default_group_covariance.max_eig_cov;
        groupConfig.prior_covariance_matrix = meanCovariance;
        groupConfig.prior_strength = config.prior_strength;

        const std::string groupLabel =
            "residuals_dim_" + std::to_string(dimension) + "_octave_" + std::to_string(octave);
        const cov_auto_tune::core::GroupKey groupKey =
            registry.makeGroupKeyForEdgeType(groupLabel, groupIndex, dimension);
        registry.registerGroup(groupKey, groupLabel, "", meanInformation);
        registry.setGroupConfig(groupKey, groupConfig);

        for(auto* edge : groupEdges)
        {
            Eigen::Map<Eigen::MatrixXd> information(edge->informationData(), dimension, dimension);
            information = meanInformation;
            cov_auto_tune::g2o::registerMeasurementGroup(registry, edge, groupKey);
        }
        ++groupIndex;
    }

    if(groupIndex == 0)
        LOG(ERROR) << "[AUTOTUNE] RegisterGroups registered nothing for dimension " << dimension
                   << " -- " << groups.size() << " octave bucket(s), all with zero edges";
}

struct EdgeGroups
{
    std::map<int, std::vector<g2o::OptimizableGraph::Edge*>> mono;
    std::map<int, std::vector<g2o::OptimizableGraph::Edge*>> stereo;
};

EdgeGroups GroupByOctaveBand(
    const std::vector<g2o::OptimizableGraph::Edge*>& vpEdgesMono,
    const std::vector<int>& vnOctaveMono,
    const std::vector<g2o::OptimizableGraph::Edge*>& vpEdgesStereo,
    const std::vector<int>& vnOctaveStereo,
    const std::vector<std::vector<int>>& octaveBands)
{
    EdgeGroups groups;
    // for(size_t i = 0; i < vpEdgesMono.size(); ++i)
    //     groups.mono[OctaveBandIndex(vnOctaveMono[i], octaveBands)].push_back(vpEdgesMono[i]);
    // for(size_t i = 0; i < vpEdgesStereo.size(); ++i)
    //     groups.stereo[OctaveBandIndex(vnOctaveStereo[i], octaveBands)].push_back(vpEdgesStereo[i]);

    for(size_t i = 0; i < vpEdgesMono.size(); ++i)
        groups.mono[vnOctaveMono[i]].push_back(vpEdgesMono[i]);
    for(size_t i = 0; i < vpEdgesStereo.size(); ++i)
        groups.stereo[vnOctaveStereo[i]].push_back(vpEdgesStereo[i]);
    return groups;
}

void LogChi2ForType(
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
    LOG(INFO) << "[AUTOTUNE CHI2] " << stage << " " << type
              << " edge_count=" << edges.size()
              << " mean_chi2=" << chi2Sum / static_cast<double>(edges.size())
              << " huber_delta=" << huberDelta
              << " fraction_above_huber=" << static_cast<double>(aboveThreshold) / static_cast<double>(edges.size());
}

void LogChi2ByType(
    const char* stage,
    const std::vector<g2o::OptimizableGraph::Edge*>& vpEdgesMono,
    const std::vector<g2o::OptimizableGraph::Edge*>& vpEdgesStereo,
    double huberDeltaMono,
    double huberDeltaStereo)
{
    LogChi2ForType(stage, "mono", vpEdgesMono, huberDeltaMono);
    LogChi2ForType(stage, "stereo", vpEdgesStereo, huberDeltaStereo);
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
    const EdgeGroups groups =
        GroupByOctaveBand(vpEdgesMono, vnOctaveMono, vpEdgesStereo, vnOctaveStereo, config.octave_bands);

    cov_auto_tune::AutotunerConfig tunerConfig = config;
    tunerConfig.default_group_covariance.prior_strength = tunerConfig.prior_strength;

    auto registry = std::make_shared<cov_auto_tune::core::GroupRegistry>(tunerConfig.default_group_covariance);

    RegisterGroups(groups.mono, 2, config, *registry);
    RegisterGroups(groups.stereo, 3, config, *registry);

    tunerConfig.grouping_method = cov_auto_tune::AutotunerConfig::GroupingMethod::ByCell;
    tunerConfig.allow_unregistered_edges = true;
    tunerConfig.group_registry = registry;
    tunerConfig.verbose_diagnostics = tunerConfig.verbose;

    cov_auto_tune::g2o::Autotuner autotuner(&optimizer, tunerConfig);

    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    LogChi2ByType("initial", vpEdgesMono, vpEdgesStereo, config.huber_delta_mono, config.huber_delta_stereo);

    if(config.pre_iterations > 0 && !(pbStopFlag && *pbStopFlag))
        optimizer.optimize(config.pre_iterations);

    for(int outer = 0; outer < config.outer_iterations && !(pbStopFlag && *pbStopFlag); ++outer)
    {
        for(int t = 0; t < config.tune_iterations && !(pbStopFlag && *pbStopFlag); ++t)
        {
            autotuner.tune();
            const cov_auto_tune::g2o::ClampHitPercentages pct = autotuner.lastClampHitPercentages();
            g_autotuneClampStats.emplace_back(outer, t, pct.lower_pct, pct.upper_pct);
        }
        
        autotuner.printGroups();

        if(!(pbStopFlag && *pbStopFlag))
            optimizer.optimize(config.inner_iterations);
    }

    if(config.post_iterations > 0 && !(pbStopFlag && *pbStopFlag))
        optimizer.optimize(config.post_iterations);

    optimizer.computeActiveErrors();
    LogChi2ByType("final", vpEdgesMono, vpEdgesStereo, config.huber_delta_mono, config.huber_delta_stereo);
}

void SaveAutotuneClampStatsCSV(const std::string& path)
{
    std::ofstream f(path);
    if(!f.is_open())
    {
        LOG(ERROR) << "[AUTOTUNE] SaveAutotuneClampStatsCSV: could not open " << path;
        return;
    }

    f << "outer,tune,pct_low,pct_high\n";
    for(const auto& [outer, tune, pct_low, pct_high] : g_autotuneClampStats)
        f << outer << ',' << tune << ',' << pct_low << ',' << pct_high << '\n';

    LOG(INFO) << "[AUTOTUNE] Saved " << g_autotuneClampStats.size() << " clamp-stat rows to " << path;
}

} // namespace ORB_SLAM3
