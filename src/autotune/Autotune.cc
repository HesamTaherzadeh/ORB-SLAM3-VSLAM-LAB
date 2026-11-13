#include "autotune/Autotune.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <utility>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <glog/logging.h>
#include <nlohmann/json.hpp>

#include "cov_auto_tune/autotuner_config.h"
#include "cov_auto_tune/g2o/autotuner.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "OptimizableTypes.h"

namespace ORB_SLAM3
{
namespace autotune
{

nlohmann::json g_autotuneBAs = nlohmann::json::array();
std::map<std::string, Eigen::MatrixXd> map_of_label_to_covariance;
bool first_ba_ {true};

constexpr bool kInjectStereoPriorCovariance = true;
const std::array<Eigen::Matrix3d, 8> kInjectedStereoPriorCovariance = {
    (Eigen::Matrix3d() << 0.0314469005, 0.002091210622, 0.0206216124, 0.002091210622, 0.02729088488, 0.006535285937, 0.0206216124, 0.006535285937, 0.04723221464).finished(),
    (Eigen::Matrix3d() << 0.07184304266, -0.008367688954, 0.08018921142, -0.008367688954, 0.0508347109, -0.02190677893, 0.08018921142, -0.02190677893, 0.1283398326).finished(),
    (Eigen::Matrix3d() << 0.1250845353, -0.008401589428, 0.0877756768, -0.008401589428, 0.08543019314, -0.01181946057, 0.0877756768, -0.01181946057, 0.1131777498).finished(),
    (Eigen::Matrix3d() << 0.1921148773, -0.003418968975, 0.1476742599, -0.003418968975, 0.180372884, -0.0119341926, 0.1476742599, -0.0119341926, 0.2092585363).finished(),
    (Eigen::Matrix3d() << 0.2514426454, 0.0355595879, 0.1849882856, 0.0355595879, 0.3027468784, 0.02222430104, 0.1849882856, 0.02222430104, 0.2779799358).finished(),
    (Eigen::Matrix3d() << 0.4424429585, -0.03046504079, 0.4093511661, -0.03046504079, 0.3830644762, -0.06501813386, 0.4093511661, -0.06501813386, 0.5546131209).finished(),
    (Eigen::Matrix3d() << 0.751989317, -0.005349621388, 0.6718328119, -0.005349621388, 0.6961910351, -0.08614001829, 0.6718328119, -0.08614001829, 0.8663139485).finished(),
    (Eigen::Matrix3d() << 1.76564417, 0.1596446412, 1.669418929, 0.1596446412, 1.087848242, 0.1096829513, 1.669418929, 0.1096829513, 1.90218122).finished(),
};

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
        const bool injected = kInjectStereoPriorCovariance && dimension == 3 &&
                              octave >= 0 && octave < static_cast<int>(kInjectedStereoPriorCovariance.size());
        const char* prior_source;
        if (injected){
            groupConfig.prior_covariance_matrix = kInjectedStereoPriorCovariance[octave];
            prior_source = "injected";
        }else if (config.use_last_covariance_as_prior && it != map_of_label_to_covariance.end()){
            groupConfig.prior_covariance_matrix = it->second;
            prior_source = "map";
        }else{
            groupConfig.prior_covariance_matrix = meanCovariance;
            prior_source = "mean";
        }
        LOG(INFO) << "[Autotune.cc]prior " << label << " from=" << prior_source
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

EdgeGroups GroupByOctaveBand(
    const std::vector<g2o::OptimizableGraph::Edge*>& vpEdgesMono,
    const std::vector<int>& vnOctaveMono,
    const std::vector<g2o::OptimizableGraph::Edge*>& vpEdgesStereo,
    const std::vector<int>& vnOctaveStereo,
    const std::vector<std::vector<int>>& octaveBands)
{
    EdgeGroups groups;
    for(size_t i = 0; i < vpEdgesMono.size(); ++i)
        groups.mono[OctaveBandIndex(vnOctaveMono[i], octaveBands)].push_back(vpEdgesMono[i]);
    for(size_t i = 0; i < vpEdgesStereo.size(); ++i)
        groups.stereo[OctaveBandIndex(vnOctaveStereo[i], octaveBands)].push_back(vpEdgesStereo[i]);
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

// see docs d1.md 
nlohmann::json Objective(const cov_auto_tune::g2o::Groups& groups)
{
    double value = 0.0;
    nlohmann::json perGroup = nlohmann::json::array();
    for(const auto& [label, group] : groups)
    {
        if(group.information.size() == 0)
            continue;
        const Eigen::MatrixXd covariance = group.information.inverse();
        const double logDet = std::log(covariance.determinant());
        const double k = static_cast<double>(group.edges.size());
        const double ps = group.config.prior_strength;
        const double inner = (ps * k * group.config.prior_covariance_matrix).cwiseProduct(covariance).sum();   // <A, B>
        const double term = -(ps + 1.0) * k * logDet + inner;
        value += term;
        perGroup.push_back({{"group", label}, {"edges", group.edges.size()}, {"log_det", logDet},
                            {"inner_product", inner}, {"term", term}});
    }
    return {{"value", value}, {"groups", perGroup}};
}


nlohmann::json g_poseOnlySteps = nlohmann::json::array();    
long g_poseOnlyBA = -1;                                      
template <int Dim, typename EdgeT>
nlohmann::json TuneType(const std::vector<EdgeT*>& edges, const std::vector<int>& octaves,
                        const std::vector<float>& scale, unsigned long frameId, int stage)
{
    using Matrix = Eigen::Matrix<double, Dim, Dim>; // initialize the matrix 
    Matrix base = Matrix::Zero();
    int n = 0;
    for(size_t i = 0; i < edges.size(); ++i)
    {
        if(edges[i]->level() != 0) // do not use the outliers 
            continue;
        edges[i]->computeError(); // update cache 
        const auto& r = edges[i]->error(); // get residuals 
        base += r * r.transpose() / scale[octaves[i]]; // scale it by 1.2^(2octave) so a bigger octave gets smaller residual
        ++n;
    }
    if(n < 10)
    {
        LOG(WARNING) << "[POSE ONLY AUTOTUNE] ba=" << g_poseOnlyBA << " frame=" << frameId << " stage=" << stage
                     << " " << (Dim == 2 ? "mono" : "stereo") << ": only " << n << " inliers (< 10), base not updated";
        return {{"edges", n}, {"base", nullptr}};
    }
    base /= n;

    const Matrix information = base.inverse();
    for(size_t i = 0; i < edges.size(); ++i)
        edges[i]->setInformation(information / scale[octaves[i]]); // dividing the information matrix by the same value -- so a bigger octave gets less weight

    nlohmann::json rows = nlohmann::json::array();
    for(int r = 0; r < Dim; ++r)
    {
        nlohmann::json row = nlohmann::json::array();
        for(int c = 0; c < Dim; ++c)
            row.push_back(base(r, c));
        rows.push_back(row);
    }
    return {{"edges", n}, {"base", rows}};
}

} // namespace autotune

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
    autotune::g_autotuneBAs.push_back({{"ba", autotune::g_autotuneBAs.size()}, {"chi2", nlohmann::json::array()}, {"groups", nlohmann::json::array()}});
    auto& record = autotune::g_autotuneBAs.back();
    record["optimizations"] = nlohmann::json::array();
    record["objective"] = nlohmann::json::array();             // one entry per outer iteration
    const auto optimize = [&](const char* stage, int requested)
    {
        const int iterations = optimizer.optimize(requested);
        LOG(INFO) << "[Autotune.cc] optimize " << stage << " iterations=" << iterations << "/" << requested;
        const bool optimizationStop = pbStopFlag && *pbStopFlag;
        record["optimizations"].push_back({{"stage", stage}, {"requested", requested}, {"iterations", iterations},
                                           {"reached_stop_criterion", iterations > 0 && iterations < requested && !optimizationStop},
                                           {"tracking_stop_flag", optimizationStop}});
    };
    const autotune::EdgeGroups groups =
        autotune::GroupByOctaveBand(vpEdgesMono, vnOctaveMono, vpEdgesStereo, vnOctaveStereo, config.octave_bands);

    cov_auto_tune::AutotunerConfig tunerConfig = config;
    tunerConfig.default_group_covariance.prior_strength = tunerConfig.prior_strength;

    cov_auto_tune::g2o::Groups tunerGroups;
    autotune::RegisterGroups(groups.mono, 2, config, tunerGroups);
    autotune::RegisterGroups(groups.stereo, 3, config, tunerGroups);

    tunerConfig.grouping_method = cov_auto_tune::AutotunerConfig::GroupingMethod::ByCell;
    tunerConfig.allow_unregistered_edges = true;
    tunerConfig.verbose = false;               // nothing printed during the run
    tunerConfig.verbose_diagnostics = false;

    cov_auto_tune::g2o::Autotuner autotuner(&optimizer, tunerConfig, std::move(tunerGroups));

    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    autotune::RecordChi2ByType("initial", vpEdgesMono, vpEdgesStereo, config.huber_delta_mono, config.huber_delta_stereo);

    LOG(INFO) << "[Autotune.cc] BA start mono_edges=" << vpEdgesMono.size() << " stereo_edges=" << vpEdgesStereo.size()
              << " stop=" << (pbStopFlag && *pbStopFlag);
    if(config.pre_iterations > 0 && !(pbStopFlag && *pbStopFlag))
        optimize("pre", config.pre_iterations);
    LOG(INFO) << "[Autotune.cc]after pre stop=" << (pbStopFlag && *pbStopFlag);

    for(int outer = 0; outer < config.outer_iterations && !(pbStopFlag && *pbStopFlag); ++outer)
    {
        for(int t = 0; t < config.tune_iterations && !(pbStopFlag && *pbStopFlag); ++t)
        {
            autotuner.tune();
            LOG(INFO) << "[Autotune.cc] tune outer=" << outer << " t=" << t;
        }

        if(!(pbStopFlag && *pbStopFlag))
            optimize("inner", config.inner_iterations);

        nlohmann::json objective = autotune::Objective(autotuner.groups());
        optimizer.computeActiveErrors();                                  
        const double robustChi2 = optimizer.activeRobustChi2();
        objective["outer"] = outer;
        objective["active_robust_chi2"] = robustChi2;
        objective["total"] = objective["value"].get<double>() + robustChi2;
        LOG(INFO) << "[Autotune.cc] objective outer=" << outer << " groups=" << objective["value"].get<double>()
                  << " active_robust_chi2=" << robustChi2 << " total=" << objective["total"].get<double>();
        record["objective"].push_back(objective);
    }

    if(config.post_iterations > 0 && !(pbStopFlag && *pbStopFlag))
        optimize("post", config.post_iterations);

    optimizer.computeActiveErrors();
    autotune::RecordChi2ByType("final", vpEdgesMono, vpEdgesStereo, config.huber_delta_mono, config.huber_delta_stereo);

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
            autotune::map_of_label_to_covariance[label] = group.information.inverse();

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
    LOG(INFO) << "[Autotune.cc] BA end time_s=" << record["time_s"].get<double>();
}

void RunVanillaWithStats(
    g2o::SparseOptimizer& optimizer,
    const std::vector<g2o::OptimizableGraph::Edge*>& vpEdgesMono,
    const std::vector<g2o::OptimizableGraph::Edge*>& vpEdgesStereo,
    int iterations,
    double huberDeltaMono,
    double huberDeltaStereo,
    bool* pbStopFlag)
{
    const auto start = std::chrono::steady_clock::now();
    autotune::g_autotuneBAs.push_back({{"ba", autotune::g_autotuneBAs.size()}, {"chi2", nlohmann::json::array()},
                             {"groups", nlohmann::json::array()}, {"optimizations", nlohmann::json::array()}});
    auto& record = autotune::g_autotuneBAs.back();

    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    autotune::RecordChi2ByType("initial", vpEdgesMono, vpEdgesStereo, huberDeltaMono, huberDeltaStereo);

    const int done = optimizer.optimize(iterations);
    const bool optimizationStop = pbStopFlag && *pbStopFlag;
    record["optimizations"].push_back({{"stage", "vanilla"}, {"requested", iterations}, {"iterations", done},
                                       {"reached_stop_criterion", done > 0 && done < iterations && !optimizationStop},
                                       {"tracking_stop_flag", optimizationStop}});

    optimizer.computeActiveErrors();
    autotune::RecordChi2ByType("final", vpEdgesMono, vpEdgesStereo, huberDeltaMono, huberDeltaStereo);
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
             << " huber_delta=" << std::setprecision(4) << (std::string(type) == "mono" ? huberDeltaMono : huberDeltaStereo)
             << std::setprecision(2) << " above delta " << 100.0 * before["frac_above_huber"].get<double>() << "% -> "
             << 100.0 * after["frac_above_huber"].get<double>() << "%";
    }
    LOG(INFO) << line.str();
}

void RecordErased(size_t erasedMono, size_t erasedStereo, double chi2EraseMono, double chi2EraseStereo)
{
    if(autotune::g_autotuneBAs.empty())
        return;
    autotune::g_autotuneBAs.back()["erased"] = {{"mono", erasedMono}, {"stereo", erasedStereo},
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
    f << autotune::g_autotuneBAs.dump(2) << '\n';
}


void TunePoseOnly(
    unsigned long frameId,
    int stage, 
    const std::vector<EdgeSE3ProjectXYZOnlyPose*>& vpEdgesMono,
    const std::vector<int>& vnOctaveMono,
    const std::vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*>& vpEdgesStereo,
    const std::vector<int>& vnOctaveStereo,
    const std::vector<float>& vLevelSigma2)
{
    if(stage == 0)          // stage 0 = first of ORB's 4 rounds = a new PoseOptimization call (a frame can have several)
        ++autotune::g_poseOnlyBA;
    autotune::g_poseOnlySteps.push_back({{"ba", autotune::g_poseOnlyBA}, {"frame", frameId}, {"stage", stage},
                               {"mono", autotune::TuneType<2>(vpEdgesMono, vnOctaveMono, vLevelSigma2, frameId, stage)},
                               {"stereo", autotune::TuneType<3>(vpEdgesStereo, vnOctaveStereo, vLevelSigma2, frameId, stage)}});
}

void SavePoseOnlyStats(const std::string& path)
{
    std::ofstream f(path);
    if(!f.is_open())
    {
        LOG(ERROR) << "[AUTOTUNE] SavePoseOnlyStats: could not open " << path;
        return;
    }
    f << autotune::g_poseOnlySteps.dump(2) << '\n';
}

} // namespace ORB_SLAM3
