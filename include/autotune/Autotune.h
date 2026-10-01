/**
 * Patched By Hesam
 */

#ifndef ORB_SLAM3_AUTOTUNE_H
#define ORB_SLAM3_AUTOTUNE_H

#include <cmath>
#include <string>
#include <vector>

#include "Thirdparty/g2o/g2o/core/sparse_optimizer.h"
#include "Thirdparty/g2o/g2o/core/optimizable_graph.h"

namespace ORB_SLAM3
{

struct AutotuneConfig
{
    int outer_iterations = 2;
    int inner_iterations = 10;
    int tune_iterations = 1;
    int pre_iterations = 0;
    int post_iterations = 0;
    bool use_wishart_prior = true;
    bool use_k_as_denom = false;
    bool diagonal_constraint = true;
    double min_eig_cov = 1.0e-3;
    double max_eig_cov = 100.0;
    double prior_strength = 0.1;
    bool use_identity_prior = false;  // skip the empirical mean-covariance step; identity for both initial edge information and the prior
    std::vector<std::vector<int>> octave_bands = {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}};

    bool verbose = false;

    double huber_delta_mono = std::sqrt(5.991);
    double huber_delta_stereo = std::sqrt(7.815);
};


AutotuneConfig LoadAutotuneConfig(const std::string& path);

/**
 * Patched By Hesam
 */
struct VanillaConfig
{
    double stop_criteria_threshold = 1e-3;
    int num_iterations = 10;
};

VanillaConfig LoadVanillaConfig(const std::string& path);

void SaveAutotuneClampStatsCSV(const std::string& path);

void RunLiveOctaveAutotune(
    g2o::SparseOptimizer& optimizer,
    const std::vector<g2o::OptimizableGraph::Edge*>& vpEdgesMono,
    const std::vector<int>& vnOctaveMono,
    const std::vector<g2o::OptimizableGraph::Edge*>& vpEdgesStereo,
    const std::vector<int>& vnOctaveStereo,
    const AutotuneConfig& config = AutotuneConfig(),
    bool* pbStopFlag = nullptr);

} // namespace ORB_SLAM3

#endif // ORB_SLAM3_AUTOTUNE_H
