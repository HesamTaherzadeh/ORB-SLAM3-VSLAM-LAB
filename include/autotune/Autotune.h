/**
 * Patched By Hesam
 */

#ifndef ORB_SLAM3_AUTOTUNE_H
#define ORB_SLAM3_AUTOTUNE_H

#include <cmath>
#include <string>
#include <vector>

#include "cov_auto_tune/autotuner_config.h"
#include "Thirdparty/g2o/g2o/core/sparse_optimizer.h"
#include "Thirdparty/g2o/g2o/core/optimizable_graph.h"

namespace ORB_SLAM3
{

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
    const cov_auto_tune::AutotunerConfig& config = cov_auto_tune::AutotunerConfig(),
    bool* pbStopFlag = nullptr);

} // namespace ORB_SLAM3

#endif // ORB_SLAM3_AUTOTUNE_H
