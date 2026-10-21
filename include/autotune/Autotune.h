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
    double correlationcoeff_uLuR = 0.0;   // correlation of uL and uR in the stereo covariance of the local BA (0 = original vanilla)
    int identity_information = 0;         // 1: every local-BA edge's information = I instead of ORB's octave weight (0 = original vanilla)
};

VanillaConfig LoadVanillaConfig(const std::string& path);

// writes one JSON record per local BA: chi2 before / after and the clamp flags of every group after the last tune()
void SaveAutotuneStats(const std::string& path);

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
