#ifndef ORB_SLAM3_AUTOTUNE_H
#define ORB_SLAM3_AUTOTUNE_H

#include <array>  
#include <cmath>
#include <string>
#include <vector>

#include <Eigen/Core>  

#include "cov_auto_tune/autotuner_config.h"
#include "Thirdparty/g2o/g2o/core/sparse_optimizer.h"
#include "Thirdparty/g2o/g2o/core/optimizable_graph.h"

namespace ORB_SLAM3
{

using VanillaConfig = cov_auto_tune::VanillaConfig;  // defined in autotune_core (cov_auto_tune/autotuner_config.h)
using CovMode = cov_auto_tune::CovMode;              // orb | fixed | autotune, one per stage (VanillaConfig::cov_*)

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

void RecordErased(size_t erasedMono, size_t erasedStereo, double chi2EraseMono, double chi2EraseStereo);

void RunVanillaWithStats(
    g2o::SparseOptimizer& optimizer,
    const std::vector<g2o::OptimizableGraph::Edge*>& vpEdgesMono,
    const std::vector<g2o::OptimizableGraph::Edge*>& vpEdgesStereo,
    int iterations,
    double huberDeltaMono,  
    double huberDeltaStereo,
    bool* pbStopFlag = nullptr);

} // namespace ORB_SLAM3

#endif // ORB_SLAM3_AUTOTUNE_H
