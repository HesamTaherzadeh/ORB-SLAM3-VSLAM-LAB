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

namespace g2o { class EdgeStereoSE3ProjectXYZOnlyPose; }

namespace ORB_SLAM3
{

class EdgeSE3ProjectXYZOnlyPose;

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

// pose-only BA, cov pose_only: autotune. Per type: base = sum_i r_i r_i^T / scale(o_i) / N over the inliers (level 0),
// scale(o) = 1.2^(2o) = mvLevelSigma2[o]; every edge gets covariance base * scale(o). Logs ba, stage and the two bases.
void TunePoseOnly(
    unsigned long frameId,
    int stage,
    const std::vector<EdgeSE3ProjectXYZOnlyPose*>& vpEdgesMono,
    const std::vector<int>& vnOctaveMono,
    const std::vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*>& vpEdgesStereo,
    const std::vector<int>& vnOctaveStereo,
    const std::vector<float>& vLevelSigma2);

void SavePoseOnlyStats(const std::string& path);

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
