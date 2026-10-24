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

// observations the local BA erased after optimizing (chi2 over the erase threshold or negative depth), into the current BA's record
void RecordErased(size_t erasedMono, size_t erasedStereo, double chi2EraseMono, double chi2EraseStereo);

// vanilla local BA with the same per-BA record as the autotune one (chi2 initial / final, the optimize() call, no groups),
// frac_above_huber against ORB's erase thresholds (5.991 mono, 7.815 stereo); one [VANILLA BA] glog line per BA
void RunVanillaWithStats(
    g2o::SparseOptimizer& optimizer,
    const std::vector<g2o::OptimizableGraph::Edge*>& vpEdgesMono,
    const std::vector<g2o::OptimizableGraph::Edge*>& vpEdgesStereo,
    int iterations,
    bool* pbStopFlag = nullptr);

} // namespace ORB_SLAM3

#endif // ORB_SLAM3_AUTOTUNE_H
