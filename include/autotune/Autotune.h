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

struct VanillaConfig
{
    double stop_criteria_threshold = 1e-3;
    int num_iterations = 10;
    int remove_mono = 0;  // remove or keep mono observations (both in full ba and local ba)
    int disable_huber = 0;  // Disable or enable huber (only in the final ba robust = bRobust && !disableHuber)
    bool fixed_cov_final_ba = false;  // fix only the one ba that is called in shutdown -- rest use orb model 
    bool fixed_cov_every_ba = false;  // fix for everything -- excluding pnp pose only
    bool fixed_cov_every_full_ba = false;  // fix for all of the full ba (needs fixed_cov_final_ba)
    bool fixed_cov_pose_optimization = false;  // fixes pnp
    bool fixed_cov_triangulation = false;  // CreateNewMapPoints reprojection gates (5.991 / 7.8)
    bool fixed_cov_fuse = false;  // Fuse reprojection gates (7.8 / 5.99)
    bool fixed_cov_epipolar = false;  // SearchForTriangulation epipolar gate: kp2 noise = fixed_cov_mono[octave]
    double huber_delta_mono = std::sqrt(5.991);  
    double huber_delta_stereo = std::sqrt(7.815); 
    double epipolar_chi2 = 3.84;  // 1-dof chi2 of the epipolar gate (3.84 = 95%), used with ORB or fixed cov
    std::array<Eigen::Matrix2d, 8> fixed_cov_mono{}; // here to be read from config 
    std::array<Eigen::Matrix3d, 8> fixed_cov_stereo{};  
};

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
