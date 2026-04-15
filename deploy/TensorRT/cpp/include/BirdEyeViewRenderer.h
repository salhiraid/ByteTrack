#pragma once

#include <opencv2/opencv.hpp>

#include <string>
#include <unordered_map>
#include <vector>

#include "STrack.h"

struct BEVRenderedTrackInfo {
    int frame_index = -1;
    int track_id = -1;
    int class_id = -1;
    cv::Point3f gp = cv::Point3f(0.0F, 0.0F, 0.0F);
    cv::Point2i bev_pixel = cv::Point2i(-1, -1);
    bool inside_bev = false;
    int track_state = TrackState::Tracked;
};

class BirdEyeViewRenderer {
public:
    BirdEyeViewRenderer(const std::string& output_dir,
                        float y_min_m,
                        float y_max_m,
                        int canvas_width = 900,
                        int canvas_height = 900,
                        int padding_px = 50,
                        int grid_step_m = 10);

    void renderFrame(const std::vector<STrack>& tracked_stracks,
                     const std::vector<STrack>& lost_stracks,
                     int frame_index);

    cv::Point2i worldToImage(float x_m, float y_m, bool* inside = nullptr) const;

    const std::vector<BEVRenderedTrackInfo>& latestRenderedTracks() const;

private:
    struct Range {
        float min_x;
        float max_x;
        float min_y;
        float max_y;
    };

    bool ensureOutputDirectory(const std::string& dir) const;
    std::string frameOutputPath(int frame_index) const;
    cv::Mat buildCanvas() const;
    void drawGridAndAxes(cv::Mat& canvas) const;
    void drawTrackPoint(cv::Mat& canvas, const STrack& track, int frame_index);
    cv::Scalar getTrackColor(int track_id);
    std::string resolveClassText(const STrack& track) const;

private:
    std::string output_dir_;
    Range world_range_;
    int canvas_width_;
    int canvas_height_;
    int padding_px_;
    int grid_step_m_;

    std::unordered_map<int, cv::Scalar> track_color_table_;
    std::vector<BEVRenderedTrackInfo> latest_rendered_tracks_;
};
