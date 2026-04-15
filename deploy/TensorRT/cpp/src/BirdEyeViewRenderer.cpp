#include "BirdEyeViewRenderer.h"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>

namespace {
cv::Scalar hsvToBgr(float h_deg, float s, float v) {
    cv::Mat hsv(1, 1, CV_32FC3, cv::Scalar(h_deg, s, v));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    const cv::Vec3f color = bgr.at<cv::Vec3f>(0, 0);
    return cv::Scalar(color[0] * 255.0F, color[1] * 255.0F, color[2] * 255.0F);
}

float clamp01(float v) {
    return std::max(0.0F, std::min(1.0F, v));
}
}  // namespace

BirdEyeViewRenderer::BirdEyeViewRenderer(const std::string& output_dir,
                                         float y_min_m,
                                         float y_max_m,
                                         int canvas_width,
                                         int canvas_height,
                                         int padding_px,
                                         int grid_step_m)
    : output_dir_(output_dir),
      canvas_width_(canvas_width),
      canvas_height_(canvas_height),
      padding_px_(padding_px),
      grid_step_m_(grid_step_m) {
    world_range_.min_x = -40.0F;
    world_range_.max_x = 50.0F;
    world_range_.min_y = std::min(y_min_m, y_max_m);
    world_range_.max_y = std::max(y_min_m, y_max_m);

    ensureOutputDirectory(output_dir_);
}

void BirdEyeViewRenderer::renderFrame(const std::vector<STrack>& tracked_stracks,
                                      const std::vector<STrack>& lost_stracks,
                                      int frame_index) {
    cv::Mat canvas = buildCanvas();
    latest_rendered_tracks_.clear();
    latest_rendered_tracks_.reserve(tracked_stracks.size() + lost_stracks.size());

    for (size_t i = 0; i < tracked_stracks.size(); ++i) {
        drawTrackPoint(canvas, tracked_stracks[i], frame_index);
    }
    for (size_t i = 0; i < lost_stracks.size(); ++i) {
        drawTrackPoint(canvas, lost_stracks[i], frame_index);
    }

    const std::string output_path = frameOutputPath(frame_index);
    cv::imwrite(output_path, canvas);
}

cv::Point2i BirdEyeViewRenderer::worldToImage(float x_m, float y_m, bool* inside) const {
    const float x_span = world_range_.max_x - world_range_.min_x;
    const float y_span = world_range_.max_y - world_range_.min_y;
    const float usable_w = static_cast<float>(canvas_width_ - 2 * padding_px_);
    const float usable_h = static_cast<float>(canvas_height_ - 2 * padding_px_);

    const float nx = (x_m - world_range_.min_x) / x_span;
    const float ny = (y_m - world_range_.min_y) / y_span;

    const int px = static_cast<int>(std::round(padding_px_ + clamp01(nx) * usable_w));
    const int py = static_cast<int>(std::round(canvas_height_ - padding_px_ - clamp01(ny) * usable_h));

    const bool is_inside = (x_m >= world_range_.min_x && x_m <= world_range_.max_x && y_m >= world_range_.min_y &&
                            y_m <= world_range_.max_y);
    if (inside != nullptr) {
        *inside = is_inside;
    }

    return cv::Point2i(px, py);
}

const std::vector<BEVRenderedTrackInfo>& BirdEyeViewRenderer::latestRenderedTracks() const {
    return latest_rendered_tracks_;
}

bool BirdEyeViewRenderer::ensureOutputDirectory(const std::string& dir) const {
    if (dir.empty()) {
        return false;
    }

    std::string current;
    current.reserve(dir.size());
    for (size_t i = 0; i < dir.size(); ++i) {
        current.push_back(dir[i]);
        if (dir[i] == '/' || i + 1 == dir.size()) {
            if (current.empty() || current == "/") {
                continue;
            }
            if (mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
                return false;
            }
        }
    }
    return true;
}

std::string BirdEyeViewRenderer::frameOutputPath(int frame_index) const {
    std::ostringstream oss;
    oss << output_dir_ << "/bev_" << std::setfill('0') << std::setw(6) << frame_index << ".jpg";
    return oss.str();
}

cv::Mat BirdEyeViewRenderer::buildCanvas() const {
    cv::Mat canvas(canvas_height_, canvas_width_, CV_8UC3, cv::Scalar(245, 245, 245));
    drawGridAndAxes(canvas);
    return canvas;
}

void BirdEyeViewRenderer::drawGridAndAxes(cv::Mat& canvas) const {
    const cv::Point2i top_left(padding_px_, padding_px_);
    const cv::Point2i bottom_right(canvas_width_ - padding_px_, canvas_height_ - padding_px_);

    cv::rectangle(canvas, top_left, bottom_right, cv::Scalar(190, 190, 190), 1, cv::LINE_AA);

    for (int x = static_cast<int>(std::ceil(world_range_.min_x)); x <= static_cast<int>(std::floor(world_range_.max_x));
         x += grid_step_m_) {
        bool inside = false;
        const int px = worldToImage(static_cast<float>(x), world_range_.min_y, &inside).x;
        if (!inside && (x < world_range_.min_x || x > world_range_.max_x)) {
            continue;
        }
        cv::line(canvas,
                 cv::Point(px, padding_px_),
                 cv::Point(px, canvas_height_ - padding_px_),
                 cv::Scalar(220, 220, 220),
                 1,
                 cv::LINE_AA);
        cv::putText(canvas,
                    std::to_string(x) + "m",
                    cv::Point(px + 2, canvas_height_ - padding_px_ + 18),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.35,
                    cv::Scalar(100, 100, 100),
                    1,
                    cv::LINE_AA);
    }

    for (int y = static_cast<int>(std::ceil(world_range_.min_y)); y <= static_cast<int>(std::floor(world_range_.max_y));
         y += grid_step_m_) {
        bool inside = false;
        const int py = worldToImage(world_range_.min_x, static_cast<float>(y), &inside).y;
        if (!inside && (y < world_range_.min_y || y > world_range_.max_y)) {
            continue;
        }
        cv::line(canvas,
                 cv::Point(padding_px_, py),
                 cv::Point(canvas_width_ - padding_px_, py),
                 cv::Scalar(220, 220, 220),
                 1,
                 cv::LINE_AA);
        cv::putText(canvas,
                    std::to_string(y) + "m",
                    cv::Point(8, py - 2),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.35,
                    cv::Scalar(100, 100, 100),
                    1,
                    cv::LINE_AA);
    }

    cv::putText(canvas,
                "BEV map (meters)",
                cv::Point(padding_px_, 25),
                cv::FONT_HERSHEY_SIMPLEX,
                0.6,
                cv::Scalar(60, 60, 60),
                2,
                cv::LINE_AA);

    cv::putText(canvas,
                "x range: [-40, 50] m",
                cv::Point(canvas_width_ - 210, 25),
                cv::FONT_HERSHEY_SIMPLEX,
                0.45,
                cv::Scalar(80, 80, 80),
                1,
                cv::LINE_AA);
}

void BirdEyeViewRenderer::drawTrackPoint(cv::Mat& canvas, const STrack& track, int frame_index) {
    const float gp_x = track.gp.size() > 0 ? track.gp[0] : 0.0F;
    const float gp_y = track.gp.size() > 1 ? track.gp[1] : 0.0F;
    bool inside = false;
    const cv::Point2i pixel = worldToImage(gp_x, gp_y, &inside);
    const cv::Scalar color = getTrackColor(track.track_id);

    BEVRenderedTrackInfo rendered;
    rendered.frame_index = frame_index;
    rendered.track_id = track.track_id;
    rendered.class_id = track.class_id;
    rendered.gp = track.gp;
    rendered.bev_pixel = pixel;
    rendered.inside_bev = inside;
    rendered.track_state = track.track_state;
    latest_rendered_tracks_.push_back(rendered);

    if (!inside) {
        return;
    }

    if (track.track_state == TrackState::Lost) {
        cv::drawMarker(canvas, pixel, color, cv::MARKER_TILTED_CROSS, 14, 2, cv::LINE_AA);
    } else {
        cv::circle(canvas, pixel, 5, color, cv::FILLED, cv::LINE_AA);
        cv::circle(canvas, pixel, 9, color, 1, cv::LINE_AA);
    }

    const std::string class_text = resolveClassText(track);
    std::ostringstream oss;
    oss << "id:" << track.track_id;
    if (!class_text.empty()) {
        oss << " cls:" << class_text;
    }
    if (track.track_state == TrackState::Lost) {
        oss << " LOST";
    }

    cv::putText(canvas,
                oss.str(),
                cv::Point(pixel.x + 8, pixel.y - 8),
                cv::FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                2,
                cv::LINE_AA);
    cv::putText(canvas,
                oss.str(),
                cv::Point(pixel.x + 8, pixel.y - 8),
                cv::FONT_HERSHEY_SIMPLEX,
                0.42,
                cv::Scalar(255, 255, 255),
                1,
                cv::LINE_AA);
}

cv::Scalar BirdEyeViewRenderer::getTrackColor(int track_id) {
    const std::unordered_map<int, cv::Scalar>::const_iterator it = track_color_table_.find(track_id);
    if (it != track_color_table_.end()) {
        return it->second;
    }

    const int hue_seed = (track_id * 37) % 180;
    const cv::Scalar color = hsvToBgr(static_cast<float>(hue_seed), 0.85F, 0.95F);
    track_color_table_[track_id] = color;
    return color;
}

std::string BirdEyeViewRenderer::resolveClassText(const STrack& track) const {
    if (track.class_id >= 0) {
        return std::to_string(track.class_id);
    }
    return "";
}
