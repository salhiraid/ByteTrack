#include "BirdEyeViewTracker.h"

BirdEyeViewTracker::BirdEyeViewTracker(int frame_rate,
                                       int track_buffer,
                                       const std::string& bev_output_dir,
                                       float y_min_m,
                                       float y_max_m,
                                       int canvas_width,
                                       int canvas_height,
                                       int padding_px,
                                       int grid_step_m)
    : BYTETracker(frame_rate, track_buffer),
      bev_renderer_(bev_output_dir, y_min_m, y_max_m, canvas_width, canvas_height, padding_px, grid_step_m),
      bev_frame_index_(0) {
}

vector<STrack> BirdEyeViewTracker::update(const vector<Object>& objects) {
    vector<STrack> output_tracks = BYTETracker::update(objects);

    ++bev_frame_index_;
    bev_renderer_.renderFrame(get_tracked_stracks(), get_lost_stracks(), bev_frame_index_);

    return output_tracks;
}

const BirdEyeViewRenderer& BirdEyeViewTracker::renderer() const {
    return bev_renderer_;
}

BirdEyeViewRenderer& BirdEyeViewTracker::renderer() {
    return bev_renderer_;
}
