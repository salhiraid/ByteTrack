#pragma once

#include "BYTETracker.h"
#include "BirdEyeViewRenderer.h"

class BirdEyeViewTracker : public BYTETracker {
public:
    BirdEyeViewTracker(int frame_rate,
                       int track_buffer,
                       const std::string& bev_output_dir,
                       float y_min_m,
                       float y_max_m,
                       int canvas_width = 900,
                       int canvas_height = 900,
                       int padding_px = 50,
                       int grid_step_m = 10);

    vector<STrack> update(const vector<Object>& objects) override;

    const BirdEyeViewRenderer& renderer() const;
    BirdEyeViewRenderer& renderer();

private:
    void appendTracksToBevInput(const vector<STrack>& tracks,
                                int expected_state,
                                vector<BEVTrackPoint>& bev_tracks) const;

private:
    BirdEyeViewRenderer bev_renderer_;
    int bev_frame_index_;
};
