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

    vector<BEVTrackPoint> bev_tracks;
    bev_tracks.reserve(get_tracked_stracks().size() + get_lost_stracks().size());

    appendTracksToBevInput(get_tracked_stracks(), TrackState::Tracked, bev_tracks);
    appendTracksToBevInput(get_lost_stracks(), TrackState::Lost, bev_tracks);

    ++bev_frame_index_;
    bev_renderer_.renderFrame(bev_tracks, bev_frame_index_);

    return output_tracks;
}

const BirdEyeViewRenderer& BirdEyeViewTracker::renderer() const {
    return bev_renderer_;
}

BirdEyeViewRenderer& BirdEyeViewTracker::renderer() {
    return bev_renderer_;
}

void BirdEyeViewTracker::appendTracksToBevInput(const vector<STrack>& tracks,
                                                int expected_state,
                                                vector<BEVTrackPoint>& bev_tracks) const {
    for (size_t i = 0; i < tracks.size(); ++i) {
        if (tracks[i].state != expected_state) {
            continue;
        }

        BEVTrackPoint point;
        point.track_id = tracks[i].track_id;
        point.class_id = tracks[i].class_id;
        point.gp = tracks[i].gp;
        point.is_active = tracks[i].is_activated;
        point.track_state = tracks[i].state;
        bev_tracks.push_back(point);
    }
}
