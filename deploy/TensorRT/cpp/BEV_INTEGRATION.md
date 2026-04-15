# BEV mapping call location in `BYTETracker::update`

The BEV mapping is now called **inside `BYTETracker::update`**, right after:

```cpp
this->lost_stracks.assign(resb.begin(), resb.end());
```

Then:

```cpp
if (bev_renderer.get() != nullptr)
{
    bev_frame_index++;
    bev_renderer->renderFrame(this->tracked_stracks, this->lost_stracks, bev_frame_index);
}
```

So each frame maps directly from tracker state vectors:

- `vector<STrack> tracked_stracks`
- `vector<STrack> lost_stracks`

## Enable mapping from `bytetrack.cpp`

```cpp
BYTETracker tracker(fps, 30);
tracker.enable_bev_mapping("bev_outputs", -30.0f, 60.0f, 1000, 1000);
```

`STrack::gp` is used directly and expected as `std::vector<float>{x, y}`.
