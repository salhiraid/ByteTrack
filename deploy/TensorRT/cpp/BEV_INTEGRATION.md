# BirdEyeViewTracker integration example

`BirdEyeViewTracker` is a child class of `BYTETracker` and feeds BEV rendering directly with:

- `vector<STrack> tracked_stracks`
- `vector<STrack> lost_stracks`

No object-level BEV input is used.

```cpp
#include "BirdEyeViewTracker.h"

BirdEyeViewTracker tracker(
    fps,
    30,
    "bev_outputs",  // output directory
    -30.0f,          // configurable y min [m]
    60.0f,           // configurable y max [m]
    1000,
    1000);

vector<STrack> output_stracks = tracker.update(objects);
```

## Ground point format in `STrack`

The BEV renderer reads `STrack::gp` directly, where `gp` is a `std::vector<float>`
of size 2 containing world coordinates:

- `gp[0]` -> `x` (meters)
- `gp[1]` -> `y` (meters)

Internally after each update:

```cpp
bev_renderer_.renderFrame(get_tracked_stracks(), get_lost_stracks(), bev_frame_index_);
```
