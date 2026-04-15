# BirdEyeViewTracker integration example

`BirdEyeViewTracker` is a child class of `BYTETracker` and now feeds BEV rendering using
ByteTrack state containers directly:

- `vector<STrack> tracked_stracks`
- `vector<STrack> lost_stracks`

No object-level BEV adapter is needed.

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

Internally, after each `update(...)`, the subclass calls:

```cpp
bev_renderer_.renderFrame(get_tracked_stracks(), get_lost_stracks(), bev_frame_index_);
```

So BEV rendering operates directly on `STrack` instances (tracked + lost).
