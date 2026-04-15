# BirdEyeViewTracker integration example

The cleanest integration is to use `BirdEyeViewTracker`, a child class of `BYTETracker`.
It calls the original ByteTrack update logic, then renders **both tracked and lost tracks**
into BEV for each frame.

```cpp
#include "BirdEyeViewTracker.h"

// once before the frame loop
BirdEyeViewTracker tracker(
    fps,
    30,
    "bev_outputs",  // output directory
    -30.0f,          // configurable y min [m]
    60.0f,           // configurable y max [m]
    1000,
    1000);

// in the frame loop (same call pattern as BYTETracker)
vector<STrack> output_stracks = tracker.update(objects);
```

## Passing class and ground-point (`gp`) to ByteTrack tracks

`Object` now includes:

- `label` (class id)
- `gp` (`cv::Point3f`) for world coordinates `(x, y, z)` where `z = 0`

When detections are converted to `STrack`, class and `gp` are copied, so tracked/lost states
carry BEV information without changing ByteTrack association/matching behavior.

If your detector does not yet provide `gp`, set it before calling `update(...)`.
