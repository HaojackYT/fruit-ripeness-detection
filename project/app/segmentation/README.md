Segmentation module

This module provides simple HSV-based segmentation utilities for Fruit Ripeness Detection.

Functions:
- `segment_bgr(img, config)`: segment from BGR uint8 image (common OpenCV format)
- `segment_hsv(hsv, config)`: segment from HSV float image (H in degrees, S/V in [0,1])
- `mask_to_bboxes(mask)`, `get_largest_bbox(mask)`
- `overlay_mask_on_bgr(img, mask)`

CLI:
```
python -m app.segmentation.cli --image path/to/img.jpg --output-mask mask.png --output-overlay overlay.jpg
```

Notes for frontend integration:
- The segmentation accepts raw bytes or BGR arrays; if your frontend sends base64, decode into bytes and send to `segment_from_bytes` or read into OpenCV via `cv2.imdecode`.
- Configuration options (thresholds, morphological kernel, min_area) can be passed as a dict or JSON file to the CLI.
