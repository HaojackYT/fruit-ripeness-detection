"""Quick import + smoke test for segmentation package.
Creates a synthetic image, runs segmentation, and writes an overlay to disk.
"""
import sys
import os
from pathlib import Path

# Make sure project root 'project' is importable
sys.path.insert(0, os.path.join(os.getcwd(), "project"))

from app.segmentation import segment_bgr, overlay_mask_on_bgr
import numpy as np
import cv2

out_dir = Path("project/app/segmentation/test_output")
out_dir.mkdir(parents=True, exist_ok=True)

# synthetic image: green background, red circle in center (simulates fruit)
img = np.full((256, 256, 3), fill_value=(40, 200, 40), dtype=np.uint8)
cv2.circle(img, (128, 128), 60, (0, 0, 200), thickness=-1)

mask = segment_bgr(img, config={"min_area": 100})
overlay = overlay_mask_on_bgr(img, mask, color=(0, 0, 255), alpha=0.5)

mask_path = out_dir / "mask.png"
overlay_path = out_dir / "overlay.jpg"
cv2.imwrite(str(mask_path), mask)
cv2.imwrite(str(overlay_path), overlay)

print("SMOKE-TEST: mask sum=", int(mask.sum()))
print("Wrote:", mask_path, overlay_path)
print("IMPORT_OK")
