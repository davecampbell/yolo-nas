# Files Modified to Fix YOLO-NAS URL Issue

## Files Modified to Fix YOLO-NAS URL Issue

### 1. **File: `/Users/dave/projects/yolo-nas/venv/lib/python3.10/site-packages/super_gradients/training/pretrained_models.py`**

**Lines 47-49: Updated YOLO-NAS model URLs**

**Before:**
```python
"yolo_nas_s_coco": "https://sghub.deci.ai/models/yolo_nas_s_coco.pth",
"yolo_nas_m_coco": "https://sghub.deci.ai/models/yolo_nas_m_coco.pth",
"yolo_nas_l_coco": "https://sghub.deci.ai/models/yolo_nas_l_coco.pth",
```

**After:**
```python
"yolo_nas_s_coco": "https://sg-hub-nv.s3.amazonaws.com/models/yolo_nas_s_coco.pth",
"yolo_nas_m_coco": "https://sg-hub-nv.s3.amazonaws.com/models/yolo_nas_m_coco.pth",
"yolo_nas_l_coco": "https://sg-hub-nv.s3.amazonaws.com/models/yolo_nas_l_coco.pth",
```

**Purpose:** This file contains the `MODEL_URLS` dictionary that maps model names to their download URLs. I updated the three YOLO-NAS model entries to point to your new S3 bucket instead of the old `sghub.deci.ai` domain.

---

### 2. **File: `/Users/dave/projects/yolo-nas/venv/lib/python3.10/site-packages/super_gradients/training/utils/checkpoint_utils.py`**

**Line 1592: Updated URL parsing logic**

**Before:**
```python
unique_filename = url.split("https://sghub.deci.ai/models/")[1].replace("/", "_").replace(" ", "_")
```

**After:**
```python
unique_filename = url.split("https://sg-hub-nv.s3.amazonaws.com/models/")[1].replace("/", "_").replace(" ", "_")
```

**Purpose:** This line is part of the `load_pretrained_weights()` function that downloads and caches model weights. It extracts the filename from the URL to use as a local cache filename. The code was hard-coded to split on the old URL format, so I updated it to handle the new S3 bucket URL format.

---

## Summary of Changes

- **2 files modified** in the super-gradients package
- **4 lines changed total** (3 URLs + 1 URL parsing line)
- **Root cause:** The old `sghub.deci.ai` domain was no longer accessible
- **Solution:** Updated all references to use your new bucket `https://sg-hub-nv.s3.amazonaws.com/models/`

These changes ensure that when you call `models.get("yolo_nas_s", pretrained_weights="coco")`, the model will download weights from your new bucket instead of the broken old URL.
