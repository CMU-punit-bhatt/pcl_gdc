# pcl_gdc
GDC based pointcloud alignment derived from PseudoLidar++ paper

-------------

```python
from gdc import GDC

# perfrom GDC on predicted and GT depth and obtain corrected depth
correctedDepth = GDC(pred_depth=predDepth, gt_depth=velo2depth, calib=calib)
```

see [test.py](test.py) for details on how to use GDC with Kitti RAW.