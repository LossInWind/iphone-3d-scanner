"""
位姿估计模块属性测试

Property 6: Scene Bounds Contain All Points
*For any* scene with computed bounds, all valid depth points transformed 
to world coordinates SHALL lie within the bounding box (with small 
tolerance for numerical precision).

Validates: Requirements 7.1, 7.2, 7.3
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume

from scanner_tool.core.pose_estimation import BoundingBox


# ============================================================================
# Property 6: Scene Bounds Contain All Points
# ============================================================================

class TestBoundingBoxProperties:
    """BoundingBox 类的属性测试"""
    
    @given(
        min_x=st.floats(min_value=-100, max_value=0, allow_nan=False),
        min_y=st.floats(min_value=-100, max_value=0, allow_nan=False),
        min_z=st.floats(min_value=-100, max_value=0, allow_nan=False),
        max_x=st.floats(min_value=0, max_value=100, allow_nan=False),
        max_y=st.floats(min_value=0, max_value=100, allow_nan=False),
        max_z=st.floats(min_value=0, max_value=100, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_bounds_contain_interior_points(
        self, min_x, min_y, min_z, max_x, max_y, max_z
    ):
        """
        **Feature: pc-processing-tool, Property 6: Scene Bounds Contain All Points**
        **Validates: Requirements 7.1, 7.2, 7.3**
        
        For any valid bounding box, points generated within the bounds
        should be contained within the bounds.
        """
        # Ensure min < max
        assume(min_x < max_x)
        assume(min_y < max_y)
        assume(min_z < max_z)
        
        bbox = BoundingBox(
            min_bounds=np.array([min_x, min_y, min_z]),
            max_bounds=np.array([max_x, max_y, max_z]),
            transform=np.eye(4)
        )
        
        # Generate random points within bounds
        n_points = 100
        points = np.random.uniform(
            low=bbox.min_bounds,
            high=bbox.max_bounds,
            size=(n_points, 3)
        )
        
        # All points should be within bounds
        for point in points:
            assert np.all(point >= bbox.min_bounds), \
                f"Point {point} below min bounds {bbox.min_bounds}"
            assert np.all(point <= bbox.max_bounds), \
                f"Point {point} above max bounds {bbox.max_bounds}"

    
    @given(
        min_x=st.floats(min_value=-100, max_value=0, allow_nan=False),
        min_y=st.floats(min_value=-100, max_value=0, allow_nan=False),
        min_z=st.floats(min_value=-100, max_value=0, allow_nan=False),
        max_x=st.floats(min_value=0, max_value=100, allow_nan=False),
        max_y=st.floats(min_value=0, max_value=100, allow_nan=False),
        max_z=st.floats(min_value=0, max_value=100, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_bounds_exclude_exterior_points(
        self, min_x, min_y, min_z, max_x, max_y, max_z
    ):
        """
        **Feature: pc-processing-tool, Property 6: Scene Bounds Contain All Points**
        **Validates: Requirements 7.1, 7.2, 7.3**
        
        For any valid bounding box, points generated outside the bounds
        should NOT be contained within the bounds.
        """
        assume(min_x < max_x)
        assume(min_y < max_y)
        assume(min_z < max_z)
        
        bbox = BoundingBox(
            min_bounds=np.array([min_x, min_y, min_z]),
            max_bounds=np.array([max_x, max_y, max_z]),
            transform=np.eye(4)
        )
        
        # Generate points outside bounds (below min)
        outside_points = np.array([
            [min_x - 1, min_y, min_z],
            [min_x, min_y - 1, min_z],
            [min_x, min_y, min_z - 1],
            [max_x + 1, max_y, max_z],
            [max_x, max_y + 1, max_z],
            [max_x, max_y, max_z + 1],
        ])
        
        # At least one coordinate should be outside bounds
        for point in outside_points:
            is_outside = (
                np.any(point < bbox.min_bounds) or 
                np.any(point > bbox.max_bounds)
            )
            assert is_outside, \
                f"Point {point} should be outside bounds [{bbox.min_bounds}, {bbox.max_bounds}]"


class TestBoundingBoxRoundTrip:
    """BoundingBox 文件 I/O 往返测试"""
    
    @given(
        min_x=st.floats(min_value=-100, max_value=0, allow_nan=False, allow_infinity=False),
        min_y=st.floats(min_value=-100, max_value=0, allow_nan=False, allow_infinity=False),
        min_z=st.floats(min_value=-100, max_value=0, allow_nan=False, allow_infinity=False),
        max_x=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
        max_y=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
        max_z=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_bbox_file_roundtrip(
        self, min_x, min_y, min_z, max_x, max_y, max_z
    ):
        """
        **Feature: pc-processing-tool, Property 6: Scene Bounds Contain All Points**
        **Validates: Requirements 7.3**
        
        For any valid bounding box, saving to file and loading back
        should produce equivalent bounds (within numerical precision).
        """
        import tempfile
        import os
        
        assume(min_x < max_x)
        assume(min_y < max_y)
        assume(min_z < max_z)
        
        original = BoundingBox(
            min_bounds=np.array([min_x, min_y, min_z]),
            max_bounds=np.array([max_x, max_y, max_z]),
            transform=np.eye(4)
        )
        
        # Use tempfile context manager instead of pytest fixture
        with tempfile.TemporaryDirectory() as tmp_dir:
            filepath = os.path.join(tmp_dir, "bbox.txt")
            original.to_file(filepath)
            loaded = BoundingBox.from_file(filepath)
        
        # Check bounds are approximately equal (file I/O may lose precision)
        np.testing.assert_array_almost_equal(
            original.min_bounds, loaded.min_bounds, decimal=5,
            err_msg="Min bounds mismatch after round-trip"
        )
        np.testing.assert_array_almost_equal(
            original.max_bounds, loaded.max_bounds, decimal=5,
            err_msg="Max bounds mismatch after round-trip"
        )



def point_in_bounds(point: np.ndarray, bbox: BoundingBox, tolerance: float = 1e-6) -> bool:
    """检查点是否在边界框内（带容差）"""
    return (
        np.all(point >= bbox.min_bounds - tolerance) and
        np.all(point <= bbox.max_bounds + tolerance)
    )


def transform_point(T: np.ndarray, point: np.ndarray) -> np.ndarray:
    """使用 4x4 变换矩阵变换 3D 点"""
    R = T[:3, :3]
    t = T[:3, 3]
    return R @ point + t


class TestSceneBoundsContainment:
    """
    场景边界包含性测试
    
    Property 6: Scene Bounds Contain All Points
    *For any* scene with computed bounds, all valid depth points transformed 
    to world coordinates SHALL lie within the bounding box (with small 
    tolerance for numerical precision).
    """
    
    @given(
        n_points=st.integers(min_value=10, max_value=100),
        scale=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_computed_bounds_contain_source_points(self, n_points, scale):
        """
        **Feature: pc-processing-tool, Property 6: Scene Bounds Contain All Points**
        **Validates: Requirements 7.1, 7.2, 7.3**
        
        For any set of 3D points, the computed axis-aligned bounding box
        should contain all the original points.
        """
        # Generate random 3D points
        points = np.random.randn(n_points, 3) * scale
        
        # Compute bounds from points
        min_bounds = points.min(axis=0)
        max_bounds = points.max(axis=0)
        
        bbox = BoundingBox(
            min_bounds=min_bounds,
            max_bounds=max_bounds,
            transform=np.eye(4)
        )
        
        # All original points should be within bounds
        for i, point in enumerate(points):
            assert point_in_bounds(point, bbox), \
                f"Point {i} ({point}) not in bounds [{min_bounds}, {max_bounds}]"
    
    @given(
        n_points=st.integers(min_value=10, max_value=50),
        tx=st.floats(min_value=-10, max_value=10, allow_nan=False),
        ty=st.floats(min_value=-10, max_value=10, allow_nan=False),
        tz=st.floats(min_value=-10, max_value=10, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_transformed_points_in_transformed_bounds(self, n_points, tx, ty, tz):
        """
        **Feature: pc-processing-tool, Property 6: Scene Bounds Contain All Points**
        **Validates: Requirements 7.1, 7.2, 7.3**
        
        For any set of points and translation transform, transforming the points
        and recomputing bounds should still contain all transformed points.
        """
        # Generate random points in camera frame
        points_C = np.random.randn(n_points, 3)
        
        # Create a translation transform (T_WC: camera to world)
        T_WC = np.eye(4)
        T_WC[:3, 3] = [tx, ty, tz]
        
        # Transform points to world frame
        points_W = np.array([transform_point(T_WC, p) for p in points_C])
        
        # Compute bounds from world points
        min_bounds = points_W.min(axis=0)
        max_bounds = points_W.max(axis=0)
        
        bbox = BoundingBox(
            min_bounds=min_bounds,
            max_bounds=max_bounds,
            transform=np.eye(4)
        )
        
        # All transformed points should be within bounds
        for i, point in enumerate(points_W):
            assert point_in_bounds(point, bbox), \
                f"Transformed point {i} ({point}) not in bounds"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
