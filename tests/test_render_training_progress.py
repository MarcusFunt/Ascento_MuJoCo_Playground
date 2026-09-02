import numpy as np
import pytest

from tools.render_training_progress import tile_frames


@pytest.mark.parametrize("frame_count", [1, 2, 3, 4, 5, 7, 11, 12])
def test_tile_frames_pads_partial_final_rows(frame_count):
    frames = [np.full((3, 5, 3), index, dtype=np.uint8) for index in range(frame_count)]

    preview = tile_frames(frames)

    expected_rows = (frame_count + 3) // 4
    assert preview.shape == (expected_rows * 3, 4 * 5, 3)
    for index, frame in enumerate(frames):
        row, column = divmod(index, 4)
        tile = preview[row * 3:(row + 1) * 3, column * 5:(column + 1) * 5]
        np.testing.assert_array_equal(tile, frame)

    if frame_count % 4:
        last_row = preview[(expected_rows - 1) * 3:]
        assert np.all(last_row[:, (frame_count % 4) * 5:] == 0)


def test_tile_frames_rejects_empty_input():
    with pytest.raises(ValueError, match="no frames"):
        tile_frames([])
