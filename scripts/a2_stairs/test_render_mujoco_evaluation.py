"""CPU-only checks of trajectory-to-video sampling (no GL context required)."""
import unittest

import numpy as np

from render_mujoco_evaluation import end_detail, frame_indices, scene_for_terrain


class VideoSamplingTests(unittest.TestCase):
    def test_thirty_seconds_at_25_fps_is_750_frames(self):
        times = np.arange(1, 1501) * .02
        indices = frame_indices(times, 25)
        self.assertEqual(len(indices), 750)
        self.assertTrue((np.diff(indices) >= 0).all())
        self.assertGreaterEqual(indices.min(), 0)
        self.assertLess(indices.max(), len(times))
        requested = np.arange(750) / 25
        self.assertLessEqual(float(abs(times[indices] - requested).max()), .02000001)

    def test_floating_point_physics_time_does_not_add_frame(self):
        times = np.cumsum(np.full(1500, .02))
        self.assertEqual(len(frame_indices(times, 25)), 750)

    def test_short_episode_duration_rounds_up_only_one_frame(self):
        times = np.arange(1, 330) * .02
        indices = frame_indices(times, 25)
        self.assertEqual(len(indices), 165)
        self.assertGreaterEqual(len(indices) / 25, times[-1])
        self.assertLess(len(indices) / 25 - times[-1], 1 / 25)

    def test_single_record_has_valid_frame(self):
        np.testing.assert_array_equal(frame_indices(np.array([.02]), 25), [0])

    def test_flat_scene_is_not_stairs(self):
        self.assertEqual(scene_for_terrain("flat").name, "flat.xml")
        self.assertEqual(scene_for_terrain("stairs").name, "stairs_12.xml")
        self.assertTrue(scene_for_terrain("flat").is_file())

    def test_flat_end_card_does_not_claim_stairs_or_path_success(self):
        completed = end_detail({"terrain": "flat", "success": True})
        self.assertIn("Flat walking completed", completed)
        self.assertIn("straight-path accuracy not certified", completed)
        self.assertNotIn("Ascent", completed)
        self.assertIn("ended early", end_detail({"terrain": "flat", "success": False}))

    def test_stair_end_card_retains_standing_failure(self):
        self.assertIn("standing check NOT passed", end_detail({"terrain": "stairs", "success": False,
                                                               "ascent_completed": True}))


if __name__ == "__main__":
    unittest.main()
