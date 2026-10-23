"""Run with: python -m unittest discover -s scripts/a2_stairs -p 'test_*.py'."""
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import evaluate_mujoco as ev


class EvaluationContractTests(unittest.TestCase):
    def test_geometry_matches_checked_in_scene_and_twelve_risers(self):
        model = mujoco.MjModel.from_xml_path(str(ev.SCENE))
        ev.validate_scene(model)
        self.assertAlmostEqual(ev.SPEC.top_height, 1.8)
        self.assertAlmostEqual(ev.SPEC.top_start, 3.7)
        self.assertAlmostEqual(ev.SPEC.top_end, 5.1)
        boxes = ev.geometry.build_stair_boxes(ev.SPEC)
        heights = [box.bounds[1][2] for box in boxes[:-1]]
        np.testing.assert_allclose(np.diff([0] + heights), 0.15, atol=1e-12)
        self.assertEqual(ET.tostring(ET.fromstring(ev.SCENE.read_text())),
                         ET.tostring(ET.fromstring(ev.scene_xml())))

    def test_platform_completion_needs_all_feet_load_height_and_upright(self):
        radius = np.full(4, .032)
        foot = np.array([[4.4, .2, 1.832], [4.4, -.2, 1.832], [3.9, .2, 1.832], [3.9, -.2, 1.832]])
        force = np.full(4, 50.)
        self.assertTrue(ev.settled_on_top(foot, radius, force, 0))
        bad = foot.copy(); bad[2, 0] = 3.7
        self.assertFalse(ev.settled_on_top(bad, radius, force, 0))
        bad = foot.copy(); bad[0, 2] += .04
        self.assertFalse(ev.settled_on_top(bad, radius, force, 0))
        self.assertFalse(ev.contact_on_top(bad, radius, force))
        self.assertFalse(ev.contact_on_top(foot, radius, [50, 50, 0, 50]))
        self.assertFalse(ev.settled_on_top(foot, radius, [50, 50, 0, 50], 0))
        self.assertFalse(ev.settled_on_top(foot, radius, force, 31))

    def test_stop_accepts_individual_top_contacts_but_not_unlanded_swing(self):
        radius = np.full(4, .032)
        foot = np.array([[4.4, .2, 1.832], [4.4, -.2, 1.832], [3.9, .2, 1.832], [3.9, -.2, 1.832]])
        seen = np.zeros(4, bool)
        self.assertFalse(ev.ready_to_stop(foot, radius, seen))
        for leg in range(4):
            force = np.zeros(4); force[leg] = 50
            seen |= ev.valid_top_contact(foot, radius, force)
            self.assertEqual(ev.ready_to_stop(foot, radius, seen), leg == 3)
        swing = foot.copy(); swing[0, 2] += .1
        self.assertTrue(ev.ready_to_stop(swing, radius, seen))
        retreat = foot.copy(); retreat[2, 2] -= .15
        self.assertFalse(ev.ready_to_stop(retreat, radius, seen))

    def test_width_exit_checks_foot_extent_and_only_course_region(self):
        radius = np.full(4, .032)
        foot = np.array([[2., 1.469, .4], [2., -.2, .4], [1., .2, .2], [1., -.2, .2]])
        self.assertTrue(ev.outside_course(foot, radius))
        foot[0, 1] = 1.467
        self.assertFalse(ev.outside_course(foot, radius))
        foot[0] = [.5, 2., .1]
        self.assertFalse(ev.outside_course(foot, radius))

    def test_training_tracker_matches_stop_and_settle_contract(self):
        import importlib.util
        import sys
        import torch
        path = ev.GEOMETRY_PATH.with_name("stair_logic.py")
        spec = importlib.util.spec_from_file_location("a2_test_stair_logic", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        tracker = module.StairCourseTracker(1, "cpu")
        foot = np.array([[4.4, .2, 1.832], [4.4, -.2, 1.832], [3.9, .2, 1.832], [3.9, -.2, 1.832]])
        radius = np.full(4, .032)
        seen = np.zeros(4, bool)
        specs = torch.tensor([[12, .15, .2, 1.5, 1.2, 3.]])
        root = torch.tensor([[4.2, 0, 2.2]])
        gravity = torch.tensor([[0., 0., -1.]])
        active = torch.tensor([True])
        def update(forces):
            tracker.update(torch.tensor(foot[None], dtype=torch.float32),
                           torch.tensor(np.tile(forces, (1, 10, 1)), dtype=torch.float32),
                           root, gravity, specs, active, .02)
        for leg in range(4):
            forces = np.zeros(4); forces[leg] = 50.
            seen |= ev.valid_top_contact(foot, radius, forces)
            update(forces)
            self.assertEqual(bool(tracker.stop[0]), ev.ready_to_stop(foot, radius, seen))
        self.assertFalse(bool(tracker.success[0]))
        for _ in range(19):
            update(np.full(4, 50.))
        self.assertFalse(bool(tracker.success[0]))
        update(np.full(4, 50.))
        self.assertTrue(bool(tracker.success[0]))
        self.assertTrue(ev.settled_on_top(foot, radius, np.full(4, 50.), 0))
        foot[0, 1] = 1.48
        update(np.full(4, 50.))
        self.assertTrue(ev.outside_course(foot, radius))
        self.assertTrue(bool(tracker.failed[0]))

    def test_ground_height_has_exact_rises_no_descent(self):
        xy = np.array([[1.49, 0], [1.501, 0], [1.701, 0], [3.701, 0], [5.101, 0], [3.8, 1.6]])
        np.testing.assert_allclose(ev.ground_height(xy), [0, .15, .3, 1.8, 0, 0])

    def test_settling_is_excluded_from_velocity_tracking(self):
        a = {"time": np.arange(1, 5), "base_pos": np.array([[1.6, 0, 0], [2.5, 0, 0], [3.3, 0, 0], [3.8, 0, 0]]),
             "command": np.array([[.5, 0, 0], [.5, 0, 0], [.5, 0, 0], [0, 0, 0]])}
        masks = ev.stage_masks(a, "stairs")
        np.testing.assert_array_equal(masks["ascent"], [True, True, True, False])
        np.testing.assert_array_equal(masks["settling"], [False, False, False, True])

    def test_phase_alignment_removes_shift_and_reflects_hip(self):
        n = 301
        phase = np.arange(n) % 40
        q = np.zeros((n, 4, 3))
        contact = np.zeros((n, 4))
        for leg in range(4):
            p = (phase + (20 if leg % 2 else 0)) % 40
            q[:, leg, :] = np.sin(2 * np.pi * p[:, None] / 40) * [.1, .3, .2]
            if leg % 2:
                q[:, leg, 0] *= -1
            contact[:, leg] = p < 20
        result = ev.phase_metrics({"joint_pos": q.reshape(n, 12), "stance_fraction": contact}, np.ones(n, bool), .02)
        np.testing.assert_allclose(result["front"]["waveform_rmse_deg"], 0, atol=1e-12)
        self.assertGreater(result["complete_cycles_per_leg"][0], 5)


if __name__ == "__main__":
    unittest.main()
