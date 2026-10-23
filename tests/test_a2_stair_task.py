"""CPU-only tests for physical stair geometry, completion and cohort curriculum."""

import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import trimesh


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "source/robot_lab/robot_lab/tasks/a2"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


geometry = load_file("a2_stairs_geometry_test", TASK / "stair_geometry.py")
logic = load_file("a2_stairs_logic_test", TASK / "stair_logic.py")


def runtime_functions(*names):
    """Use the actual generator/wrapper functions without importing Isaac Sim."""
    tree = ast.parse((TASK / "mdp/stairs.py").read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = dict(np=np, torch=torch, trimesh=trimesh,
                 get_stair_spec=geometry.get_stair_spec, build_stair_boxes=geometry.build_stair_boxes,
                 curriculum_spec=geometry.curriculum_spec, validate_stair_boxes=geometry.validate_stair_boxes)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "stairs.py", "exec"), scope)
    return scope


class GeometryTests(unittest.TestCase):
    def test_target_has_twelve_risers_and_coplanar_landing(self):
        spec = geometry.get_stair_spec()
        boxes = geometry.build_stair_boxes(spec)
        geometry.validate_stair_boxes(spec, boxes)
        self.assertEqual(len(boxes), 13)
        self.assertAlmostEqual(spec.top_height, 1.8)
        self.assertAlmostEqual(spec.top_start, 3.7)
        self.assertAlmostEqual(spec.top_end, 5.1)
        self.assertAlmostEqual(spec.top_end - spec.top_start, 1.4)
        for i, x in enumerate(spec.riser_positions):
            self.assertAlmostEqual(geometry.height_at(spec, x + 1e-7) - geometry.height_at(spec, x - 1e-7), 0.15)
            self.assertAlmostEqual(geometry.height_at(spec, x + 0.10), (i + 1) * 0.15)

    def test_both_axes_and_every_curriculum_level_are_preserved(self):
        for level, n in enumerate((6, 8, 10, 12)):
            spec = geometry.curriculum_spec(level)
            self.assertEqual(spec.num_steps, n)
            self.assertEqual((spec.rise, spec.tread), (0.15, 0.20))
        for level, height in enumerate((0.12, 0.135, 0.15)):
            spec = geometry.curriculum_spec(level, mode="height")
            self.assertEqual((spec.num_steps, spec.tread, spec.rise), (12, 0.20, height))
        for rise, tread in ((0.14, 0.20), (0.16, 0.20), (0.15, 0.18), (0.15, 0.22)):
            spec = geometry.get_stair_spec(rise=rise, tread=tread)
            geometry.validate_stair_boxes(spec, geometry.build_stair_boxes(spec))

    def test_actual_mesh_callback_has_exact_surface_heights_and_treads(self):
        funcs = runtime_functions("spec_for_terrain", "straight_stairs_terrain")
        for mode, difficulty, n, rise, tread in (("fixed", 0.99, 12, .15, .20),
                                                ("fixed", 0.1, 12, .15, .18),
                                                ("steps", 0.3, 8, .15, .20)):
            cfg = SimpleNamespace(num_steps=12, rise=rise, tread=tread, approach=1.5,
                landing=1.2, width=3.0, start_x=1.0, size=(12., 9.), border_width=.5,
                curriculum_mode=mode, step_counts=(6, 8, 10, 12), heights=(.12, .135, .15))
            meshes, origin = funcs["straight_stairs_terrain"](difficulty, cfg)
            self.assertEqual(len(meshes), n + 2)  # floor, treads, landing
            np.testing.assert_allclose(origin, [1., 4.5, 0.])
            for i in range(n):
                mesh = meshes[i + 1]
                np.testing.assert_allclose(mesh.extents, [tread, 3., (i + 1) * rise], atol=1e-9)
                self.assertAlmostEqual(mesh.bounds[0, 0] - origin[0], 1.5 + i * tread)
                self.assertAlmostEqual(mesh.bounds[1, 2], (i + 1) * rise)

    def test_invalid_geometry_is_rejected(self):
        for kwargs in (dict(num_steps=0), dict(num_steps=3.5), dict(rise=-1), dict(tread=float("nan"))):
            with self.assertRaises(ValueError):
                geometry.get_stair_spec(**kwargs)

    def test_critic_contact_observation_keeps_original_sixteen_values(self):
        fn = runtime_functions("legacy_critic_contact_forces")["legacy_critic_contact_forces"]
        forces = torch.arange(2 * 4 * 5 * 3).reshape(2, 4, 5, 3).float()
        env = SimpleNamespace(scene=SimpleNamespace(sensors={"contact": SimpleNamespace(data=SimpleNamespace(net_forces_w_history=forces))}))
        cfg = SimpleNamespace(name="contact", body_ids=[0, 1, 3, 4])
        norms = torch.norm(forces[:, :3, cfg.body_ids], dim=-1)
        expected = torch.cat([norms.max(dim=1).values[:, None], norms], dim=1).flatten(1)
        actual = fn(env, cfg)
        self.assertEqual(actual.shape, (2, 16))
        torch.testing.assert_close(actual, expected)


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.tracker = logic.StairCourseTracker(1, "cpu")
        self.feet = torch.tensor([[[4.4, .25, 1.832], [4.4, -.25, 1.832],
                                   [3.9, .25, 1.832], [3.9, -.25, 1.832]]])
        self.forces = torch.full((1, 4, 4), 80.)
        self.root = torch.tensor([[4.1, 0., 2.2]])
        self.gravity = torch.tensor([[0., 0., -1.]])
        self.specs = torch.tensor([[12., .15, .20, 1.5, 1.2, 3.]])
        self.active = torch.tensor([True])

    def update(self):
        self.tracker.update(self.feet, self.forces, self.root, self.gravity, self.specs, self.active, .02)

    def test_success_requires_point_four_seconds_not_single_touch(self):
        for _ in range(19):
            self.update()
        self.assertTrue(self.tracker.stop.item())
        self.assertTrue(self.tracker.ascent_completed.item())
        self.assertFalse(self.tracker.success.item())
        self.update()
        self.assertTrue(self.tracker.success.item())
        self.assertAlmostEqual(self.tracker.contact_time_s.item(), .4, places=5)
        self.assertAlmostEqual(self.tracker.foot_contact_time_s.item(), 1.6, places=5)

    def test_alternating_contacts_allow_stop_but_airborne_rear_does_not(self):
        self.forces[:] = 0
        self.forces[:, :, :2] = 80
        self.feet[:, 2:, 2] += .10
        self.update()
        self.assertFalse(self.tracker.stop.item())
        self.feet[:, 2:, 2] -= .10
        self.forces[:] = 0
        self.forces[:, :, 2:] = 80
        self.update()
        self.assertTrue(self.tracker.stop.item())
        self.assertFalse(self.tracker.success.item())

    def test_wrong_height_and_missing_contacts_do_not_complete(self):
        self.feet[:, 2:, 2] -= .15
        for _ in range(25):
            self.update()
        self.assertFalse(self.tracker.stop.item())
        self.assertFalse(self.tracker.success.item())

    def test_contact_duration_uses_physics_samples(self):
        self.forces[:, :2] = 0
        self.update()
        self.assertAlmostEqual(self.tracker.contact_time_s.item(), .01, places=6)
        self.assertAlmostEqual(self.tracker.foot_contact_time_s.item(), .04, places=6)

    def test_width_sphere_boundary_and_tilt_are_failures(self):
        self.feet[0, 0, 1] = 1.48
        self.update()
        self.assertTrue(self.tracker.failed.item())
        self.tracker.reset(torch.tensor([0]))
        self.feet[0, 0, 1] = .25
        self.gravity[:] = torch.tensor([[1., 0., 0.]])
        self.update()
        self.assertTrue(self.tracker.failed.item())

    def test_flat_environments_never_complete_or_fail_a_course(self):
        self.active[:] = False
        self.feet[:, :, 1] = 5
        self.update()
        self.assertFalse(self.tracker.failed.item())
        self.assertFalse(self.tracker.success.item())
        self.assertEqual(self.tracker.contact_time_s.item(), 0.)


class CompletionModeTests(unittest.TestCase):
    def test_ascent_is_default_but_strict_mode_waits_for_settling(self):
        ascent = torch.tensor([True, True, False, True])
        settled = torch.tensor([False, True, False, True])
        failed = torch.tensor([False, False, False, True])
        self.assertEqual(logic.resolve_stair_completion(ascent, settled, failed).tolist(),
                         [True, True, False, False])
        self.assertEqual(logic.resolve_stair_completion(ascent, settled, failed, True).tolist(),
                         [False, True, False, False])

    def test_actual_termination_uses_selected_criterion(self):
        fn = runtime_functions("stair_completed")["stair_completed"]
        for strict, expected in ((False, True), (True, False)):
            command = SimpleNamespace(update_course=lambda: None,
                completed=lambda: logic.resolve_stair_completion(torch.tensor([True]),
                    torch.tensor([False]), torch.tensor([False]), strict))
            env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: command))
            self.assertEqual(fn(env).item(), expected)

    def test_actual_curriculum_uses_ascent_success_and_leaves_fixed_target_unchanged(self):
        fn = runtime_functions("stair_completion_curriculum")["stair_completion_curriculum"]
        for strict, expected in ((False, 1), (True, 0)):
            n = 101
            command = SimpleNamespace(curriculum_mask=torch.tensor([True]*100 + [False]),
                cohort=logic.StairCohortCurriculum(4, "cpu"),
                completed=lambda: logic.resolve_stair_completion(torch.ones(n, dtype=torch.bool),
                    torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool), strict))
            terrain = SimpleNamespace(terrain_levels=torch.zeros(n, dtype=torch.long),
                terrain_types=torch.zeros(n, dtype=torch.long), terrain_origins=torch.zeros((4, 1, 3)),
                env_origins=torch.zeros((n, 3)))
            env = SimpleNamespace(episode_length_buf=torch.ones(n, dtype=torch.long),
                command_manager=SimpleNamespace(get_term=lambda name: command),
                scene=SimpleNamespace(terrain=terrain))
            self.assertEqual(fn(env, torch.arange(n)), expected)
            self.assertTrue((terrain.terrain_levels[:100] == expected).all())
            self.assertEqual(terrain.terrain_levels[-1].item(), 0)


class CohortTests(unittest.TestCase):
    def test_pooled_hundred_episodes_ninety_percent_unlocks_next_level(self):
        cohort = logic.StairCohortCurriculum(4, "cpu")
        outcomes = torch.tensor([True] * 90 + [False] * 10)
        cohort.record(torch.zeros(50, dtype=torch.long), outcomes[:50])
        self.assertEqual(cohort.unlocked_level, 0)
        cohort.record(torch.zeros(50, dtype=torch.long), outcomes[50:])
        self.assertEqual(cohort.unlocked_level, 1)
        self.assertEqual(cohort.counts.tolist(), [100, 0, 0, 0])

    def test_low_success_does_not_unlock_and_latest_window_can_recover(self):
        cohort = logic.StairCohortCurriculum(4, "cpu")
        cohort.record(torch.zeros(100, dtype=torch.long), torch.zeros(100, dtype=torch.bool))
        self.assertEqual(cohort.unlocked_level, 0)
        cohort.record(torch.zeros(100, dtype=torch.long), torch.ones(100, dtype=torch.bool))
        self.assertEqual(cohort.unlocked_level, 1)

    def test_last_level_does_not_wrap_to_an_easy_level(self):
        cohort = logic.StairCohortCurriculum(4, "cpu")
        for level in range(4):
            cohort.record(torch.full((100,), level), torch.ones(100, dtype=torch.bool))
        self.assertEqual(cohort.unlocked_level, 3)


class TrainingStateTests(unittest.TestCase):
    def make_env(self):
        tree = ast.parse((TASK / "env/stairs_env.py").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        scope = {"torch": torch, "A2Env": object}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "stairs_env.py", "exec"), scope)
        env = scope["A2StairsEnv"]()
        env.num_envs, env.device = 4, "cpu"
        env.cfg = SimpleNamespace(stair_curriculum_mode="steps")
        term = SimpleNamespace(cohort=logic.StairCohortCurriculum(4, "cpu"),
                               cfg=SimpleNamespace(require_settled_completion=False))
        for name in ("completed_episodes", "successful_episodes", "settled_episodes", "ascent_episodes", "failed_episodes"):
            setattr(term, name, torch.zeros(4, dtype=torch.long))
        for name in ("assigned_seconds_total", "contact_seconds_total", "foot_contact_seconds_total"):
            setattr(term, name, torch.zeros(4))
        env.command_manager = SimpleNamespace(get_term=lambda name: term)
        origins = torch.arange(4*20*3).reshape(4, 20, 3).float()
        terrain = SimpleNamespace(terrain_levels=torch.tensor([0, 1, 2, 3]), max_terrain_level=4,
            terrain_types=torch.tensor([0, 8, 12, 16]), terrain_origins=origins, env_origins=torch.zeros((4, 3)))
        env.scene = SimpleNamespace(terrain=terrain)
        env.episode_length_buf = torch.ones(4, dtype=torch.long)
        env.reset_calls = 0
        def reset():
            env.reset_calls += 1
        env.reset = reset
        return env, term

    def test_checkpoint_restores_curriculum_counts_and_origins_without_live_episode(self):
        env, term = self.make_env()
        term.cohort.unlocked_level = 2
        term.cohort.counts[:] = torch.tensor([150, 100, 10, 0])
        term.completed_episodes[:] = torch.tensor([2, 4, 6, 8])
        term.contact_seconds_total[:] = torch.tensor([.2, .4, .6, .8])
        state = env.get_training_state()
        term.completed_episodes.zero_()
        env.scene.terrain.terrain_levels.zero_()
        env.set_training_state(state)
        self.assertEqual(term.completed_episodes.tolist(), [2, 4, 6, 8])
        self.assertEqual(env.scene.terrain.terrain_levels.tolist(), [0, 1, 2, 3])
        self.assertEqual(term.cohort.unlocked_level, 2)
        self.assertEqual(env.reset_calls, 1)
        self.assertEqual(env.episode_length_buf.sum().item(), 0)
        torch.testing.assert_close(env.scene.terrain.env_origins,
            env.scene.terrain.terrain_origins[env.scene.terrain.terrain_levels, env.scene.terrain.terrain_types])

    def test_changed_env_count_or_curriculum_rejects_resume(self):
        env, _ = self.make_env()
        state = env.get_training_state()
        state["num_envs"] = 8
        with self.assertRaises(ValueError):
            env.set_training_state(state)
        state["num_envs"] = 4
        state["curriculum_mode"] = "height"
        with self.assertRaises(ValueError):
            env.set_training_state(state)


    def test_completion_mode_is_saved_and_mismatch_rejects_resume(self):
        env, term = self.make_env()
        state = env.get_training_state()
        self.assertIs(state["require_settled_completion"], False)
        term.cfg.require_settled_completion = True
        with self.assertRaisesRegex(ValueError, "completion criterion"):
            env.set_training_state(state)
        term.cfg.require_settled_completion = False
        state["version"] = 1
        with self.assertRaises(ValueError):
            env.set_training_state(state)


if __name__ == "__main__":
    unittest.main()
