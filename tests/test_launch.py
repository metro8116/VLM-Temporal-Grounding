from pathlib import Path
import tempfile
import unittest

from vtg.launch import ROOT, build_command, load_config


class LaunchTests(unittest.TestCase):
    def test_native_abs_relative_are_controlled_pair(self):
        absolute = load_config(ROOT / "time_r1/configs/abs.json")
        relative = load_config(ROOT / "time_r1/configs/rel.json")
        self.assertEqual(absolute["trainer_args"], relative["trainer_args"])
        self.assertNotEqual(absolute["representation"], relative["representation"])

    def test_global_batch_and_attention_are_applied(self):
        config = load_config(ROOT / "timelens/configs/full_pro6000.json")
        cmd, env, cwd, args = build_command(config, "model", "data.json", "run", attention="sdpa")
        self.assertEqual(args["gradient_accumulation_steps"], 8)
        self.assertEqual(args["disable_flash_attn2"], True)
        self.assertIn("--nproc_per_node=8", cmd)
        self.assertEqual(cwd, ROOT / "timelens")
        self.assertIn(str(ROOT), env["PYTHONPATH"])

    def test_invalid_generation_batch_is_rejected(self):
        config = load_config(ROOT / "timelens/configs/full_pro6000.json")
        with self.assertRaises(ValueError):
            build_command(config, "model", "data.json", "run", gpus=1)

    def test_explicit_resume_and_weights_are_not_confused(self):
        config = load_config(ROOT / "time_r1/configs/abs.json")
        _, _, _, fresh = build_command(config, "model", "data.json", "run")
        _, _, _, resumed = build_command(config, "model", "data.json", "run", resume="checkpoint-50")
        self.assertIsNone(fresh["resume_checkpoint"])
        self.assertTrue(Path(resumed["resume_checkpoint"]).is_absolute())

    def test_local_model_is_resolved_before_changing_working_directory(self):
        config = load_config(ROOT / "time_r1/configs/abs.json")
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            relative = str(Path(folder).relative_to(Path.cwd()))
            _, _, _, args = build_command(config, relative, "data.json", "run")
            self.assertEqual(args["model_id"], str(Path(folder).resolve()))
        _, _, _, args = build_command(config, "Qwen/Qwen2.5-VL-7B-Instruct", "data.json", "run")
        self.assertEqual(args["model_id"], "Qwen/Qwen2.5-VL-7B-Instruct")


if __name__ == "__main__":
    unittest.main()
