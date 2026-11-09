"""Check GPU assignment and failure propagation without GPUs or model training."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class GPUSchedulerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bash = shutil.which('bash')
        if cls.bash is None:
            candidate = Path('C:/Program Files/Git/bin/bash.exe')
            cls.bash = str(candidate) if candidate.is_file() else None
        if cls.bash is None:
            raise unittest.SkipTest('Bash is unavailable')
        script = Path(__file__).with_name('train.sh').read_text(encoding='utf-8')
        cls.scheduler = script[script.index('\npids=()'):]

    def run_scheduler(self, pool, *, parallel=1, fail=False):
        # Lock directories make simultaneous jobs on the same GPU fail the test.
        preamble = '''set -euo pipefail
EXPERIMENT_ROOT=. RUN_ROOT=. GPU_FIXED=0 GPU_MATCHED=1
run_training() {
  mkdir "gpu_$3.lock" || return 9
  printf 'JOB %s %s %s\\n' "$1" "$2" "$3"
  sleep 0.03
  rmdir "gpu_$3.lock"
  if [[ "${FAIL_JOB:-0}" == 1 && "$1/$2" == interleaved3/fixed ]]; then return 7; fi
}
'''
        preamble += f'gpu_pool=({" ".join(map(str, pool))})\nPARALLEL={parallel}\nFAIL_JOB={int(fail)}\n'
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, 'scheduler.sh').write_text(preamble + self.scheduler, encoding='utf-8', newline='\n')
            result = subprocess.run([self.bash, 'scheduler.sh'], cwd=tmp, capture_output=True, text=True, timeout=20)
        jobs = [tuple(line.split()[1:]) for line in result.stdout.splitlines() if line.startswith('JOB ')]
        return result, jobs

    def test_gpu_pools(self):
        expected = [(split, mode) for split in ('negative3', 'interleaved3', 'spanning5')
                    for mode in ('fixed', 'matched')]
        for pool in ([0], [0, 2, 4], [0, 1, 2, 3, 4, 5]):
            with self.subTest(pool=pool):
                result, jobs = self.run_scheduler(pool)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertCountEqual(jobs, [(split, mode, str(pool[i % len(pool)]))
                                            for i, (split, mode) in enumerate(expected)])

    def test_legacy_pair_mode(self):
        result, jobs = self.run_scheduler([])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(jobs), 6)
        self.assertTrue(all(gpu == ('0' if mode == 'fixed' else '1') for _, mode, gpu in jobs))

    def test_serial_mode(self):
        result, jobs = self.run_scheduler([0, 2, 4], parallel=0)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([gpu for _, _, gpu in jobs], ['0', '2', '4', '0', '2', '4'])

    def test_failed_training_makes_script_fail(self):
        result, _ = self.run_scheduler([0, 1, 2], fail=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Training failed', result.stderr)


if __name__ == '__main__':
    unittest.main()
