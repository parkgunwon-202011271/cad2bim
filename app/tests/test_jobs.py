"""Standard-library checks for file handoff; no conversion engine required."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import jobs


class JobHandoffTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_folder_and_result_resolution(self):
        folder = self.root / 'drawing'
        folder.mkdir()
        self.assertEqual(jobs.prepare_input([str(folder)]), folder.resolve())
        output = folder / jobs.OUT_DIR_NAME
        output.mkdir()
        (output / 'building.dae').write_text('<COLLADA/>')
        self.assertEqual(jobs.resolve_result(folder), output)
        self.assertEqual(jobs.resolve_result(output), output)
        with self.assertRaises(ValueError):
            jobs.resolve_result(self.root)

    def test_file_copy_preserves_original_and_unique_job(self):
        source = self.root / 'plan.DXF'
        source.write_text('original')
        with patch.object(jobs, 'jobs_root', return_value=self.root / 'jobs'):
            first = jobs.prepare_input([str(source)])
            second = jobs.prepare_input([str(source)])
        self.assertNotEqual(first, second)
        self.assertEqual((first / source.name).read_text(), 'original')
        self.assertEqual(source.read_text(), 'original')

    def test_duplicate_basenames_are_rejected_before_copy(self):
        paths = []
        for folder, body in [('a', 'floor one'), ('b', 'floor two')]:
            parent = self.root / folder
            parent.mkdir()
            source = parent / 'plan.dxf'
            source.write_text(body)
            paths.append(str(source))
        with patch.object(jobs, 'jobs_root', return_value=self.root / 'jobs'):
            with self.assertRaises(ValueError):
                jobs.prepare_input(paths)
        self.assertFalse((self.root / 'jobs').exists())

    def test_invalid_inputs_and_count_excludes_results(self):
        with self.assertRaises(ValueError):
            jobs.prepare_input([])
        text = self.root / 'notes.txt'
        text.write_text('not a drawing')
        with self.assertRaises(ValueError):
            jobs.prepare_input([str(text)])
        (self.root / 'floor.dxf').touch()
        nested = self.root / 'nested'
        nested.mkdir()
        (nested / 'floor.DWG').touch()
        output = self.root / jobs.OUT_DIR_NAME
        output.mkdir()
        (output / 'generated.dxf').touch()
        self.assertEqual(jobs.count_drawings(self.root), 2)
        with self.assertRaises(ValueError):
            jobs.prepare_input([str(self.root), str(nested)])

    def test_case_only_duplicate_is_rejected_for_windows(self):
        paths = []
        for folder, name in [('a', 'plan.dxf'), ('b', 'PLAN.DXF')]:
            parent = self.root / folder
            parent.mkdir()
            source = parent / name
            source.write_text(folder)
            paths.append(str(source))
        with patch.object(jobs, 'jobs_root', return_value=self.root / 'jobs'):
            with self.assertRaises(ValueError):
                jobs.prepare_input(paths)

    def test_summary_counts_match_build_output(self):
        summary = jobs.parse_summary('floor[F2]: walls=12 doors=3\nifc: walls=20 doors=6 windows=4\n')
        self.assertEqual(summary['floors'][0], {'key': 'F2', 'walls': '12', 'doors': '3'})
        self.assertEqual(summary['totals']['windows'], '4')
        self.assertEqual(jobs.parse_summary(''), {'floors': [], 'totals': {}})


if __name__ == '__main__':
    unittest.main()
