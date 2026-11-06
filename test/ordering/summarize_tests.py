"""Integrity tests for replacing historical results with a corrected cohort."""
import csv
import tempfile
import unittest
from pathlib import Path
import summarize as report


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        report.ASSET_ROOTS.clear()
        self.addCleanup(report.ASSET_ROOTS.clear)
        self.base = self.root / "original"
        self.new = self.root / "corrected"
        self.fixture(self.base, False)
        self.fixture(self.new, True)

    def fixture(self, root, corrected, sha="a"*64, n=20):
        folder = root/"bodyy5"
        folder.mkdir(parents=True, exist_ok=True)
        fields = ["matrix","method","status","n","nnz_A","nnz_L","factor_median_s"]
        with (folder/"metrics.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            if not corrected:
                w.writerow(dict(matrix="bodyy5",method="amd",status="OK",n=n,nnz_A=60,nnz_L=100,factor_median_s=.1))
            w.writerow(dict(matrix="bodyy5",method="stiles",status="OK",n=n,nnz_A=60,nnz_L=80 if corrected else 1000,factor_median_s=.09 if corrected else 1))
        marker = "stiles_current_export=inverted_for_gather\n" if corrected else ""
        (folder/"index.html").write_text("<pre>CPU=sapphirerapids\n"+marker+"sha256="+sha+"\n</pre>")
        (folder/"stiles-factor.svg").write_text("corrected" if corrected else "old")

    def test_replace_only_stiles_and_its_assets(self):
        groups, metadata, provenance = report.load(self.base)
        old_amd = dict(groups["bodyy5"]["amd"])
        self.assertEqual(groups["bodyy5"]["stiles"]["status"],"DIRECTION_INVALID")
        audit, cohorts = report.merge_corrected(groups,metadata,provenance,self.new)
        self.assertEqual(groups["bodyy5"]["amd"],old_amd)
        self.assertEqual(groups["bodyy5"]["stiles"]["nnz_L"],80)
        self.assertEqual(audit[0]["nnz_L"],1000)
        self.assertIn("corrected_stiles",cohorts["bodyy5"])
        self.assertEqual(report.asset(self.base,"bodyy5","stiles-factor.svg").read_text(),"corrected")

    def test_mismatched_hash_is_rejected_without_mutation(self):
        self.fixture(self.new,True,sha="b"*64)
        groups,metadata,provenance = report.load(self.base)
        with self.assertRaisesRegex(ValueError,"SHA-256 mismatch"):
            report.merge_corrected(groups,metadata,provenance,self.new)
        self.assertEqual(groups["bodyy5"]["stiles"]["nnz_L"],1000)

    def test_dimensions_must_match_even_if_hash_claim_matches(self):
        self.fixture(self.new,True,n=21)
        groups,metadata,provenance = report.load(self.base)
        with self.assertRaisesRegex(ValueError,"dimensions/nonzeros"):
            report.merge_corrected(groups,metadata,provenance,self.new)

    def test_old_convention_cannot_be_used_as_corrected(self):
        self.fixture(self.new,False)
        groups,metadata,provenance = report.load(self.base)
        with self.assertRaisesRegex(ValueError,"convention marker"):
            report.merge_corrected(groups,metadata,provenance,self.new)


if __name__ == "__main__":
    unittest.main()
