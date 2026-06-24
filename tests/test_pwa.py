import json
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")


def read(*parts):
    with open(os.path.join(*parts)) as fh:
        return fh.read()


class TestPwaAssets(unittest.TestCase):
    REQUIRED = ["index.html", "app.js", "styles.css", "manifest.webmanifest", "sw.js", "icon.svg"]

    def test_assets_present(self):
        for name in self.REQUIRED:
            self.assertTrue(os.path.isfile(os.path.join(DOCS, name)), f"missing docs/{name}")

    def test_manifest_valid_and_icon_exists(self):
        manifest = json.loads(read(DOCS, "manifest.webmanifest"))
        for key in ("name", "start_url", "display", "icons"):
            self.assertIn(key, manifest)
        self.assertTrue(manifest["icons"], "manifest has no icons")
        for icon in manifest["icons"]:
            self.assertTrue(os.path.isfile(os.path.join(DOCS, icon["src"])), f"missing icon {icon['src']}")

    def test_service_worker_shell_files_exist(self):
        sw = read(DOCS, "sw.js")
        shell = re.search(r"SHELL\s*=\s*\[(.*?)\]", sw, re.S).group(1)
        for rel in re.findall(r'"\./([^"]*)"', shell):
            if rel == "":  # "./" -> directory root
                continue
            self.assertTrue(os.path.isfile(os.path.join(DOCS, rel)), f"sw caches missing {rel}")

    def test_index_references_resolve(self):
        html = read(DOCS, "index.html")
        for rel in ("app.js", "styles.css", "manifest.webmanifest", "icon.svg"):
            self.assertIn(rel, html, f"index.html does not reference {rel}")
            self.assertTrue(os.path.isfile(os.path.join(DOCS, rel)))

    def test_dispatch_targets_existing_workflow(self):
        # The app fires this workflow by filename; it must exist or runs 404.
        app = read(DOCS, "app.js")
        m = re.search(r"workflows/([\w.-]+\.yml)/dispatches", app)
        self.assertIsNotNone(m, "app.js does not dispatch a workflow")
        wf = m.group(1)
        self.assertTrue(os.path.isfile(os.path.join(ROOT, ".github", "workflows", wf)),
                        f"dispatched workflow {wf} not found")


if __name__ == "__main__":
    unittest.main()
