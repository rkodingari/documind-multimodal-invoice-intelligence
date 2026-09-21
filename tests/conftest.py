import os
from pathlib import Path

TEST_ROOT = Path("/tmp/documind-tests")
TEST_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["DOCUMIND_DATABASE_URL"] = "sqlite:////tmp/documind-tests/documind.db"
os.environ["DOCUMIND_UPLOAD_DIR"] = str(TEST_ROOT / "uploads")
