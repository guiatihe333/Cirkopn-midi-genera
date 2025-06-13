import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import tempfile
from midi_utils import export_ckc, import_ckc_bin, import_cki_txt


def test_import_cki_txt_missing():
    with tempfile.TemporaryDirectory() as tmp:
        missing = os.path.join(tmp, "nofile.cki")
        try:
            import_cki_txt(missing)
        except IOError:
            pass
        else:
            raise AssertionError("Expected IOError")


def test_export_ckc_roundtrip():
    notes = [
        [[0, 60, 100, 120, 0], [120, 62, 100, 120, 0]],
        [[0, 64, 100, 120, 1]]
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.ckc")
        export_ckc(notes, path)
        loaded = import_ckc_bin(path)
        assert len(loaded) == len(notes)
        assert loaded[0][0][1] == 60
        assert loaded[1][0][1] == 64
