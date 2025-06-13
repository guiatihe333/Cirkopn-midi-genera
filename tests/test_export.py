import importlib
import json
import sys
import types
import numpy as np

class DummyStreamlit(types.ModuleType):
    def __getattr__(self, name):
        def no_op(*args, **kwargs):
            return None
        return no_op

sys.modules.setdefault('streamlit', DummyStreamlit('streamlit'))
sys.modules.setdefault('streamlit_drawable_canvas', DummyStreamlit('streamlit_drawable_canvas'))

midigen = importlib.import_module('midigen')


def make_notes():
    return [np.array([[0, 60, 100, 120, 0], [120, 62, 100, 120, 0]])]


def test_export_ckc(tmp_path):
    path = tmp_path / 'test.ckc'
    midigen.export_ckc(make_notes(), str(path))
    assert path.exists()
    with open(path, 'rb') as f:
        assert f.read(4) == b'CKC0'


def test_export_cki(tmp_path):
    path = tmp_path / 'test.cki'
    midigen.export_cki(make_notes(), str(path))
    assert path.exists()
    with open(path) as f:
        assert f.readline().startswith('CKI')


def test_export_p3pattern_json(tmp_path):
    path = tmp_path / 'pattern.json'
    midigen.export_p3pattern_json(make_notes(), str(path))
    assert path.exists()
    with open(path) as f:
        data = json.load(f)
    assert 'p3_pattern' in data


def test_export_auxrows(tmp_path):
    path = tmp_path / 'aux.json'
    midigen.export_auxrows(make_notes(), str(path))
    assert path.exists()
    with open(path) as f:
        data = json.load(f)
    assert 'aux_rows' in data
