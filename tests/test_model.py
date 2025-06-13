import importlib
import sys
import tempfile
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


def test_train_poly_model(tmp_path):
    notes = np.array([[0, 60, 64, 120, 0], [120, 62, 64, 120, 0]], dtype=float)
    model_file = tmp_path / 'model.pt'
    params = {'lr': 0.001, 'epochs': 1, 'model_path': str(model_file)}
    model = midigen.train_poly_model(notes, params)
    assert model is not None
    assert model_file.exists()
