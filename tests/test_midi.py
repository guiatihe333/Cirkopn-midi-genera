import importlib
import io
import sys
import tempfile
import types

import numpy as np
import mido

class DummyStreamlit(types.ModuleType):
    def __getattr__(self, name):
        def no_op(*args, **kwargs):
            return None
        return no_op

sys.modules.setdefault('streamlit', DummyStreamlit('streamlit'))
sys.modules.setdefault('streamlit_drawable_canvas', DummyStreamlit('streamlit_drawable_canvas'))

midigen = importlib.import_module('midigen')

def test_midi_to_poly_dataset_simple():
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.Message('note_on', note=60, velocity=100, time=0))
    track.append(mido.Message('note_off', note=60, velocity=100, time=480))
    with tempfile.NamedTemporaryFile(suffix='.mid') as tmp:
        mid.save(tmp.name)
        data = midigen.midi_to_poly_dataset(tmp.name)
    assert data.shape[1] == 5
    assert data[0][1] == 60
