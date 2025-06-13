import os
import numpy as np
from midigen import notes_to_midi_file, export_midi_file, midi_to_poly_dataset

def test_notes_to_midi_file_roundtrip(tmp_path):
    notes = [np.array([[0, 60, 100, 120, 0]], dtype=int)]
    midi = notes_to_midi_file(notes, tempo=120)
    path = tmp_path / "out.mid"
    midi.save(path)
    assert path.exists()
    new_notes = midi_to_poly_dataset(str(path))
    assert new_notes.size > 0


def test_export_midi_file(tmp_path):
    notes = [np.array([[0, 60, 100, 120, 0]], dtype=int)]
    path = tmp_path / "out.mid"
    export_midi_file(notes, str(path), tempo=120)
    assert os.path.exists(path)
