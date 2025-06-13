import os
import numpy as np
import mido
from midi_utils import midi_to_poly_dataset, notes_to_midi_file, export_midi_file


def test_roundtrip_tmpfile(tmp_path):
    """Convert notes to midi and back again."""
    notes = [np.array([[0, 60, 100, 60, 0], [60, 62, 100, 60, 0]])]
    midi = notes_to_midi_file(notes, tempo=120)
    temp_midi = tmp_path / "test.mid"
    midi.save(temp_midi)

    data = midi_to_poly_dataset(open(temp_midi, "rb"))
    assert data.shape[1] == 5
    assert data.shape[0] > 0


def test_export_midi_file(tmp_path):
    notes = [np.array([[0, 60, 100, 60, 0]])]
    out_path = tmp_path / "out.mid"
    export_midi_file(notes, out_path, tempo=120)
    assert out_path.exists()
    mf = mido.MidiFile(out_path)
    assert len(mf.tracks) > 0


