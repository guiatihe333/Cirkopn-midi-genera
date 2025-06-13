import mido
import numpy as np


def midi_to_poly_dataset(midi_file):
    """Parse a MIDI file or file-like object into a polyphonic dataset.

    Parameters
    ----------
    midi_file : str or file-like
        Path to a MIDI file or an object with a ``read`` method returning bytes.

    Returns
    -------
    numpy.ndarray
        Array with columns [delta_tick, note, velocity, duration, track_index].
    """
    try:
        if hasattr(midi_file, "read"):
            mid = mido.MidiFile(file=midi_file)
        else:
            mid = mido.MidiFile(midi_file)

        all_notes_for_ai = []
        for i, mtrack in enumerate(mid.tracks):
            active_notes = {}
            abs_tick = 0
            for msg in mtrack:
                abs_tick += msg.time
                if msg.type == "note_on" and msg.velocity > 0:
                    active_notes[(msg.note, msg.channel)] = abs_tick
                elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                    if (msg.note, msg.channel) in active_notes:
                        start_tick = active_notes.pop((msg.note, msg.channel))
                        duration = abs_tick - start_tick
                        if duration > 0:
                            all_notes_for_ai.append([
                                start_tick,
                                msg.note,
                                msg.velocity,
                                duration,
                                i,
                            ])
            for (note, channel), start_tick in active_notes.items():
                duration = abs_tick - start_tick
                if duration > 0:
                    all_notes_for_ai.append([
                        start_tick,
                        note,
                        64,
                        duration,
                        i,
                    ])

        all_notes_for_ai.sort(key=lambda x: x[0])

        final_dataset = []
        last_abs_tick = 0
        for abs_tick, note, velocity, duration, track_idx in all_notes_for_ai:
            delta_tick = abs_tick - last_abs_tick
            final_dataset.append([delta_tick, note, velocity, duration, track_idx])
            last_abs_tick = abs_tick

        return np.array(final_dataset) if final_dataset else np.array([])
    except Exception as e:
        raise RuntimeError(f"Erro ao processar arquivo MIDI: {e}")


def notes_to_midi_file(notes, tempo=120):
    """Convert notes in dataset format to a ``mido.MidiFile`` object."""
    mid = mido.MidiFile()

    max_track_idx = 0
    if notes:
        for track_data in notes:
            if track_data.size > 0:
                max_track_idx = max(max_track_idx, int(track_data[:, 4].max()))

    for _ in range(max_track_idx + 1):
        mid.add_track()

    for track_idx, track_data in enumerate(notes):
        cumulative_tick = 0
        absolute_notes_data = []
        for delta_tick, note, velocity, duration, _ in track_data:
            cumulative_tick += delta_tick
            absolute_notes_data.append([cumulative_tick, note, velocity, duration, track_idx])

        if absolute_notes_data:
            sorted_track_data = np.array(absolute_notes_data)[np.argsort(np.array(absolute_notes_data)[:, 0])]
        else:
            sorted_track_data = []

        track_messages = []
        for start_tick, note, velocity, duration, _ in sorted_track_data:
            track_messages.append((start_tick, mido.Message("note_on", note=int(note), velocity=int(velocity))))
            end_tick = start_tick + duration
            track_messages.append((end_tick, mido.Message("note_off", note=int(note), velocity=int(velocity))))

        track_messages.sort(key=lambda x: x[0])

        last_tick = 0
        for tick, msg in track_messages:
            delta_time = int(tick - last_tick)
            msg.time = delta_time
            mid.tracks[track_idx].append(msg)
            last_tick = tick

    if mid.tracks:
        mid.tracks[0].insert(0, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo)))

    return mid


def export_midi_file(notes, out_path, tempo=120):
    """Write notes to a MIDI file on disk."""
    mid = notes_to_midi_file(notes, tempo=tempo)
    mid.save(out_path)
    return out_path
