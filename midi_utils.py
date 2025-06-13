import json
import struct
import mido
import numpy as np
from collections import defaultdict


def midi_to_poly_dataset(midi_file):
    import streamlit as st
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
                if msg.type == 'note_on' and msg.velocity > 0:
                    active_notes[(msg.note, msg.channel)] = abs_tick
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if (msg.note, msg.channel) in active_notes:
                        start_tick = active_notes.pop((msg.note, msg.channel))
                        duration = abs_tick - start_tick
                        if duration > 0:
                            all_notes_for_ai.append([
                                start_tick,
                                msg.note,
                                msg.velocity,
                                duration,
                                i
                            ])
            for (note, channel), start_tick in active_notes.items():
                duration = abs_tick - start_tick
                if duration > 0:
                    all_notes_for_ai.append([start_tick, note, 64, duration, i])
        all_notes_for_ai.sort(key=lambda x: x[0])
        final_dataset = []
        last_abs_tick = 0
        for note_event in all_notes_for_ai:
            abs_tick, note, velocity, duration, track_idx = note_event
            delta_tick = abs_tick - last_abs_tick
            final_dataset.append([delta_tick, note, velocity, duration, track_idx])
            last_abs_tick = abs_tick
        return np.array(final_dataset) if final_dataset else np.array([])
    except (IOError, ValueError) as e:
        st.error(f"Erro ao processar arquivo MIDI: {e}")
        return np.array([])


def notes_to_midi_file(notes, tempo=120):
    mid = mido.MidiFile()
    max_track_idx = 0
    if notes:
        for track_data in notes:
            if track_data.size > 0:
                max_track_idx = max(max_track_idx, int(track_data[:, 4].max()))
    for _ in range(max_track_idx + 1):
        mid.tracks.append(mido.MidiTrack())
    for track_data in notes:
        abs_tick = 0
        track_idx = int(track_data[0][4]) if len(track_data) > 0 else 0
        for note_event in track_data:
            delta, note, velocity, duration, track = note_event
            abs_tick += delta
            mid.tracks[track].append(mido.Message('note_on', note=int(note), velocity=int(velocity), time=delta))
            mid.tracks[track].append(mido.Message('note_off', note=int(note), velocity=0, time=int(duration)))
    if mid.tracks:
        mid.tracks[0].insert(0, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo)))
    return mid


def export_midi_file(notes, out_path, tempo=120):
    import streamlit as st
    try:
        midi_file = notes_to_midi_file(notes, tempo=tempo)
        midi_file.save(out_path)
        return out_path
    except (IOError, ValueError) as e:
        st.error(f"Erro ao exportar MIDI: {e}")
        return None


def export_ckc(notes, out_path):
    import streamlit as st
    try:
        with open(out_path, "wb") as f:
            f.write(b'CKC0')
            for t, track in enumerate(notes):
                for n in track:
                    if n[3] > 0:
                        f.write(struct.pack('<I', int(n[0])))
                        f.write(struct.pack('B', int(n[1])))
                        f.write(struct.pack('B', int(n[2])))
                        f.write(struct.pack('B', int(n[3])))
                        f.write(struct.pack('B', t))
        return out_path
    except IOError as e:
        st.error(f"Erro ao exportar CKC: {e}")
        return None


def export_cki(notes, out_path):
    import streamlit as st
    try:
        with open(out_path, "w") as f:
            f.write("CKI,1\n")
            for t, track in enumerate(notes):
                for n in track:
                    if n[3] > 0:
                        line = f"{int(n[0])},{int(n[1])},{int(n[2])},{int(n[3])},{t}\n"
                        f.write(line)
        return out_path
    except IOError as e:
        st.error(f"Erro ao exportar CKI: {e}")
        return None


def export_p3pattern_json(notes, out_path):
    import streamlit as st
    try:
        p3 = []
        for t, track in enumerate(notes):
            for n in track:
                if n[3] > 0:
                    ev = {
                        "step": int(n[0]),
                        "note": int(n[1]),
                        "velocity": int(n[2]),
                        "length": int(n[3]),
                        "track": t
                    }
                    p3.append(ev)
        with open(out_path, "w") as f:
            json.dump({"p3_pattern": p3}, f, indent=2)
        return out_path
    except IOError as e:
        st.error(f"Erro ao exportar P3 Pattern JSON: {e}")
        return None


def export_auxrows(notes, out_path):
    import streamlit as st
    try:
        aux = []
        for t, track in enumerate(notes):
            for n in track:
                if n[3] > 0:
                    aux.append({
                        "tick": int(n[0]),
                        "cc_num": 74,
                        "cc_val": int(n[2]),
                        "track": t
                    })
        with open(out_path, "w") as f:
            json.dump({"aux_rows": aux}, f, indent=2)
        return out_path
    except IOError as e:
        st.error(f"Erro ao exportar Aux Rows: {e}")
        return None


def import_ckc_bin(path):
    import streamlit as st
    try:
        with open(path, "rb") as f:
            header = f.read(4)
            if header != b'CKC0':
                st.error("Arquivo CKC inválido")
                return []
            data = f.read()
        notes = []
        for i in range(0, len(data), 9):
            step = struct.unpack('<I', data[i:i+4])[0]
            note, velocity, length, track = struct.unpack('BBBB', data[i+4:i+8])
            notes.append([step, note, velocity, length, track])
        tracks = defaultdict(list)
        for n in notes:
            tracks[n[4]].append(n)
        return [np.array(tracks[k]) for k in sorted(tracks)]
    except IOError as e:
        st.error(f"Erro ao importar CKC: {e}")
        return []


def import_cki_txt(path):
    import streamlit as st
    try:
        notes = []
        with open(path) as f:
            for line in f:
                if line.startswith("CKI"): continue
                vals = [int(x) for x in line.strip().split(",")]
                if len(vals) == 5:
                    notes.append(vals)
                else:
                    st.warning(f"Linha CKI com formato inesperado: {line.strip()}")
        tracks = defaultdict(list)
        for n in notes:
            tracks[n[4]].append(n)
        return [np.array(tracks[k]) for k in sorted(tracks)]
    except (IOError, ValueError) as e:
        st.error(f"Erro ao importar CKI: {e}")
        return []


def import_auxrows_json(path):
    import streamlit as st
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("aux_rows", [])
    except (IOError, json.JSONDecodeError) as e:
        st.error(f"Erro ao importar Aux Rows JSON: {e}")
        return []
