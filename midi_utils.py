# MIDI utility functions
import numpy as np
import mido
import struct
import json
import tempfile
from midi2audio import FluidSynth
import streamlit as st
from collections import defaultdict


def midi_to_poly_dataset(midi_file):
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
        for note_event in all_notes_for_ai:
            abs_tick, note, velocity, duration, track_idx = note_event
            delta_tick = abs_tick - last_abs_tick
            final_dataset.append([delta_tick, note, velocity, duration, track_idx])
            last_abs_tick = abs_tick
        return np.array(final_dataset) if final_dataset else np.array([])
    except Exception as e:
        st.error(f"Erro ao processar arquivo MIDI: {e}")
        return np.array([])


def notes_to_midi_file(notes, tempo=120):
    mid = mido.MidiFile()
    max_track_idx = 0
    if notes:
        for track_data in notes:
            if track_data.size > 0:
                max_track_idx = max(max_track_idx, int(track_data[:, 4].max()))
    for i in range(max_track_idx + 1):
        mid.add_track()
    for track_idx, track_data in enumerate(notes):
        cumulative_tick = 0
        absolute_notes_data = []
        for note_event in track_data:
            delta_tick, note, velocity, duration, _ = note_event
            cumulative_tick += delta_tick
            absolute_notes_data.append([cumulative_tick, note, velocity, duration, track_idx])
        if len(absolute_notes_data) > 0:
            sorted_track_data = np.array(absolute_notes_data)[np.argsort(np.array(absolute_notes_data)[:, 0])]
        else:
            sorted_track_data = []
        track_messages = []
        for note_event in sorted_track_data:
            start_tick, note, velocity, duration, _ = note_event
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


def breakbeat_poly(notes):
    res = []
    for track in notes:
        idx = np.arange(len(track))
        np.random.shuffle(idx)
        res.append(track[idx])
    return res


def microtiming_poly(notes, max_shift=5):
    result = []
    for track in notes:
        if len(track) == 0:
            result.append(track)
            continue
        abs_events = []
        cumulative = 0
        for evt in track:
            cumulative += evt[0]
            abs_events.append([cumulative, evt[1], evt[2], evt[3], evt[4]])
        shifted = []
        for tick, note, vel, dur, trk in abs_events:
            shifted_tick = max(0, tick + np.random.randint(-max_shift, max_shift + 1))
            shifted.append([shifted_tick, note, vel, dur, trk])
        shifted.sort(key=lambda x: x[0])
        delta_track = []
        prev_tick = 0
        for tick, note, vel, dur, trk in shifted:
            delta = tick - prev_tick
            delta_track.append([max(0, delta), note, vel, dur, trk])
            prev_tick = tick
        result.append(np.array(delta_track))
    return result


def lfo_automation(length, cc_num, depth=64, freq=0.1, base=64, channel=0):
    t = np.arange(length)
    values = (base + depth * np.sin(2 * np.pi * freq * t / length)).astype(int)
    return [[tick, channel, cc_num, int(val)] for tick, val in enumerate(values)]


def envelope_automation(length, cc_num, attack=0.1, decay=0.2, sustain=0.7, release=0.2, max_val=127, channel=0):
    atk_len = int(length * attack)
    dec_len = int(length * decay)
    sus_len = int(length * sustain)
    rel_len = length - (atk_len + dec_len + sus_len)
    values = np.concatenate([
        np.linspace(0, max_val, atk_len),
        np.linspace(max_val, max_val*0.7, dec_len),
        np.ones(sus_len) * max_val * 0.7,
        np.linspace(max_val*0.7, 0, rel_len)
    ])
    return [[tick, channel, cc_num, int(val)] for tick, val in enumerate(values)]


def randomize_automation(length, cc_num, minv=0, maxv=127, channel=0):
    values = np.random.randint(minv, maxv+1, length)
    return [[tick, channel, cc_num, int(val)] for tick, val in enumerate(values)]


def export_ckc(notes, out_path):
    try:
        with open(out_path, "wb") as f:
            f.write(b'CKC0')
            f.write(struct.pack('>H', 1))
            f.write(struct.pack('>H', 0))
            total_events = sum([len(t) for t in notes])
            f.write(struct.pack('>H', total_events))
            for t, track in enumerate(notes):
                for n in track:
                    if n[3] > 0:
                        f.write(struct.pack('>I', int(n[0])))
                        f.write(struct.pack('B', int(n[1])))
                        f.write(struct.pack('B', int(n[2])))
                        f.write(struct.pack('B', int(n[3])))
                        f.write(struct.pack('B', t))
        return out_path
    except Exception as e:
        st.error(f"Erro ao exportar CKC: {e}")
        return None


def export_cki(notes, out_path):
    try:
        with open(out_path, "w") as f:
            f.write("CKI,1\n")
            for t, track in enumerate(notes):
                for n in track:
                    if n[3] > 0:
                        line = f"{int(n[0])},{int(n[1])},{int(n[2])},{int(n[3])},{t}\n"
                        f.write(line)
        return out_path
    except Exception as e:
        st.error(f"Erro ao exportar CKI: {e}")
        return None


def export_p3pattern_json(notes, out_path):
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
                        "track": t,
                    }
                    p3.append(ev)
        with open(out_path, "w") as f:
            json.dump({"p3_pattern": p3}, f, indent=2)
        return out_path
    except Exception as e:
        st.error(f"Erro ao exportar P3 Pattern JSON: {e}")
        return None


def export_auxrows(notes, out_path):
    try:
        aux = []
        for t, track in enumerate(notes):
            for n in track:
                if n[3] > 0:
                    aux.append({
                        "tick": int(n[0]),
                        "cc_num": 74,
                        "cc_val": int(n[2]),
                        "track": t,
                    })
        with open(out_path, "w") as f:
            json.dump({"aux_rows": aux}, f, indent=2)
        return out_path
    except Exception as e:
        st.error(f"Erro ao exportar Aux Rows: {e}")
        return None


def import_ckc_bin(path):
    try:
        with open(path, "rb") as f:
            _ = f.read(4)
            n_patterns = struct.unpack('>H', f.read(2))[0]
            _ = f.read(2)
            n_events = struct.unpack('>H', f.read(2))[0]
            notes = []
            for _ in range(n_events):
                tick = struct.unpack('>I', f.read(4))[0]
                note = struct.unpack('B', f.read(1))[0]
                vel = struct.unpack('B', f.read(1))[0]
                duration = struct.unpack('B', f.read(1))[0]
                track = struct.unpack('B', f.read(1))[0]
                notes.append([tick, note, vel, duration, track])
        tracks = defaultdict(list)
        for n in notes:
            tracks[n[4]].append(n)
        return [np.array(tracks[k]) for k in sorted(tracks)]
    except Exception as e:
        st.error(f"Erro ao importar CKC: {e}")
        return []


def import_cki_txt(path):
    try:
        notes = []
        with open(path) as f:
            for line in f:
                if line.startswith("CKI"):
                    continue
                vals = [int(x) for x in line.strip().split(",")]
                if len(vals) == 5:
                    notes.append(vals)
                else:
                    st.warning(f"Linha CKI com formato inesperado: {line.strip()}")
        tracks = defaultdict(list)
        for n in notes:
            tracks[n[4]].append(n)
        return [np.array(tracks[k]) for k in sorted(tracks)]
    except Exception as e:
        st.error(f"Erro ao importar CKI: {e}")
        return []


def import_auxrows_json(path):
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("aux_rows", [])
    except Exception as e:
        st.error(f"Erro ao importar Aux Rows JSON: {e}")
        return []


def is_note_in_scale(note, scale_notes):
    return (note % 12) in scale_notes


def validate_advanced(notes, ccs, scale_root=60, scale_type="major", max_poly=8, pitch_range=(0, 127)):
    scales = {
        "major": [0, 2, 4, 5, 7, 9, 11],
        "minor": [0, 2, 3, 5, 7, 8, 10],
        "dorian": [0, 1, 3, 5, 7, 9, 10],
        "phrygian": [0, 1, 3, 5, 7, 8, 10],
    }
    scale_pcs = scales.get(scale_type, scales["major"])
    errors = []
    for t, track in enumerate(notes):
        for n in track:
            if not is_note_in_scale(n[1], scale_pcs):
                errors.append(f"Track {t} note {n[1]} ({n[1]%12}) fora da escala {scale_type}")
            if not (pitch_range[0] <= n[1] <= pitch_range[1]):
                errors.append(f"Track {t} note {n[1]} fora do range permitido")
    all_ticks = {}
    for t, track in enumerate(notes):
        cumulative_tick = 0
        for n in track:
            cumulative_tick += n[0]
            all_ticks.setdefault(cumulative_tick, []).append((t, n[1]))
    for tick, events in all_ticks.items():
        if len(events) > max_poly:
            errors.append(f"Polifonia excessiva ({len(events)}) no tick {tick}")
    cc_map = {cc[0]: cc[3] for cc in ccs if cc[2] == 1}
    for t, track in enumerate(notes):
        cumulative_tick = 0
        for n in track:
            cumulative_tick += n[0]
            cc_val = cc_map.get(cumulative_tick, 127)
            if cc_val < 20 and n[2] > 60:
                errors.append(f"Track {t} nota em tick {cumulative_tick} velocity alta ({n[2]}) com CC1 muito baixo ({cc_val})")
    return errors


def midi_preview_sf2(notes, sf2_path, tempo=120):
    try:
        midi_file = notes_to_midi_file(notes, tempo=tempo)
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
            midi_file.save(tmp.name)
            tmp.flush()
            audio_path = tmp.name.replace(".mid", ".wav")
            FluidSynth(sf2_path).midi_to_audio(tmp.name, audio_path)
            with open(audio_path, "rb") as f:
                st.audio(f, format="audio/wav")
    except Exception as e:
        st.error(f"Erro ao gerar preview de áudio: {e}")

