import numpy as np
import mido
import torch
import torch.nn as nn
import torch.nn.functional as F
import struct
import json
import tempfile
from midi2audio import FluidSynth
from collections import defaultdict
import os

# -------- MIDI Parser --------
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
                                i
                            ])
            for (note, channel), start_tick in active_notes.items():
                duration = abs_tick - start_tick
                if duration > 0:
                    all_notes_for_ai.append([
                        start_tick,
                        note,
                        64,
                        duration,
                        i
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
        raise IOError(f"Erro ao processar arquivo MIDI: {e}")


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


# -------- IA Polifônica --------
class PolyTransformer(nn.Module):
    def __init__(self, n_tracks=4, d_model=128, num_lstm_layers=2, dropout_rate=0.2):
        super().__init__()
        self.note_embed = nn.Embedding(128, d_model // 2)
        self.track_embed = nn.Embedding(n_tracks, d_model // 4)
        self.continuous_linear = nn.Linear(3, d_model // 4)
        lstm_input_size = (d_model // 2) + (d_model // 4) + (d_model // 4)
        self.lstm = nn.LSTM(lstm_input_size, d_model, num_layers=num_lstm_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.output_delta_tick = nn.Linear(d_model, 1)
        self.output_note = nn.Linear(d_model, 128)
        self.output_velocity = nn.Linear(d_model, 1)
        self.output_duration = nn.Linear(d_model, 1)
        self.output_track = nn.Linear(d_model, n_tracks)

    def forward(self, x):
        delta_tick_data = x[:, :, 0].unsqueeze(-1)
        note_data = x[:, :, 1].long()
        velocity_data = x[:, :, 2].unsqueeze(-1)
        duration_data = x[:, :, 3].unsqueeze(-1)
        track_data = x[:, :, 4].long()
        note_embedded = self.note_embed(note_data)
        track_embedded = self.track_embed(track_data)
        continuous_features = torch.cat([delta_tick_data, velocity_data, duration_data], dim=-1)
        continuous_projected = self.continuous_linear(continuous_features.float())
        combined_input = torch.cat([note_embedded, track_embedded, continuous_projected], dim=-1)
        lstm_out, _ = self.lstm(combined_input)
        lstm_out = self.dropout(lstm_out)
        pred_delta_tick = F.relu(self.output_delta_tick(lstm_out)) + 1
        pred_note_logits = self.output_note(lstm_out)
        pred_velocity = torch.sigmoid(self.output_velocity(lstm_out)) * 127
        pred_duration = F.relu(self.output_duration(lstm_out)) + 1
        pred_track_logits = self.output_track(lstm_out)
        return pred_delta_tick, pred_note_logits, pred_velocity, pred_duration, pred_track_logits


def train_poly_model(notes_data, params):
    try:
        if notes_data.shape[0] == 0:
            return None
        data_tensor = torch.from_numpy(notes_data.astype(float)).float().unsqueeze(0)
        n_tracks_in_data = int(notes_data[:, 4].max()) + 1 if notes_data.shape[0] > 0 else 1
        model = PolyTransformer(
            n_tracks=n_tracks_in_data,
            d_model=params.get("d_model", 128),
            num_lstm_layers=params.get("num_lstm_layers", 2),
            dropout_rate=params.get("dropout_rate", 0.2),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        criterion_continuous = nn.MSELoss()
        criterion_note = nn.CrossEntropyLoss()
        criterion_track = nn.CrossEntropyLoss()
        for epoch in range(params["epochs"]):
            optimizer.zero_grad()
            pred_delta_tick, pred_note_logits, pred_velocity, pred_duration, pred_track_logits = model(data_tensor)
            target_delta_tick = data_tensor[:, :, 0].unsqueeze(-1)
            target_note = data_tensor[:, :, 1].long()
            target_velocity = data_tensor[:, :, 2].unsqueeze(-1)
            target_duration = data_tensor[:, :, 3].unsqueeze(-1)
            target_track = data_tensor[:, :, 4].long()
            loss_delta_tick = criterion_continuous(pred_delta_tick, target_delta_tick)
            loss_note = criterion_note(pred_note_logits.view(-1, 128), target_note.view(-1))
            loss_velocity = criterion_continuous(pred_velocity, target_velocity)
            loss_duration = criterion_continuous(pred_duration, target_duration)
            loss_track = criterion_track(pred_track_logits.view(-1, n_tracks_in_data), target_track.view(-1))
            total_loss = loss_delta_tick + loss_note + loss_velocity + loss_duration + loss_track
            total_loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), params["model_path"])
        return model
    except Exception as e:
        raise ValueError(f"Erro ao treinar o modelo de IA: {e}")


def infer_poly_sequence(model, seq_len=512, n_tracks=4, prime_sequence=None, randomness_temp=1.0):
    try:
        model.eval()
        if prime_sequence is None or prime_sequence.shape[0] == 0:
            prime_sequence = np.array([[100, 60, 80, 100, 0]], dtype=float)
        current_sequence = prime_sequence.copy()
        for i in range(seq_len):
            input_note = torch.from_numpy(current_sequence[-1:].astype(float)).float().unsqueeze(0)
            with torch.no_grad():
                pred_delta_tick, pred_note_logits, pred_velocity, pred_duration, pred_track_logits = model(input_note)
            pred_delta_tick_val = max(1, int(pred_delta_tick.item()))
            pred_note_prob = F.softmax(pred_note_logits / randomness_temp, dim=-1)
            pred_note_val = torch.multinomial(pred_note_prob.squeeze(0), 1).item()
            pred_velocity_val = int(np.clip(pred_velocity.item(), 0, 127))
            pred_duration_val = max(1, int(pred_duration.item()))
            pred_track_prob = F.softmax(pred_track_logits / randomness_temp, dim=-1)
            pred_track_val = torch.multinomial(pred_track_prob.squeeze(0), 1).item()
            new_note_features = np.array([pred_delta_tick_val, pred_note_val, pred_velocity_val, pred_duration_val, pred_track_val], dtype=float)
            current_sequence = np.vstack([current_sequence, new_note_features])
        out_sequences_by_track = defaultdict(list)
        for note_event in current_sequence:
            out_sequences_by_track[int(note_event[4])].append(note_event)
        final_out_sequences = [np.array(out_sequences_by_track[k]) for k in sorted(out_sequences_by_track.keys())]
        return final_out_sequences
    except Exception as e:
        raise ValueError(f"Erro ao inferir sequência com o modelo de IA: {e}")


# -------- Algoritmos de Manipulação MIDI --------
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


# -------- Automação MIDI --------
def lfo_automation(length, cc_num, depth=64, freq=0.1, base=64, channel=0):
    t = np.arange(length)
    values = (base + depth * np.sin(2 * np.pi * freq * t / length)).astype(int)
    ccs = [[tick, int(val)] for tick, val in enumerate(values)]
    return ccs


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
    ccs = [[tick, int(val)] for tick, val in enumerate(values)]
    return ccs


def randomize_automation(length, cc_num, minv=0, maxv=127, channel=0):
    values = np.random.randint(minv, maxv+1, length)
    ccs = [[tick, int(val)] for tick, val in enumerate(values)]
    return ccs


# -------- Exportação Cirklon CKC/CKI --------
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
        raise IOError(f"Erro ao exportar CKC: {e}")


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
        raise IOError(f"Erro ao exportar CKI: {e}")


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
                        "track": t
                    }
                    p3.append(ev)
        with open(out_path, "w") as f:
            json.dump({"p3_pattern":p3}, f, indent=2)
        return out_path
    except Exception as e:
        raise IOError(f"Erro ao exportar P3 Pattern JSON: {e}")


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
                        "track": t
                    })
        with open(out_path, "w") as f:
            json.dump({"aux_rows":aux}, f, indent=2)
        return out_path
    except Exception as e:
        raise IOError(f"Erro ao exportar Aux Rows: {e}")


# -------- Importação Cirklon CKC/CKI/AuxRows --------
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
        raise IOError(f"Erro ao importar CKC: {e}")


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
        tracks = defaultdict(list)
        for n in notes:
            tracks[n[4]].append(n)
        return [np.array(tracks[k]) for k in sorted(tracks)]
    except Exception as e:
        raise IOError(f"Erro ao importar CKI: {e}")


def import_auxrows_json(path):
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("aux_rows", [])
    except Exception as e:
        raise IOError(f"Erro ao importar Aux Rows JSON: {e}")


# -------- Validação Musical Avançada --------
def is_note_in_scale(note, scale_notes):
    return (note % 12) in scale_notes


def validate_advanced(notes, ccs, scale_root=60, scale_type="major", max_poly=8, pitch_range=(0,127)):
    scales = {
        "major": [0,2,4,5,7,9,11],
        "minor": [0,2,3,5,7,8,10],
        "dorian": [0,1,3,5,7,9,10],
        "phrygian": [0,1,3,5,7,8,10]
    }
    scale_pcs = scales.get(scale_type, scales["major"])
    errors = []
    for t, track in enumerate(notes):
        for n in track:
            if not is_note_in_scale(n[1], scale_pcs):
                errors.append(f"Track {t} note {n[1]} ({n[1]%12}) fora da escala {scale_type}")
            if not (pitch_range[0]<=n[1]<=pitch_range[1]):
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
    cc_map = {cc[0]: cc[3] for cc in ccs if cc[2]==1}
    for t, track in enumerate(notes):
        cumulative_tick = 0
        for n in track:
            cumulative_tick += n[0]
            cc_val = cc_map.get(cumulative_tick, 127)
            if cc_val < 20 and n[2] > 60:
                errors.append(f"Track {t} nota em tick {cumulative_tick} velocity alta ({n[2]}) com CC1 muito baixo ({cc_val})")
    return errors


# -------- Preview MIDI com SoundFont --------
def midi_preview_sf2(notes, sf2_path, tempo=120):
    try:
        midi_file = notes_to_midi_file(notes, tempo=tempo)
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
            midi_file.save(tmp.name)
            tmp.flush()
            audio_path = tmp.name.replace(".mid", ".wav")
            FluidSynth(sf2_path).midi_to_audio(tmp.name, audio_path)
        return audio_path
    except Exception as e:
        raise IOError(f"Erro ao gerar preview de áudio: {e}")
