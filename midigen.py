import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from streamlit_drawable_canvas import st_canvas
import copy
import logging

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

class InstrumentManager:
    def __init__(self):
        self.instruments = []
        self.next_id = 1
        self.selected_instrument_id = None

    def create_instrument(self, name="Novo Instrumento"):
        inst = {
            "id": self.next_id,
            "name": name,
            "midi_port": 1,
            "midi_channel": 1,
            "default_note": "C3",
            "track_values": [],
            "note_rows": [],
            "ccs": [],
        }
        for slot in range(1, 9):
            inst["track_values"].append({
                "slot": slot,
                "type": "Empty",
                "label": "",
                "cc": 0,
                "track_control": "",
            })
        for slot in range(1, 9):
            inst["note_rows"].append({
                "slot": slot,
                "name": "",
                "note": "",
                "velocity": 100,
            })
        for slot in range(1, 9):
            inst["ccs"].append({
                "slot": slot,
                "label": "",
                "cc": 0,
                "default": 0,
            })
        self.instruments.append(inst)
        self.selected_instrument_id = self.next_id
        self.next_id += 1
        return inst

    def duplicate_instrument(self, instrument_id):
        import copy
        inst = self.get_instrument_by_id(instrument_id)
        if inst:
            new_inst = copy.deepcopy(inst)
            new_inst["id"] = self.next_id
            new_inst["name"] += " (Cópia)"
            self.instruments.append(new_inst)
            self.selected_instrument_id = self.next_id
            self.next_id += 1
            return new_inst
        return None

    def delete_instrument(self, instrument_id):
        self.instruments = [i for i in self.instruments if i["id"] != instrument_id]
        if self.selected_instrument_id == instrument_id:
            self.selected_instrument_id = self.instruments[0]["id"] if self.instruments else None

    def select_instrument(self, instrument_id):
        if any(i["id"] == instrument_id for i in self.instruments):
            self.selected_instrument_id = instrument_id

    def get_instrument_by_id(self, instrument_id):
        return next((i for i in self.instruments if i["id"] == instrument_id), None)

    def list_instruments(self, search=""):
        return [
            i for i in self.instruments
            if search.lower() in i["name"].lower() or search in str(i["id"])
        ]

class UndoRedoManager:
    def __init__(self, manager):
        self.manager = manager
        self.undo_stack = []
        self.redo_stack = []

    def snapshot(self):
        self.undo_stack.append(copy.deepcopy(self.manager.instruments))
        self.redo_stack.clear()

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(copy.deepcopy(self.manager.instruments))
            self.manager.instruments = self.undo_stack.pop()
            if self.manager.selected_instrument_id not in [i["id"] for i in self.manager.instruments]:
                self.manager.selected_instrument_id = (
                    self.manager.instruments[0]["id"] if self.manager.instruments else None
                )

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(copy.deepcopy(self.manager.instruments))
            self.manager.instruments = self.redo_stack.pop()
            if self.manager.selected_instrument_id not in [i["id"] for i in self.manager.instruments]:
                self.manager.selected_instrument_id = (
                    self.manager.instruments[0]["id"] if self.manager.instruments else None
                )

def validate_instrument(inst, all_instruments):
    errors = []
    if not inst["name"]:
        errors.append("O nome do instrumento não pode ser vazio.")
    elif sum(i["name"] == inst["name"] for i in all_instruments) > 1:
        errors.append("Nome de instrumento duplicado.")
    if inst["midi_port"] not in range(1, 13):
        errors.append("Porta MIDI deve ser de 1 a 12.")
    if inst["midi_channel"] not in range(1, 17):
        errors.append("Canal MIDI deve ser de 1 a 16.")
    if not inst["default_note"]:
        errors.append("Nota padrão não pode ser vazia.")
    used_ccs = set()
    for tv in inst.get("track_values", []):
        if tv["type"] == "MidiCC":
            if tv["cc"] in used_ccs:
                errors.append(f"CC duplicado no Track Value Slot {tv['slot']}")
            used_ccs.add(tv["cc"])
        if tv["type"] == "TrackControl" and not tv["track_control"]:
            errors.append(f"Track Control não selecionado no Slot {tv['slot']}")
    for nr in inst.get("note_rows", []):
        if nr["name"] and not nr["note"]:
            errors.append(f"Note Row '{nr['name']}' está sem nota definida.")
    used_ccs_cc = set()
    for cc in inst.get("ccs", []):
        if cc["cc"] in used_ccs_cc and cc["cc"] != 0:
            errors.append(f"CC duplicado no CC Slot {cc['slot']}")
        if cc["cc"] != 0:
            used_ccs_cc.add(cc["cc"])
    return errors

def track_values_panel(instrument):
    st.subheader("Track Values")
    for tv in instrument["track_values"]:
        with st.expander(f"Slot {tv['slot']}"):
            tv["label"] = st.text_input(f"Label (Slot {tv['slot']})", tv.get("label", ""), key=f"label_{tv['slot']}")
            tv_type = st.selectbox(
                "Tipo",
                ["MidiCC", "TrackControl", "Empty"],
                index={"MidiCC":0,"TrackControl":1,"Empty":2}[tv.get("type","Empty")],
                key=f"type_{tv['slot']}"
            )
            tv["type"] = tv_type
            if tv_type == "MidiCC":
                tv["cc"] = st.number_input("CC #", 0, 127, tv.get("cc",0), key=f"cc_{tv['slot']}")
            elif tv_type == "TrackControl":
                tv["track_control"] = st.selectbox(
                    "Track Control",
                    [
                        "pgm", "quant%", "note%", "noteC", "velo%", "veloC",
                        "leng%", "tbase", "xpos", "octave", "knob1", "knob2"
                    ],
                    index=[
                        "pgm", "quant%", "note%", "noteC", "velo%", "veloC",
                        "leng%", "tbase", "xpos", "octave", "knob1", "knob2"
                    ].index(tv.get("track_control","pgm")),
                    key=f"tc_{tv['slot']}"
                )

def note_rows_panel(instrument):
    st.subheader("Note Rows")
    for nr in instrument["note_rows"]:
        with st.expander(f"Slot {nr['slot']}"):
            nr["name"] = st.text_input(
                f"Nome (Slot {nr['slot']})", nr.get("name", ""), key=f"nrname_{nr['slot']}"
            )
            nr["note"] = st.text_input(
                "Nota (ex: C3, D#4)", nr.get("note", ""), key=f"nrnote_{nr['slot']}"
            )
            nr["velocity"] = st.number_input(
                "Velocidade", 1, 127, nr.get("velocity", 100), key=f"nrvelo_{nr['slot']}"
            )

def ccs_panel(instrument):
    st.subheader("CCs")
    for cc in instrument["ccs"]:
        with st.expander(f"Slot {cc['slot']}"):
            cc["label"] = st.text_input(
                f"Label (Slot {cc['slot']})", cc.get("label", ""), key=f"cclabel_{cc['slot']}"
            )
            cc["cc"] = st.number_input(
                "CC #", 0, 127, cc.get("cc",0), key=f"ccnum_{cc['slot']}"
            )
            cc["default"] = st.number_input(
                "Valor Padrão", 0, 127, cc.get("default",0), key=f"ccdef_{cc['slot']}"
            )

# ----------- INICIALIZAÇÃO & PAINEL CKIEditor -----------

if "manager" not in st.session_state:
    st.session_state["manager"] = InstrumentManager()
if "undo_redo" not in st.session_state:
    st.session_state["undo_redo"] = UndoRedoManager(st.session_state["manager"])
manager = st.session_state["manager"]
undo_redo = st.session_state["undo_redo"]
 
# Todas as funções e módulos integrados em um só arquivo.
# Inclui: importação/exportação MIDI & Cirklon, IA polifônica, manipulação, automação multi-CC, piano roll, preview, validação, e FastAPI backend para integração.

import mido
import torch
import torch.nn as nn
import torch.nn.functional as F # Import F for activation functions
import struct
import json
import tempfile
from midi2audio import FluidSynth
from collections import defaultdict
import os # Import os for file operations

# -------- MIDI Parser --------
def midi_to_poly_dataset(midi_file):
    try:
        if hasattr(midi_file, "read"):
            mid = mido.MidiFile(file=midi_file)
        else:
            mid = mido.MidiFile(midi_file)
        
        all_notes_for_ai = [] # List to store all notes from all tracks
        
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
                                start_tick, # Store absolute tick for sorting
                                msg.note,
                                msg.velocity,
                                duration,
                                i # track index
                            ])

            # Handle notes that are still active at the end of the track
            for (note, channel), start_tick in active_notes.items():
                duration = abs_tick - start_tick # Assume duration until end of track
                if duration > 0:
                    all_notes_for_ai.append([
                        start_tick, # Store absolute tick for sorting
                        note,
                        64, # Default velocity for sustained notes
                        duration,
                        i # track index
                    ])

        # Sort all notes by absolute tick for interleaved sequence
        all_notes_for_ai.sort(key=lambda x: x[0])

        # Convert absolute ticks to delta ticks for the AI model
        final_dataset = []
        last_abs_tick = 0
        for note_event in all_notes_for_ai:
            abs_tick, note, velocity, duration, track_idx = note_event
            delta_tick = abs_tick - last_abs_tick
            final_dataset.append([delta_tick, note, velocity, duration, track_idx])
            last_abs_tick = abs_tick
        
        return np.array(final_dataset) if final_dataset else np.array([])
    except (OSError, ValueError) as e:
        st.error(f"Erro ao processar arquivo MIDI: {e}")
        return np.array([])
    except Exception as e:
        logger.exception("Unexpected error processing MIDI")
        st.error("Erro inesperado ao processar arquivo MIDI.")
        return np.array([])

def notes_to_midi_file(notes, tempo=120):
    mid = mido.MidiFile()
    
    # Create tracks in mido.MidiFile
    # Determine the maximum track index to create enough tracks
    max_track_idx = 0
    if notes:
        for track_data in notes:
            if track_data.size > 0:
                max_track_idx = max(max_track_idx, int(track_data[:, 4].max()))
    
    for i in range(max_track_idx + 1):
        mid.add_track()

    # Add notes to mido tracks
    for track_idx, track_data in enumerate(notes):
        # We need to convert delta_tick to absolute_tick for sorting and for note_on/off messages.
        cumulative_tick = 0
        absolute_notes_data = []
        for note_event in track_data:
            delta_tick, note, velocity, duration, _ = note_event
            cumulative_tick += delta_tick # Accumulate delta_tick
            absolute_notes_data.append([cumulative_tick, note, velocity, duration, track_idx])

        # Now sort by absolute_tick
        if len(absolute_notes_data) > 0:
            sorted_track_data = np.array(absolute_notes_data)[np.argsort(np.array(absolute_notes_data)[:, 0])]
        else:
            sorted_track_data = []
        
        # Create a list of (tick, message) pairs for this track
        track_messages = []
        for note_event in sorted_track_data:
            start_tick, note, velocity, duration, _ = note_event
            
            # Note On message
            track_messages.append((start_tick, mido.Message("note_on", note=int(note), velocity=int(velocity))))
            
            # Note Off message
            end_tick = start_tick + duration
            track_messages.append((end_tick, mido.Message("note_off", note=int(note), velocity=int(velocity))))

        # Sort all messages by tick and then convert to delta time
        track_messages.sort(key=lambda x: x[0])
        
        last_tick = 0
        for tick, msg in track_messages:
            delta_time = int(tick - last_tick)
            msg.time = delta_time # Set delta time for the message
            mid.tracks[track_idx].append(msg)
            last_tick = tick

    # Add tempo message to the first track (or a dedicated tempo track)
    if mid.tracks:
        mid.tracks[0].insert(0, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo)))

    return mid

# -------- IA Polifônica --------
class PolyTransformer(nn.Module):
    def __init__(self, n_tracks=4, d_model=128, num_lstm_layers=2, dropout_rate=0.2):
        super().__init__()
        # Embeddings for categorical features
        self.note_embed = nn.Embedding(128, d_model // 2) # Note (0-127)
        self.track_embed = nn.Embedding(n_tracks, d_model // 4) # Track index

        # Linear layer for continuous features: delta_tick, velocity, duration
        self.continuous_linear = nn.Linear(3, d_model // 4) # For delta_tick, velocity, duration

        # LSTM input size will be the sum of embedding and linear output sizes
        lstm_input_size = (d_model // 2) + (d_model // 4) + (d_model // 4) # Should sum to d_model

        self.lstm = nn.LSTM(lstm_input_size, d_model, num_layers=num_lstm_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)

        # Separate output layers for each feature
        self.output_delta_tick = nn.Linear(d_model, 1)
        self.output_note = nn.Linear(d_model, 128) # Output logits for 128 notes
        self.output_velocity = nn.Linear(d_model, 1)
        self.output_duration = nn.Linear(d_model, 1)
        self.output_track = nn.Linear(d_model, n_tracks) # Output logits for n_tracks

    def forward(self, x):
        # x is expected to be (batch_size, seq_len, 5)
        # where 5 features are [delta_tick, note, velocity, duration, track]

        # Separate features
        delta_tick_data = x[:, :, 0].unsqueeze(-1) # Keep as float for linear layer
        note_data = x[:, :, 1].long()       # Convert to long for embedding
        velocity_data = x[:, :, 2].unsqueeze(-1) # Keep as float for linear layer
        duration_data = x[:, :, 3].unsqueeze(-1) # Keep as float for linear layer
        track_data = x[:, :, 4].long()      # Convert to long for embedding

        # Embed categorical features
        note_embedded = self.note_embed(note_data)
        track_embedded = self.track_embed(track_data)

        # Process continuous features
        continuous_features = torch.cat([delta_tick_data, velocity_data, duration_data], dim=-1)
        continuous_projected = self.continuous_linear(continuous_features.float()) # Ensure float

        # Concatenate all processed features for LSTM input
        combined_input = torch.cat([note_embedded, track_embedded, continuous_projected], dim=-1)

        lstm_out, _ = self.lstm(combined_input)
        lstm_out = self.dropout(lstm_out) # Apply dropout

        # Predict each feature separately
        pred_delta_tick = F.relu(self.output_delta_tick(lstm_out)) + 1 # Ensure positive and at least 1
        pred_note_logits = self.output_note(lstm_out) # Logits for notes
        pred_velocity = torch.sigmoid(self.output_velocity(lstm_out)) * 127 # Scale to 0-127
        pred_duration = F.relu(self.output_duration(lstm_out)) + 1 # Ensure positive and at least 1
        pred_track_logits = self.output_track(lstm_out) # Logits for tracks

        return pred_delta_tick, pred_note_logits, pred_velocity, pred_duration, pred_track_logits

def train_poly_model(notes_data, params):
    try:
        if notes_data.shape[0] == 0:
            st.warning("Nenhuma nota para treinar o modelo de IA.")
            return None

        data_tensor = torch.from_numpy(notes_data.astype(float)).float().unsqueeze(0) # Add batch dimension

        # n_tracks for PolyTransformer should be derived from the data
        n_tracks_in_data = int(notes_data[:, 4].max()) + 1 if notes_data.shape[0] > 0 else 1

        model = PolyTransformer(n_tracks=n_tracks_in_data, d_model=params.get("d_model", 128), num_lstm_layers=params.get("num_lstm_layers", 2), dropout_rate=params.get("dropout_rate", 0.2))
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        
        # Define separate loss functions for each output type
        criterion_continuous = nn.MSELoss() # For delta_tick, velocity, duration
        criterion_note = nn.CrossEntropyLoss() # For note (classification over 128 notes)
        criterion_track = nn.CrossEntropyLoss() # For track (classification over n_tracks)

        for epoch in range(params["epochs"]):
            optimizer.zero_grad()
            
            # Predict outputs
            pred_delta_tick, pred_note_logits, pred_velocity, pred_duration, pred_track_logits = model(data_tensor)

            # Extract targets
            target_delta_tick = data_tensor[:, :, 0].unsqueeze(-1)
            target_note = data_tensor[:, :, 1].long()
            target_velocity = data_tensor[:, :, 2].unsqueeze(-1)
            target_duration = data_tensor[:, :, 3].unsqueeze(-1)
            target_track = data_tensor[:, :, 4].long()

            # Calculate losses
            loss_delta_tick = criterion_continuous(pred_delta_tick, target_delta_tick)
            loss_note = criterion_note(pred_note_logits.view(-1, 128), target_note.view(-1)) # Flatten for CrossEntropyLoss
            loss_velocity = criterion_continuous(pred_velocity, target_velocity)
            loss_duration = criterion_continuous(pred_duration, target_duration)
            loss_track = criterion_track(pred_track_logits.view(-1, n_tracks_in_data), target_track.view(-1)) # Flatten for CrossEntropyLoss

            # Combine losses (can use weighted sum if needed)
            total_loss = loss_delta_tick + loss_note + loss_velocity + loss_duration + loss_track
            
            total_loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                st.write(f"Epoch {epoch+1}/{params['epochs']}, Total Loss: {total_loss.item():.4f}")

        torch.save(model.state_dict(), params["model_path"])
        return model
    except (RuntimeError, ValueError) as e:
        st.error(f"Erro ao treinar o modelo de IA: {e}")
        return None
    except Exception as e:
        logger.exception("Unexpected error training model")
        st.error("Erro inesperado ao treinar o modelo de IA.")
        return None

def infer_poly_sequence(model, seq_len=512, n_tracks=4, prime_sequence=None, randomness_temp=1.0):
    try:
        model.eval() # Set model to evaluation mode
        
        # If no prime_sequence is provided, start with a dummy note
        if prime_sequence is None or prime_sequence.shape[0] == 0:
            # [delta_tick, note, velocity, duration, track]
            prime_sequence = np.array([[100, 60, 80, 100, 0]], dtype=float) # Start with a single dummy note for track 0
        
        current_sequence = prime_sequence.copy()
        
        for i in range(seq_len):
            # Take the last note from the current sequence as input for prediction
            input_note = torch.from_numpy(current_sequence[-1:].astype(float)).float().unsqueeze(0) # (1, 1, 5)

            with torch.no_grad():
                pred_delta_tick, pred_note_logits, pred_velocity, pred_duration, pred_track_logits = model(input_note)

            # Extract predicted features and ensure they are within valid ranges
            pred_delta_tick_val = max(1, int(pred_delta_tick.item()))
            
            pred_note_prob = F.softmax(pred_note_logits / randomness_temp, dim=-1)
            pred_note_val = torch.multinomial(pred_note_prob.squeeze(0), 1).item()
            
            pred_velocity_val = int(np.clip(pred_velocity.item(), 0, 127))
            
            pred_duration_val = max(1, int(pred_duration.item()))
            
            # Sample track from logits
            pred_track_prob = F.softmax(pred_track_logits / randomness_temp, dim=-1)
            pred_track_val = torch.multinomial(pred_track_prob.squeeze(0), 1).item()
            
            new_note_features = np.array([pred_delta_tick_val, pred_note_val, pred_velocity_val, pred_duration_val, pred_track_val], dtype=float)
            current_sequence = np.vstack([current_sequence, new_note_features])
        
        # Now, reconstruct the separate tracks from the single interleaved sequence
        out_sequences_by_track = defaultdict(list)
        for note_event in current_sequence:
            out_sequences_by_track[int(note_event[4])].append(note_event)
        
        # Convert to list of numpy arrays, sorted by track index
        final_out_sequences = [np.array(out_sequences_by_track[k]) for k in sorted(out_sequences_by_track.keys())] # Ensure sorted keys for consistent track order
        
        return final_out_sequences
    except (RuntimeError, ValueError) as e:
        st.error(f"Erro ao inferir sequência com o modelo de IA: {e}")
        return []
    except Exception as e:
        logger.exception("Unexpected error inferring sequence")
        st.error("Erro inesperado ao inferir sequência com o modelo de IA.")
        return []

# -------- Algoritmos de Manipulação MIDI --------
def breakbeat_poly(notes):
    res = []
    for track in notes:
        idx = np.arange(len(track))
        np.random.shuffle(idx)
        res.append(track[idx])
    return res

def microtiming_poly(notes, max_shift=5):
    """Apply random timing shifts while preserving note order."""
    result = []
    for track in notes:
        if len(track) == 0:
            result.append(track)
            continue

        # Convert delta ticks to absolute ticks for easier shifting
        abs_events = []
        cumulative = 0
        for evt in track:
            cumulative += evt[0]
            abs_events.append([cumulative, evt[1], evt[2], evt[3], evt[4]])

        # Apply random shifts and clamp at zero
        shifted = []
        for tick, note, vel, dur, trk in abs_events:
            shifted_tick = max(0, tick + np.random.randint(-max_shift, max_shift + 1))
            shifted.append([shifted_tick, note, vel, dur, trk])

        # Re-sort after shifting
        shifted.sort(key=lambda x: x[0])

        # Convert back to delta ticks
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
    ccs = [[tick, channel, cc_num, int(val)] for tick, val in enumerate(values)]
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
    ccs = [[tick, channel, cc_num, int(val)] for tick, val in enumerate(values)]
    return ccs

def randomize_automation(length, cc_num, minv=0, maxv=127, channel=0):
    values = np.random.randint(minv, maxv+1, length)
    ccs = [[tick, channel, cc_num, int(val)] for tick, val in enumerate(values)]
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
                    if n[3] > 0: # Check if duration is positive
                        f.write(struct.pack('>I', int(n[0])))
                        f.write(struct.pack('B', int(n[1])))
                        f.write(struct.pack('B', int(n[2])))
                        f.write(struct.pack('B', int(n[3])))
                        f.write(struct.pack('B', t))
        return out_path
    except (OSError, struct.error) as e:
        st.error(f"Erro ao exportar CKC: {e}")
        return None
    except Exception as e:
        logger.exception("Unexpected error exporting CKC")
        st.error("Erro inesperado ao exportar CKC.")
        return None

def export_cki(notes, out_path):
    try:
        with open(out_path, "w") as f:
            f.write("CKI,1\n")
            for t, track in enumerate(notes):
                for n in track:
                    if n[3] > 0: # Check if duration is positive
                        line = f"{int(n[0])},{int(n[1])},{int(n[2])},{int(n[3])},{t}\n" # Use n[3] for duration
                        f.write(line)
        return out_path
    except OSError as e:
        st.error(f"Erro ao exportar CKI: {e}")
        return None
    except Exception as e:
        logger.exception("Unexpected error exporting CKI")
        st.error("Erro inesperado ao exportar CKI.")
        return None

def export_p3pattern_json(notes, out_path):
    try:
        p3 = []
        for t, track in enumerate(notes):
            for n in track:
                if n[3] > 0: # Check if duration is positive
                    ev = {
                        "step": int(n[0]),
                        "note": int(n[1]),
                        "velocity": int(n[2]),
                        "length": int(n[3]), # Use n[3] for duration
                        "track": t
                    }
                    p3.append(ev)
        with open(out_path, "w") as f:
            json.dump({"p3_pattern":p3}, f, indent=2)
        return out_path
    except OSError as e:
        st.error(f"Erro ao exportar P3 Pattern JSON: {e}")
        return None
    except Exception as e:
        logger.exception("Unexpected error exporting P3 pattern")
        st.error("Erro inesperado ao exportar P3 Pattern JSON.")
        return None

def export_auxrows(notes, out_path):
    try:
        aux = []
        for t, track in enumerate(notes):
            for n in track:
                if n[3] > 0: # Check if duration is positive
                    aux.append({
                        "tick": int(n[0]),
                        "cc_num": 74,
                        "cc_val": int(n[2]), # Assuming velocity can be used as CC value for now
                        "track": t
                    })
        with open(out_path, "w") as f:
            json.dump({"aux_rows":aux}, f, indent=2)
        return out_path
    except OSError as e:
        st.error(f"Erro ao exportar Aux Rows: {e}")
        return None
    except Exception as e:
        logger.exception("Unexpected error exporting Aux Rows")
        st.error("Erro inesperado ao exportar Aux Rows.")
        return None

# -------- Importação Cirklon CKC/CKI/AuxRows --------
def import_ckc_bin(path):
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            n_patterns = struct.unpack('>H', f.read(2))[0]
            _ = f.read(2)
            n_events = struct.unpack('>H', f.read(2))[0]
            notes = []
            for _ in range(n_events):
                tick = struct.unpack('>I', f.read(4))[0]
                note = struct.unpack('B', f.read(1))[0]
                vel = struct.unpack('B', f.read(1))[0]
                duration = struct.unpack('B', f.read(1))[0] # Read duration
                track = struct.unpack('B', f.read(1))[0]
                notes.append([tick, note, vel, duration, track]) # Store duration
        tracks = defaultdict(list)
        for n in notes:
            tracks[n[4]].append(n)
        return [np.array(tracks[k]) for k in sorted(tracks)]
    except (OSError, struct.error) as e:
        st.error(f"Erro ao importar CKC: {e}")
        return []
    except Exception as e:
        logger.exception("Unexpected error importing CKC")
        st.error("Erro inesperado ao importar CKC.")
        return []

def import_cki_txt(path):
    try:
        notes = []
        with open(path) as f:
            for line in f:
                if line.startswith("CKI"): continue
                vals = [int(x) for x in line.strip().split(",")]
                # Assuming CKI format is: tick, note, velocity, duration, track
                if len(vals) == 5:
                    notes.append(vals)
                else:
                    st.warning(f"Linha CKI com formato inesperado: {line.strip()}")
        tracks = defaultdict(list)
        for n in notes:
            tracks[n[4]].append(n)
        return [np.array(tracks[k]) for k in sorted(tracks)]
    except (OSError, ValueError) as e:
        st.error(f"Erro ao importar CKI: {e}")
        return []
    except Exception as e:
        logger.exception("Unexpected error importing CKI")
        st.error("Erro inesperado ao importar CKI.")
        return []

def import_auxrows_json(path):
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("aux_rows", [])
    except (OSError, json.JSONDecodeError) as e:
        st.error(f"Erro ao importar Aux Rows JSON: {e}")
        return []
    except Exception as e:
        logger.exception("Unexpected error importing Aux Rows JSON")
        st.error("Erro inesperado ao importar Aux Rows JSON.")
        return []

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
        # For validation, we need absolute ticks. Convert delta_tick to absolute_tick.
        cumulative_tick = 0
        for n in track:
            cumulative_tick += n[0] # Add delta_tick
            all_ticks.setdefault(cumulative_tick, []).append((t, n[1]))

    for tick, events in all_ticks.items():
        if len(events) > max_poly:
            errors.append(f"Polifonia excessiva ({len(events)}) no tick {tick}")
    cc_map = {cc[0]: cc[3] for cc in ccs if cc[2]==1}
    for t, track in enumerate(notes):
        # For CC validation, we need absolute ticks. Convert delta_tick to absolute_tick.
        cumulative_tick = 0
        for n in track:
            cumulative_tick += n[0]
            cc_val = cc_map.get(cumulative_tick, 127)
            if cc_val < 20 and n[2] > 60:
                errors.append(f"Track {t} nota em tick {cumulative_tick} velocity alta ({n[2]}) com CC1 muito baixo ({cc_val})")
    return errors

# -------- UI Piano Roll Drag&Drop --------
def pianoroll_dragdrop(notes, width=1000, height=400, ticks=512, pitch_min=0, pitch_max=127):
    st.subheader("Piano Roll Polifônico Drag & Drop")
    objects = []
    for track, notes_track in enumerate(notes):
        # Convert delta_tick to absolute_tick for display on piano roll
        cumulative_tick = 0
        for n in notes_track:
            delta_tick, note, velocity, duration, _ = n
            cumulative_tick += delta_tick
            
            if duration > 0:
                left = cumulative_tick / ticks * width
                # Calculate width based on duration
                note_width = max(5, duration / ticks * width) # Minimum width of 5 pixels
                top = height - ((note-pitch_min)/(pitch_max-pitch_min) * height)
                obj = {
                    "type": "rect",
                    "left": left,
                    "top": top,
                    "width": note_width,
                    "height": 10,
                    "fill": f"rgba({60+track*40},150,255,0.7)",
                    "track": track
                }
                objects.append(obj)
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 255, 0.3)",
        background_color="white",
        update_streamlit=True,
        height=height,
        width=width,
        drawing_mode="rect",
        initial_drawing={"version":"4.4.0","objects": objects},
        key="pianoroll"
    )
    if canvas_result.json_data and "objects" in canvas_result.json_data:
        midi_events = []
        # When drawing, we get absolute positions. Need to convert back to delta_tick.
        # This is tricky because we need to preserve track order and calculate delta_tick correctly.
        # For simplicity, let's assume the user is drawing new notes, and we'll convert them to delta_tick.
        # This might not be perfect for editing existing notes.
        
        # Group events by track first
        drawn_notes_by_track = defaultdict(list)
        for obj in canvas_result.json_data["objects"]:
            x, y, w = obj["left"], obj["top"], obj["width"]
            abs_tick = int(x / width * ticks)
            note = int(pitch_max - (y / height * (pitch_max-pitch_min)))
            velocity = 100 # Default velocity for drawn notes
            duration = int(w / width * ticks) # Calculate duration from width
            track_idx = int(obj.get("track", 0))
            drawn_notes_by_track[track_idx].append([abs_tick, note, velocity, duration, track_idx])
        
        result_notes = []
        for track_idx in sorted(drawn_notes_by_track.keys()):
            track_data = drawn_notes_by_track[track_idx]
            track_data.sort(key=lambda x: x[0]) # Sort by absolute tick
            
            converted_track = []
            last_abs_tick = 0
            for note_event in track_data:
                abs_tick, note, velocity, duration, track = note_event
                delta_tick = abs_tick - last_abs_tick
                converted_track.append([delta_tick, note, velocity, duration, track])
                last_abs_tick = abs_tick
            result_notes.append(np.array(converted_track))

        return result_notes
    return notes

# -------- UI Automação Multi-CC Avançada --------
def automation_multicc_advanced(cc_events, cc_list=[1,74,10], length=512):
    st.subheader("Editor gráfico multi-CC com curvas simultâneas")
    fig = go.Figure()
    colors = ["red", "green", "blue", "orange", "purple", "cyan", "black"]
    for idx, cc in enumerate(cc_list):
        evs = cc_events.get(cc, [])
        if not evs: continue
        ticks = [ev[0] for ev in evs]
        vals = [ev[1] for ev in evs]
        fig.add_trace(go.Scatter(
            x=ticks, y=vals, mode='lines+markers',
            name=f"CC{cc}", line=dict(color=colors[idx%len(colors)])
        ))
    fig.update_layout(
        xaxis=dict(range=[0,length], title="Tick"),
        yaxis=dict(range=[0,127], title="Valor CC"),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    edited = {}
    for cc in cc_list:
        df = pd.DataFrame(cc_events.get(cc, []), columns=["tick", "value"])
        st.write(f"Edição tabular para CC{cc}")
        new_df = st.data_editor(df, use_container_width=True, num_rows="dynamic")
        edited[cc] = new_df.values.tolist()
    return edited

# -------- Preview MIDI com SoundFont --------
def midi_preview_sf2(notes, sf2_path, tempo=120):
    try:
        # Convert the notes structure to a mido.MidiFile object
        midi_file = notes_to_midi_file(notes, tempo=tempo)
        
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
            midi_file.save(tmp.name)
            tmp.flush()
            audio_path = tmp.name.replace(".mid", ".wav")
            FluidSynth(sf2_path).midi_to_audio(tmp.name, audio_path)
            with open(audio_path, "rb") as f:
                st.audio(f, format="audio/wav")
    except (OSError, ValueError) as e:
        st.error(f"Erro ao gerar preview de áudio: {e}")
    except Exception as e:
        logger.exception("Unexpected error generating audio preview")
        st.error("Erro inesperado ao gerar preview de áudio.")

# -------- MAIN APP --------

st.title("Sistema IA MIDI Cirklon - Completo e Integrado")

# Model persistence
if "trained_model" not in st.session_state:
    st.session_state["trained_model"] = None

if "notes" not in st.session_state:
    st.session_state["notes"] = []

if "tempo" not in st.session_state:
    st.session_state["tempo"] = 120

tab = st.sidebar.radio(
    "Etapa",
    [
        "CKIEditor (Instrumentos)",
        "Importar MIDI/CKC/CKI",
        "Editar Piano Roll",
        "Automação Avançada",
        "Treinar IA",
        "Gerar Música",
        "Manipular MIDI",
        "Exportar Cirklon",
        "Preview Audio",
        "Validação Musical",
        "Carregar/Salvar Modelo IA",
    ]
)

if tab == "CKIEditor (Instrumentos)":
    st.sidebar.header("Biblioteca de Instrumentos CKI")
    search = st.sidebar.text_input("Buscar instrumento")
    filtered = manager.list_instruments(search)
    for inst in filtered:
        cols = st.sidebar.columns([6, 1, 1, 1])
        is_selected = inst["id"] == manager.selected_instrument_id
        if cols[0].button(
            f"{'▶️ ' if is_selected else ''}{inst['name']} (ID: {inst['id']})",
            key=f"select_{inst['id']}"
        ):
            manager.select_instrument(inst["id"])
        if cols[1].button("🗐", key=f"dup_{inst['id']}"):
            undo_redo.snapshot()
            manager.duplicate_instrument(inst["id"])
        if cols[2].button("🗑️", key=f"del_{inst['id']}"):
            undo_redo.snapshot()
            manager.delete_instrument(inst["id"])
    if st.sidebar.button("Novo instrumento"):
        undo_redo.snapshot()
        manager.create_instrument()
    uc, rc = st.sidebar.columns(2)
    if uc.button("↩️ Undo"):
        undo_redo.undo()
    if rc.button("↪️ Redo"):
        undo_redo.redo()

    instrument = manager.get_instrument_by_id(manager.selected_instrument_id)
    if instrument:
        st.subheader(f"Edição: {instrument['name']} (ID: {instrument['id']})")
        instrument["name"] = st.text_input("Nome", instrument["name"])
        instrument["midi_port"] = st.number_input("Porta MIDI", 1, 12, instrument["midi_port"])
        instrument["midi_channel"] = st.number_input("Canal MIDI", 1, 16, instrument["midi_channel"])
        instrument["default_note"] = st.text_input("Nota padrão", instrument["default_note"])
        errors = validate_instrument(instrument, manager.instruments)
        if errors:
            for err in errors:
                st.warning(err)
        else:
            st.success("Instrumento válido!")
        with st.expander("Track Values", expanded=False):
            track_values_panel(instrument)
        with st.expander("Note Rows", expanded=False):
            note_rows_panel(instrument)
        with st.expander("CCs", expanded=False):
            ccs_panel(instrument)
        st.markdown("---")
    else:
        st.info("Nenhum instrumento selecionado.")
elif tab == "Importar MIDI/CKC/CKI":
    up_mode = st.selectbox("Tipo de importação", ["MIDI", "CKC", "CKI"])
    if up_mode == "MIDI":
        files = st.file_uploader("Arquivos MIDI (múltiplos)", type=["mid","midi"], accept_multiple_files=True)
        if files:
            all_flattened_notes = []
            for file in files:
                try:
                    flattened_notes = midi_to_poly_dataset(file)
                    if flattened_notes.shape[0] > 0:
                        all_flattened_notes.append(flattened_notes)
                except (OSError, ValueError) as e:
                    st.warning(f"Não foi possível processar {file.name}: {e}")
                except Exception as e:
                    logger.exception("Unexpected error importing MIDI file")
                    st.warning(f"Não foi possível processar {file.name} devido a um erro inesperado.")
            
            if all_flattened_notes:
                combined_flattened_notes = np.concatenate(all_flattened_notes, axis=0)
                
                # Reconstruct tracks from the combined flattened array for UI/other functions
                reconstructed_tracks = defaultdict(list)
                for note_event in combined_flattened_notes:
                    reconstructed_tracks[int(note_event[4])].append(note_event)
                
                st.session_state["notes"] = [np.array(reconstructed_tracks[k]) for k in sorted(reconstructed_tracks.keys())] # Ensure sorted keys for consistent track order
                st.success(f"Importação de {len(files)} arquivos MIDI realizada.")
            else:
                st.warning("Nenhum arquivo MIDI válido foi importado.")

    elif up_mode == "CKC":
        file = st.file_uploader("Arquivo CKC", type=["ckc"])
        if file:
            try:
                with tempfile.NamedTemporaryFile(suffix=".ckc", delete=False) as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name
                st.session_state["notes"] = import_ckc_bin(tmp_path)
                st.success("Importação CKC realizada.")
                os.remove(tmp_path) # Clean up temp file
            except OSError as e:
                st.error(f"Erro durante a importação CKC: {e}")
            except Exception as e:
                logger.exception("Unexpected error importing CKC file")
                st.error("Erro inesperado durante a importação CKC")
    elif up_mode == "CKI":
        file = st.file_uploader("Arquivo CKI", type=["cki"])
        if file:
            try:
                with tempfile.NamedTemporaryFile(suffix=".cki", delete=False) as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name
                st.session_state["notes"] = import_cki_txt(tmp_path)
                st.success("Importação CKI realizada.")
                os.remove(tmp_path) # Clean up temp file
            except OSError as e:
                st.error(f"Erro durante a importação CKI: {e}")
            except Exception as e:
                logger.exception("Unexpected error importing CKI file")
                st.error("Erro inesperado durante a importação CKI")
elif tab == "Editar Piano Roll":
    if "notes" in st.session_state and st.session_state["notes"]:
        notes_edit = pianoroll_dragdrop(st.session_state["notes"])
        st.session_state["notes"] = notes_edit
    else:
        st.warning("Por favor, importe um arquivo MIDI/CKC/CKI primeiro para editar o Piano Roll.")
elif tab == "Automação Avançada":
    if "notes" in st.session_state and st.session_state["notes"]:
        cc_list = st.multiselect("CCs para editar", [1,74,10,7,11], default=[1,74])
        # Ensure notes is not empty before trying to get max tick
        if st.session_state["notes"] and any(len(track) > 0 for track in st.session_state["notes"]):
            # When calculating length for automation, we need the maximum absolute tick.
            # The notes structure now contains delta_tick. So we need to convert.
            max_abs_tick = 0
            for track_data in st.session_state["notes"]:
                cumulative_tick = 0
                for note_event in track_data:
                    cumulative_tick += note_event[0] # Add delta_tick
                max_abs_tick = max(max_abs_tick, cumulative_tick)
            length = max_abs_tick
        else:
            length = 512 # Default length if no notes are loaded

        cc_events = {cc: lfo_automation(length, cc) for cc in cc_list}
        cc_events = automation_multicc_advanced(cc_events, cc_list, length)
        st.session_state["ccs"] = []
        for cc, events in cc_events.items():
            for tick, val in events:
                st.session_state["ccs"].append([tick,0,cc,val])
    else:
        st.warning("Por favor, importe um arquivo MIDI/CKC/CKI primeiro para configurar a automação.")
elif tab == "Treinar IA":
    if "notes" in st.session_state and st.session_state["notes"]:
        params = {"lr": 0.001, "epochs": 10, "model_path": "poly_model.pt", "d_model": 128, "num_lstm_layers": 2, "dropout_rate": 0.2} # Added d_model, num_lstm_layers, dropout_rate
        
        # Flatten notes for training
        all_notes_flat_for_training = np.concatenate(st.session_state["notes"], axis=0) if st.session_state["notes"] else np.array([])
        
        trained_model = train_poly_model(all_notes_flat_for_training, params)
        if trained_model:
            st.session_state["trained_model"] = trained_model
            st.success("Modelo treinado.")
        else:
            st.error("Falha no treinamento do modelo.")
    else:
        st.warning("Por favor, importe um arquivo MIDI/CKC/CKI para treinar o modelo de IA.")
elif tab == "Gerar Música":
    if "trained_model" in st.session_state and st.session_state["trained_model"] is not None:
        st.subheader("Parâmetros de Geração")
        # Parameters for generation
        seq_len = st.slider("Duração da Sequência (ticks)", 64, 2048, 512, step=64)
        n_tracks_gen = st.slider("Número de Trilhas a Gerar", 1, 8, 4)
        tempo_gen = st.slider("Andamento (BPM)", 60, 200, 120)
        randomness_temp = st.slider("Grau de Aleatoriedade (Temperatura)", 0.1, 2.0, 1.0, step=0.1)
        
        # Placeholder for future advanced parameters
        st.info("Parâmetros avançados como 'Intensidade de Complexidade', 'Estilo' e 'Instrumentação' serão implementados em futuras versões.")

        if st.button("Gerar Nova Música"):
            with st.spinner('Gerando música...'):
                # The infer_poly_sequence now returns a list of track arrays
                seq = infer_poly_sequence(st.session_state["trained_model"], seq_len=seq_len, n_tracks=n_tracks_gen, randomness_temp=randomness_temp)
                if seq:
                    st.session_state["notes"] = seq
                    st.session_state["tempo"] = tempo_gen # Store tempo for MIDI export
                    st.success("Música gerada.")
                else:
                    st.error("Falha na geração da música.")
    else:
        st.warning("Por favor, treine ou carregue um modelo de IA primeiro.")
elif tab == "Manipular MIDI":
    if "notes" in st.session_state and st.session_state["notes"]:
        mode = st.selectbox("Algoritmo", ["breakbeat", "microtiming"])
        try:
            if mode == "breakbeat":
                st.session_state["notes"] = breakbeat_poly(st.session_state["notes"])
            if mode == "microtiming":
                st.session_state["notes"] = microtiming_poly(st.session_state["notes"])
            st.success("Manipulação aplicada.")
        except ValueError as e:
            st.error(f"Erro durante a manipulação MIDI: {e}")
        except Exception as e:
            logger.exception("Unexpected error manipulating MIDI")
            st.error("Erro inesperado durante a manipulação MIDI")
    else:
        st.warning("Por favor, importe ou gere notas primeiro para manipular as notas.")
elif tab == "Exportar Cirklon":
    if "notes" in st.session_state and st.session_state["notes"]:
        try:
            # RE-EVALUATION: The export functions (export_ckc, export_cki, export_p3pattern_json, export_auxrows) expect absolute ticks.
            # The notes in st.session_state are in delta_tick format. So, conversion is needed here.
            notes_abs_tick_for_export = []
            for track_data in st.session_state["notes"]:
                cumulative_tick = 0
                abs_track_data = []
                for note_event in track_data:
                    delta_tick, note, velocity, duration, track_idx = note_event
                    cumulative_tick += delta_tick
                    abs_track_data.append([cumulative_tick, note, velocity, duration, track_idx])
                notes_abs_tick_for_export.append(np.array(abs_track_data))

            export_ckc(notes_abs_tick_for_export, "Pattern.ckc")
            export_cki(notes_abs_tick_for_export, "Pattern.cki")
            export_p3pattern_json(notes_abs_tick_for_export, "P3Pattern.json")
            export_auxrows(notes_abs_tick_for_export, "auxrows.json")
            st.success("Exportação realizada.")
            with open("Pattern.ckc", "rb") as f:
                st.download_button("Baixar CKC", f, file_name="Pattern.ckc")
            with open("Pattern.cki", "rb") as f:
                st.download_button("Baixar CKI", f, file_name="Pattern.cki")
        except OSError as e:
            st.error(f"Erro durante a exportação: {e}")
        except Exception as e:
            logger.exception("Unexpected error during export")
            st.error("Erro inesperado durante a exportação")
    else:
        st.warning("Por favor, importe ou gere notas primeiro para exportar para o Cirklon.")
elif tab == "Preview Audio":
    if "notes" in st.session_state and st.session_state["notes"]:
        sf2_file = st.file_uploader("SoundFont SF2 para preview", type=["sf2"])
        if sf2_file:
            # Save the uploaded SF2 to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".sf2", delete=False) as tmp_sf2:
                tmp_sf2.write(sf2_file.read())
                tmp_sf2_path = tmp_sf2.name
            
            # Use the tempo from session_state if available, otherwise default to 120
            current_tempo = st.session_state.get("tempo", 120)
            midi_preview_sf2(st.session_state["notes"], tmp_sf2_path, tempo=current_tempo)
            os.remove(tmp_sf2_path) # Clean up temp file
        else:
            st.info("Por favor, carregue um arquivo SoundFont (.sf2) para gerar o preview de áudio.")
    else:
        st.warning("Por favor, importe ou gere notas primeiro para gerar o preview de áudio.")

# -------- Validação Musical Avançada --------
elif tab == "Validação Musical":
    if "notes" in st.session_state and st.session_state["notes"]:
        # For validation, we need absolute ticks. Convert delta_tick to absolute_tick.
        notes_abs_tick_for_validation = []
        for track_data in st.session_state["notes"]:
            cumulative_tick = 0
            abs_track_data = []
            for note_event in track_data:
                delta_tick, note, velocity, duration, track_idx = note_event
                cumulative_tick += delta_tick
                abs_track_data.append([cumulative_tick, note, velocity, duration, track_idx])
            notes_abs_tick_for_validation.append(np.array(abs_track_data))

        # Default values for validation parameters
        scale_root = st.slider("Raiz da Escala", 0, 127, 60)
        scale_type = st.selectbox("Tipo de Escala", ["major", "minor", "dorian", "phrygian"])
        max_poly = st.slider("Polifonia Máxima", 1, 16, 8)
        pitch_min = st.slider("Range de Pitch Mínimo", 0, 127, 0)
        pitch_max = st.slider("Range de Pitch Máximo", 0, 127, 127)
        pitch_range = (pitch_min, pitch_max)

        # Ensure ccs is initialized, even if empty
        ccs_for_validation = st.session_state.get("ccs", [])

        try:
            errors = validate_advanced(notes_abs_tick_for_validation, ccs_for_validation, scale_root, scale_type, max_poly, pitch_range)
            if errors:
                st.error("Erros de Validação:")
                for error in errors:
                    st.write(f"- {error}")
            else:
                st.success("Nenhum erro de validação encontrado.")
        except ValueError as e:
            st.error(f"Erro durante a validação musical: {e}")
        except Exception as e:
            logger.exception("Unexpected error during musical validation")
            st.error("Erro inesperado durante a validação musical")
    else:
        st.warning("Por favor, importe ou gere notas primeiro para realizar a validação musical.")
elif tab == "Carregar/Salvar Modelo IA":
    st.subheader("Salvar Modelo Treinado")
    if st.session_state["trained_model"] is not None:
        model_name = st.text_input("Nome do arquivo do modelo (ex: meu_modelo.pt)", "poly_model.pt")
        if st.button("Salvar Modelo"):
            try:
                torch.save(st.session_state["trained_model"].state_dict(), model_name)
                st.success(f"Modelo salvo como {model_name}")
                with open(model_name, "rb") as f:
                    st.download_button("Baixar Modelo", f, file_name=model_name)
            except OSError as e:
                st.error(f"Erro ao salvar o modelo: {e}")
            except Exception as e:
                logger.exception("Unexpected error saving model")
                st.error("Erro inesperado ao salvar o modelo")
    else:
        st.info("Nenhum modelo treinado para salvar.")

    st.subheader("Carregar Modelo Treinado")
    uploaded_model_file = st.file_uploader("Carregar arquivo de modelo (.pt)", type=["pt"])
    if uploaded_model_file:
        try:
            # Determine n_tracks from current notes or default to 4 if no notes loaded
            # This is crucial for loading a model with the correct output_track dimension
            n_tracks_for_model = 4 # Default value
            if st.session_state["notes"] and any(len(track) > 0 for track in st.session_state["notes"]):
                # If notes are loaded, calculate max track index from them
                max_track_idx_in_notes = 0
                for track_data in st.session_state["notes"]:
                    if track_data.size > 0:
                        max_track_idx_in_notes = max(max_track_idx_in_notes, int(track_data[:, 4].max()))
                n_tracks_for_model = max_track_idx_in_notes + 1
            
            # Create a dummy model instance to load state_dict into
            loaded_model = PolyTransformer(n_tracks=n_tracks_for_model) # Use default d_model, num_lstm_layers, dropout_rate
            
            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_model_file:
                tmp_model_file.write(uploaded_model_file.read())
                tmp_model_path = tmp_model_file.name

            loaded_model.load_state_dict(torch.load(tmp_model_path))
            loaded_model.eval() # Set to evaluation mode
            st.session_state["trained_model"] = loaded_model
            st.success(f"Modelo {uploaded_model_file.name} carregado com sucesso.")
            os.remove(tmp_model_path) # Clean up temp file
        except (OSError, RuntimeError) as e:
            st.error(f"Erro ao carregar o modelo: {e}")
        except Exception as e:
            logger.exception("Unexpected error loading model")
            st.error("Erro inesperado ao carregar o modelo")
