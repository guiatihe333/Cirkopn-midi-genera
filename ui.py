import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from streamlit_drawable_canvas import st_canvas
import copy
from collections import defaultdict

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
            inst["track_values"].append({"slot": slot, "type": "Empty", "label": "", "cc": 0, "track_control": ""})
        for slot in range(1, 9):
            inst["note_rows"].append({"slot": slot, "name": "", "note": "", "velocity": 100})
        for slot in range(1, 9):
            inst["ccs"].append({"slot": slot, "label": "", "cc": 0, "default": 0})
        self.instruments.append(inst)
        self.selected_instrument_id = self.next_id
        self.next_id += 1
        return inst

    def duplicate_instrument(self, instrument_id):
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
        return [i for i in self.instruments if search.lower() in i["name"].lower() or search in str(i["id"])]


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
                self.manager.selected_instrument_id = self.manager.instruments[0]["id"] if self.manager.instruments else None

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(copy.deepcopy(self.manager.instruments))
            self.manager.instruments = self.redo_stack.pop()
            if self.manager.selected_instrument_id not in [i["id"] for i in self.manager.instruments]:
                self.manager.selected_instrument_id = self.manager.instruments[0]["id"] if self.manager.instruments else None


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
                index={"MidiCC": 0, "TrackControl": 1, "Empty": 2}[tv.get("type", "Empty")],
                key=f"type_{tv['slot']}"
            )
            tv["type"] = tv_type
            if tv_type == "MidiCC":
                tv["cc"] = st.number_input("CC #", 0, 127, tv.get("cc", 0), key=f"cc_{tv['slot']}")
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
                    ].index(tv.get("track_control", "pgm")),
                    key=f"tc_{tv['slot']}"
                )

def note_rows_panel(instrument):
    st.subheader("Note Rows")
    for nr in instrument["note_rows"]:
        with st.expander(f"Slot {nr['slot']}"):
            nr["name"] = st.text_input(f"Nome (Slot {nr['slot']})", nr.get("name", ""), key=f"nrname_{nr['slot']}")
            nr["note"] = st.text_input("Nota (ex: C3, D#4)", nr.get("note", ""), key=f"nrnote_{nr['slot']}")
            nr["velocity"] = st.number_input("Velocidade", 1, 127, nr.get("velocity", 100), key=f"nrvelo_{nr['slot']}")

def ccs_panel(instrument):
    st.subheader("CCs")
    for cc in instrument["ccs"]:
        with st.expander(f"Slot {cc['slot']}"):
            cc["label"] = st.text_input(f"Label (Slot {cc['slot']})", cc.get("label", ""), key=f"cclabel_{cc['slot']}")
            cc["cc"] = st.number_input("CC #", 0, 127, cc.get("cc", 0), key=f"ccnum_{cc['slot']}")
            cc["default"] = st.number_input("Valor Padrão", 0, 127, cc.get("default", 0), key=f"ccdef_{cc['slot']}")


def pianoroll_dragdrop(notes, width=1000, height=400, ticks=512, pitch_min=0, pitch_max=127):
    st.subheader("Piano Roll Polifônico Drag & Drop")
    objects = []
    for track, notes_track in enumerate(notes):
        cumulative_tick = 0
        for n in notes_track:
            delta_tick, note, velocity, duration, _ = n
            cumulative_tick += delta_tick
            if duration > 0:
                left = cumulative_tick / ticks * width
                note_width = max(5, duration / ticks * width)
                top = height - ((note - pitch_min) / (pitch_max - pitch_min) * height)
                obj = {
                    "type": "rect",
                    "left": left,
                    "top": top,
                    "width": note_width,
                    "height": 10,
                    "fill": f"rgba({60 + track * 40},150,255,0.7)",
                    "track": track,
                }
                objects.append(obj)
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 255, 0.3)",
        background_color="white",
        update_streamlit=True,
        height=height,
        width=width,
        drawing_mode="rect",
        initial_drawing={"version": "4.4.0", "objects": objects},
        key="pianoroll",
    )
    if canvas_result.json_data and "objects" in canvas_result.json_data:
        drawn_notes_by_track = defaultdict(list)
        for obj in canvas_result.json_data["objects"]:
            x, y, w = obj["left"], obj["top"], obj["width"]
            abs_tick = int(x / width * ticks)
            note = int(pitch_max - (y / height * (pitch_max - pitch_min)))
            velocity = 100
            duration = int(w / width * ticks)
            track_idx = int(obj.get("track", 0))
            drawn_notes_by_track[track_idx].append([abs_tick, note, velocity, duration, track_idx])
        result_notes = []
        for track_idx in sorted(drawn_notes_by_track.keys()):
            track_data = drawn_notes_by_track[track_idx]
            track_data.sort(key=lambda x: x[0])
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


def automation_multicc_advanced(cc_events, cc_list=[1, 74, 10], length=512):
    st.subheader("Editor gráfico multi-CC com curvas simultâneas")
    fig = go.Figure()
    colors = ["red", "green", "blue", "orange", "purple", "cyan", "black"]
    for idx, cc in enumerate(cc_list):
        evs = cc_events.get(cc, [])
        if not evs:
            continue
        ticks = [ev[0] for ev in evs]
        vals = [ev[1] for ev in evs]
        fig.add_trace(go.Scatter(x=ticks, y=vals, mode='lines+markers', name=f"CC{cc}", line=dict(color=colors[idx % len(colors)])))
    fig.update_layout(xaxis=dict(range=[0, length], title="Tick"), yaxis=dict(range=[0, 127], title="Valor CC"), height=400)
    st.plotly_chart(fig, use_container_width=True)
    edited = {}
    for cc in cc_list:
        df = pd.DataFrame(cc_events.get(cc, []), columns=["tick", "value"])
        st.write(f"Edição tabular para CC{cc}")
        new_df = st.data_editor(df, use_container_width=True, num_rows="dynamic")
        edited[cc] = new_df.values.tolist()
    return edited
