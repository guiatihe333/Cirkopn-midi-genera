import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from streamlit_drawable_canvas import st_canvas
import copy
from midi_utils import midi_to_poly_dataset, notes_to_midi_file, PolyTransformer, train_poly_model, infer_poly_sequence, breakbeat_poly, microtiming_poly, lfo_automation, envelope_automation, randomize_automation, export_ckc, export_cki, export_p3pattern_json, export_auxrows, import_ckc_bin, import_cki_txt, import_auxrows_json, validate_advanced, midi_preview_sf2

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
 
# Utility imports for file handling and model loading
import torch
import tempfile
from collections import defaultdict
import os

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
                except IOError as e:
                    st.warning(f"Não foi possível processar {file.name}: {e}")
            
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
            except IOError as e:
                st.error(f"Erro durante a importação CKC: {e}")
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
            except IOError as e:
                st.error(f"Erro durante a importação CKI: {e}")
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
        
        try:
            trained_model = train_poly_model(all_notes_flat_for_training, params)
        except ValueError as e:
            st.error(str(e))
            trained_model = None
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
                try:
                    seq = infer_poly_sequence(
                        st.session_state["trained_model"],
                        seq_len=seq_len,
                        n_tracks=n_tracks_gen,
                        randomness_temp=randomness_temp,
                    )
                except ValueError as e:
                    st.error(str(e))
                    seq = []
                if seq:
                    st.session_state["notes"] = seq
                    st.session_state["tempo"] = tempo_gen
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
        except Exception as e:
            st.error(f"Erro durante a manipulação MIDI: {e}")
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
        except IOError as e:
            st.error(f"Erro durante a exportação: {e}")
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
            try:
                audio_path = midi_preview_sf2(
                    st.session_state["notes"], tmp_sf2_path, tempo=current_tempo
                )
                with open(audio_path, "rb") as f:
                    st.audio(f, format="audio/wav")
            except IOError as e:
                st.error(str(e))
            finally:
                os.remove(tmp_sf2_path)  # Clean up temp file
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
        except Exception as e:
            st.error(f"Erro durante a validação musical: {e}")
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
            except Exception as e:
                st.error(f"Erro ao salvar o modelo: {e}")
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
        except Exception as e:
            st.error(f"Erro ao carregar o modelo: {e}")
