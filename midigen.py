import os
import tempfile
import numpy as np
import streamlit as st
import torch
from collections import defaultdict

from ui import (
    InstrumentManager,
    UndoRedoManager,
    validate_instrument,
    track_values_panel,
    note_rows_panel,
    ccs_panel,
    pianoroll_dragdrop,
    automation_multicc_advanced,
)
from midi_utils import (
    midi_to_poly_dataset,
    notes_to_midi_file,
    export_ckc,
    export_cki,
    export_p3pattern_json,
    export_auxrows,
    import_ckc_bin,
    import_cki_txt,
    import_auxrows_json,
    breakbeat_poly,
    microtiming_poly,
    lfo_automation,
    midi_preview_sf2,
    validate_advanced,
)
from model import PolyTransformer, train_poly_model, infer_poly_sequence

# Initialize session state for instrument management
if "manager" not in st.session_state:
    st.session_state["manager"] = InstrumentManager()
if "undo_redo" not in st.session_state:
    st.session_state["undo_redo"] = UndoRedoManager(st.session_state["manager"])
manager = st.session_state["manager"]
undo_redo = st.session_state["undo_redo"]

# ---------- MAIN APP ----------

st.title("Sistema IA MIDI Cirklon - Completo e Integrado")

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
    ],
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
            key=f"select_{inst['id']}",
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
        files = st.file_uploader("Arquivos MIDI (múltiplos)", type=["mid", "midi"], accept_multiple_files=True)
        if files:
            all_flattened_notes = []
            for file in files:
                try:
                    flattened_notes = midi_to_poly_dataset(file)
                    if flattened_notes.shape[0] > 0:
                        all_flattened_notes.append(flattened_notes)
                except Exception as e:
                    st.warning(f"Não foi possível processar {file.name}: {e}")
            if all_flattened_notes:
                combined_flattened_notes = np.concatenate(all_flattened_notes, axis=0)
                reconstructed_tracks = defaultdict(list)
                for note_event in combined_flattened_notes:
                    reconstructed_tracks[int(note_event[4])].append(note_event)
                st.session_state["notes"] = [np.array(reconstructed_tracks[k]) for k in sorted(reconstructed_tracks.keys())]
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
                os.remove(tmp_path)
            except Exception as e:
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
                os.remove(tmp_path)
            except Exception as e:
                st.error(f"Erro durante a importação CKI: {e}")

elif tab == "Editar Piano Roll":
    if "notes" in st.session_state and st.session_state["notes"]:
        notes_edit = pianoroll_dragdrop(st.session_state["notes"])
        st.session_state["notes"] = notes_edit
    else:
        st.warning("Por favor, importe um arquivo MIDI/CKC/CKI primeiro para editar o Piano Roll.")

elif tab == "Automação Avançada":
    if "notes" in st.session_state and st.session_state["notes"]:
        cc_list = st.multiselect("CCs para editar", [1, 74, 10, 7, 11], default=[1, 74])
        if st.session_state["notes"] and any(len(track) > 0 for track in st.session_state["notes"]):
            max_abs_tick = 0
            for track_data in st.session_state["notes"]:
                cumulative_tick = 0
                for note_event in track_data:
                    cumulative_tick += note_event[0]
                max_abs_tick = max(max_abs_tick, cumulative_tick)
            length = max_abs_tick
        else:
            length = 512
        cc_events = {cc: lfo_automation(length, cc) for cc in cc_list}
        cc_events = automation_multicc_advanced(cc_events, cc_list, length)
        st.session_state["ccs"] = []
        for cc, events in cc_events.items():
            for tick, val in events:
                st.session_state["ccs"].append([tick, 0, cc, val])
    else:
        st.warning("Por favor, importe um arquivo MIDI/CKC/CKI primeiro para configurar automação.")

elif tab == "Treinar IA":
    if "notes" in st.session_state and st.session_state["notes"]:
        params = {
            "epochs": st.number_input("Epochs", 1, 500, 50),
            "lr": st.number_input("Learning Rate", 0.0001, 0.1, 0.001, format="%f"),
            "model_path": "poly_model.pt",
        }
        if st.button("Treinar Modelo"):
            flat = np.concatenate(st.session_state["notes"], axis=0)
            with st.spinner("Treinando modelo..."):
                model = train_poly_model(flat, params)
            if model:
                st.session_state["trained_model"] = model
                st.success("Treinamento concluído.")
            else:
                st.error("Falha no treinamento do modelo.")
    else:
        st.warning("Por favor, importe um arquivo MIDI/CKC/CKI para treinar o modelo de IA.")

elif tab == "Gerar Música":
    if "trained_model" in st.session_state and st.session_state["trained_model"] is not None:
        st.subheader("Parâmetros de Geração")
        seq_len = st.slider("Duração da Sequência (ticks)", 64, 2048, 512, step=64)
        n_tracks_gen = st.slider("Número de Trilhas a Gerar", 1, 8, 4)
        tempo_gen = st.slider("Andamento (BPM)", 60, 200, 120)
        randomness_temp = st.slider("Grau de Aleatoriedade (Temperatura)", 0.1, 2.0, 1.0, step=0.1)
        st.info("Parâmetros avançados como 'Intensidade de Complexidade', 'Estilo' e 'Instrumentação' serão implementados em futuras versões.")
        if st.button("Gerar Nova Música"):
            with st.spinner('Gerando música...'):
                seq = infer_poly_sequence(st.session_state["trained_model"], seq_len=seq_len, n_tracks=n_tracks_gen, randomness_temp=randomness_temp)
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
        except Exception as e:
            st.error(f"Erro durante a exportação: {e}")
    else:
        st.warning("Por favor, importe ou gere notas primeiro para exportar para o Cirklon.")

elif tab == "Preview Audio":
    if "notes" in st.session_state and st.session_state["notes"]:
        sf2_file = st.file_uploader("SoundFont SF2 para preview", type=["sf2"])
        if sf2_file:
            with tempfile.NamedTemporaryFile(suffix=".sf2", delete=False) as tmp_sf2:
                tmp_sf2.write(sf2_file.read())
                tmp_sf2_path = tmp_sf2.name
            current_tempo = st.session_state.get("tempo", 120)
            midi_preview_sf2(st.session_state["notes"], tmp_sf2_path, tempo=current_tempo)
            os.remove(tmp_sf2_path)
        else:
            st.info("Por favor, carregue um arquivo SoundFont (.sf2) para gerar o preview de áudio.")
    else:
        st.warning("Por favor, importe ou gere notas primeiro para gerar o preview de áudio.")

elif tab == "Validação Musical":
    if "notes" in st.session_state and st.session_state["notes"]:
        notes_abs_tick_for_validation = []
        for track_data in st.session_state["notes"]:
            cumulative_tick = 0
            abs_track_data = []
            for note_event in track_data:
                delta_tick, note, velocity, duration, track_idx = note_event
                cumulative_tick += delta_tick
                abs_track_data.append([cumulative_tick, note, velocity, duration, track_idx])
            notes_abs_tick_for_validation.append(np.array(abs_track_data))
        scale_root = st.slider("Raiz da Escala", 0, 127, 60)
        scale_type = st.selectbox("Tipo de Escala", ["major", "minor", "dorian", "phrygian"])
        max_poly = st.slider("Polifonia Máxima", 1, 16, 8)
        pitch_min = st.slider("Range de Pitch Mínimo", 0, 127, 0)
        pitch_max = st.slider("Range de Pitch Máximo", 0, 127, 127)
        pitch_range = (pitch_min, pitch_max)
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
            n_tracks_for_model = 4
            if st.session_state["notes"] and any(len(track) > 0 for track in st.session_state["notes"]):
                max_track_idx_in_notes = 0
                for track_data in st.session_state["notes"]:
                    if track_data.size > 0:
                        max_track_idx_in_notes = max(max_track_idx_in_notes, int(track_data[:, 4].max()))
                n_tracks_for_model = max_track_idx_in_notes + 1
            loaded_model = PolyTransformer(n_tracks=n_tracks_for_model)
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_model_file:
                tmp_model_file.write(uploaded_model_file.read())
                tmp_model_path = tmp_model_file.name
            loaded_model.load_state_dict(torch.load(tmp_model_path))
            loaded_model.eval()
            st.session_state["trained_model"] = loaded_model
            st.success(f"Modelo {uploaded_model_file.name} carregado com sucesso.")
            os.remove(tmp_model_path)
        except Exception as e:
            st.error(f"Erro ao carregar o modelo: {e}")
