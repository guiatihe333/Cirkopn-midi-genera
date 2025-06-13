import streamlit as st

def track_values_panel(instrument):
    st.subheader("Track Values")
    for tv in instrument["track_values"]:
        with st.container():
            st.markdown(f"**Slot {tv['slot']}**")
            tv["label"] = st.text_input(
                f"Label (Slot {tv['slot']})",
                tv.get("label", ""),
                key=f"label_{tv['slot']}"
            )
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
                    ["pgm", "quant%", "note%", "noteC", "velo%", "veloC", "leng%", "tbase", "xpos", "octave", "knob1", "knob2"],
                    index=["pgm", "quant%", "note%", "noteC", "velo%", "veloC", "leng%", "tbase", "xpos", "octave", "knob1", "knob2"].index(tv.get("track_control", "pgm")),
                    key=f"tc_{tv['slot']}"
                )


def note_rows_panel(instrument):
    st.subheader("Note Rows")
    for nr in instrument["note_rows"]:
        with st.container():
            st.markdown(f"**Slot {nr['slot']}**")
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
        with st.container():
            st.markdown(f"**Slot {cc['slot']}**")
            cc["label"] = st.text_input(
                f"Label (Slot {cc['slot']})", cc.get("label", ""), key=f"cclabel_{cc['slot']}"
            )
            cc["cc"] = st.number_input(
                "CC #", 0, 127, cc.get("cc", 0), key=f"ccnum_{cc['slot']}"
            )
            cc["default"] = st.number_input(
                "Valor Padrão", 0, 127, cc.get("default", 0), key=f"ccdef_{cc['slot']}"
            )
