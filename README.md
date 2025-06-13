# Cirkopn MIDI Generator

This project provides a Streamlit application for creating and editing MIDI files compatible with the Sequentix Cirklon sequencer. It includes tools for instrument management, piano roll editing, automation, advanced validation and a polyphonic AI model for music generation.

## Setup

Install Python 3.9 or newer and the required packages:

```bash
pip install -r requirements.txt
```

## Running

Launch the Streamlit app with:

```bash
streamlit run midigen.py
```

The application will open in your browser at `http://localhost:8501/`.

## Features

* Import and export MIDI, CKC and CKI files
* Piano roll editor with drag-and-drop interface
* Multi-CC automation editor
* Polyphonic transformer model for music generation
* Validation tools to ensure notes fit a musical scale
* Audio preview using a SoundFont file

## Usage

The sidebar of the application guides you through the workflow:

1. **CKIEditor** – manage Cirklon instruments.
2. **Importar MIDI/CKC/CKI** – load existing sequences.
3. **Editar Piano Roll** – draw or edit notes.
4. **Automação Avançada** – create multi‑CC envelopes.
5. **Treinar IA/Gerar Música** – train the polyphonic model and generate ideas.
6. **Manipular MIDI** – apply breakbeat and micro‑timing tools.
7. **Exportar Cirklon/MIDI** – save results in CKC/CKI or standard MIDI.
8. **Preview Audio** – audition using a SoundFont.

## Documentation

The full Cirklon manual can be downloaded from the [Sequentix website](https://sequentix.com).

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
