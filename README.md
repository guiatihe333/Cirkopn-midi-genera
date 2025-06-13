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

## Running Tests

Install the dependencies and execute the test suite with:

```bash
pip install -r requirements.txt
pytest
```

If you encounter import errors or missing packages, rerun the install command above to ensure all requirements are present.

## Features

* Import and export MIDI, CKC and CKI files
* Piano roll editor with drag-and-drop interface
* Multi-CC automation editor
* Polyphonic transformer model for music generation
* Validation tools to ensure notes fit a musical scale
* Audio preview using a SoundFont file

## License

This project is provided as-is for demonstration purposes.
