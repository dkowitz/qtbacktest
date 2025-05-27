# Back-testing Workbench Qt Prototype

A graphical workbench for financial market back-testing with Python.

## Features

- Dataset loading and visualization (charts and tables)
- Python console for interacting with data
- Strategy editor with syntax highlighting
- Strategy execution panel
- Results analysis

## Setup and Installation

1. Create and activate a virtual environment:

```bash
# Windows
python -m venv wqt_bt
wqt_bt\Scripts\activate

# Linux/macOS
python -m venv wqt_bt
source wqt_bt/bin/activate
```

2. Install required packages:

```bash
pip install pandas matplotlib
pip install PySide6
pip install qtconsole ipython
pip install QScintilla
```

3. Run the application:

```bash
# From the project root
python -m qt_prototype
```

## Editor Features

The integrated strategy editor supports:

- Python syntax highlighting
- Auto-indentation
- Line numbers
- File open/save operations
- New strategy templates

## Console Usage

The embedded IPython console provides:

- Access to loaded datasets via the `ds` variable
- Register new datasets with `register(df, name)`
- Add overlays with `register_overlay(dataset_name, name, df)`

## Strategy Development

Strategies should be developed in the `strategies` directory and inherit from the `Strategy` base class.

## Troubleshooting

If you encounter issues with the editor or console:

- Make sure you're running from the virtual environment
- Check that all dependencies are installed
- The editor will fall back to basic editing if QScintilla is not available 