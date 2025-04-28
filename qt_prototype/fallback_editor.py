"""Simple code editor widget based on QPlainTextEdit with basic syntax highlighting."""

from __future__ import annotations

import os
import re
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QRegularExpression
from PySide6.QtGui import (
    QColor, 
    QFont, 
    QKeySequence,
    QSyntaxHighlighter,
    QTextCharFormat,
    QTextCursor
)
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMenu,
    QMessageBox,
    QVBoxLayout,
    QWidget,
    QToolBar,
    QHBoxLayout,
    QPushButton,
    QStatusBar,
    QLabel,
    QStyle,
    QPlainTextEdit,
)


class PythonHighlighter(QSyntaxHighlighter):
    """Basic Python syntax highlighter."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._formats = {}
        
        # Keyword format
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor("#569CD6"))
        keyword_format.setFontWeight(QFont.Bold)
        keywords = [
            "and", "as", "assert", "break", "class", "continue", "def",
            "del", "elif", "else", "except", "exec", "finally", "for",
            "from", "global", "if", "import", "in", "is", "lambda",
            "not", "or", "pass", "print", "raise", "return", "try",
            "while", "with", "yield", "self", "None", "True", "False"
        ]
        self._formats["keyword"] = keyword_format
        self._keyword_patterns = [r"\b%s\b" % kw for kw in keywords]
        
        # String format
        string_format = QTextCharFormat()
        string_format.setForeground(QColor("#CE9178"))
        self._formats["string"] = string_format
        
        # Comment format
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor("#6A9955"))
        self._formats["comment"] = comment_format
        
        # Function format
        function_format = QTextCharFormat()
        function_format.setForeground(QColor("#DCDCAA"))
        self._formats["function"] = function_format
        
        # Class format
        class_format = QTextCharFormat()
        class_format.setForeground(QColor("#4EC9B0"))
        class_format.setFontWeight(QFont.Bold)
        self._formats["class"] = class_format
        
    def highlightBlock(self, text):
        """Apply syntax highlighting to the given block of text."""
        # Keywords
        for pattern in self._keyword_patterns:
            expression = QRegularExpression(pattern)
            matches = expression.globalMatch(text)
            while matches.hasNext():
                match = matches.next()
                self.setFormat(
                    match.capturedStart(),
                    match.capturedLength(),
                    self._formats["keyword"]
                )
        
        # Comments (starting with #)
        expression = QRegularExpression(r"#[^\n]*")
        matches = expression.globalMatch(text)
        while matches.hasNext():
            match = matches.next()
            self.setFormat(
                match.capturedStart(),
                match.capturedLength(),
                self._formats["comment"]
            )
        
        # Strings (single quotes)
        expression = QRegularExpression(r"'[^'\\]*(\\.[^'\\]*)*'")
        matches = expression.globalMatch(text)
        while matches.hasNext():
            match = matches.next()
            self.setFormat(
                match.capturedStart(),
                match.capturedLength(),
                self._formats["string"]
            )
        
        # Strings (double quotes)
        expression = QRegularExpression(r'"[^"\\]*(\\.[^"\\]*)*"')
        matches = expression.globalMatch(text)
        while matches.hasNext():
            match = matches.next()
            self.setFormat(
                match.capturedStart(),
                match.capturedLength(),
                self._formats["string"]
            )
        
        # Function definitions
        expression = QRegularExpression(r"\bdef\s+(\w+)")
        matches = expression.globalMatch(text)
        while matches.hasNext():
            match = matches.next()
            self.setFormat(
                match.capturedStart(1),
                match.capturedLength(1),
                self._formats["function"]
            )
        
        # Class definitions
        expression = QRegularExpression(r"\bclass\s+(\w+)")
        matches = expression.globalMatch(text)
        while matches.hasNext():
            match = matches.next()
            self.setFormat(
                match.capturedStart(1),
                match.capturedLength(1),
                self._formats["class"]
            )


class PythonEditor(QPlainTextEdit):
    """Enhanced QPlainTextEdit with Python-specific features."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Set a monospaced font
        font = QFont("Consolas" if os.name == "nt" else "Monospace")
        font.setPointSize(10)
        self.setFont(font)
        
        # Set tab width
        metrics = self.fontMetrics()
        self.setTabStopDistance(4 * metrics.horizontalAdvance(' '))
        
        # Install syntax highlighter
        self._highlighter = PythonHighlighter(self.document())
        
    def keyPressEvent(self, event):
        """Handle indentation and auto-pairs."""
        # Auto-indent after colon
        if event.key() == Qt.Key_Return and event.modifiers() == Qt.NoModifier:
            cursor = self.textCursor()
            block = cursor.block()
            text = block.text()
            indent = re.match(r"^(\s*)", text).group(1)
            
            # Check if line ends with colon for additional indentation
            if text.rstrip().endswith(':'):
                indent += "    "
                
            super().keyPressEvent(event)
            self.insertPlainText(indent)
        else:
            super().keyPressEvent(event)


class CodeEditor(QWidget):
    """A code editor widget based on QPlainTextEdit with Python syntax highlighting."""

    saved = Signal(str)  # emitted when a file is saved: path

    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._current_file = None
        self._modified = False
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self._editor = PythonEditor()
        layout.addWidget(self._editor)
        
        # Setup keyboard shortcuts
        from PySide6.QtGui import QShortcut
        
        # Save shortcut (Ctrl+S)
        save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        save_shortcut.activated.connect(self.save_file)
        
        # Open shortcut (Ctrl+O)
        open_shortcut = QShortcut(QKeySequence("Ctrl+O"), self)
        open_shortcut.activated.connect(self.open_file)
        
        # New shortcut (Ctrl+N)
        new_shortcut = QShortcut(QKeySequence("Ctrl+N"), self)
        new_shortcut.activated.connect(self.new_file)
        
        # Context menu setup
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        
        # Track modifications
        self._editor.textChanged.connect(self._on_text_changed)
    
    def _on_text_changed(self):
        """Handle text changes in the editor."""
        if not self._modified:
            self._modified = True
            self._update_window_title()
    
    def _update_window_title(self):
        """Update the parent window title to reflect the current file and modified state."""
        if hasattr(self.parent(), "setWindowTitle"):
            filename = Path(self._current_file).name if self._current_file else "Untitled"
            modified_indicator = "*" if self._modified else ""
            self.parent().setWindowTitle(f"Code Editor - {filename}{modified_indicator}")
    
    def _show_context_menu(self, pos):
        """Show a context menu with editing options."""
        menu = QMenu(self)
        
        # Standard editing actions
        menu.addAction("Cut", self._editor.cut)
        menu.addAction("Copy", self._editor.copy)
        menu.addAction("Paste", self._editor.paste)
        menu.addSeparator()
        
        # File operations
        menu.addAction("New", self.new_file)
        menu.addAction("Open...", self.open_file)
        menu.addAction("Save", self.save_file)
        menu.addAction("Save As...", self.save_file_as)
        
        menu.exec_(self.mapToGlobal(pos))
    
    def text(self):
        """Get the current text content."""
        return self._editor.toPlainText()
    
    def set_text(self, text):
        """Set the editor text content."""
        self._editor.setPlainText(text)
        self._modified = False
        self._update_window_title()
    
    def new_file(self):
        """Create a new empty file."""
        if self._check_save_changes():
            self.set_text("")
            self._current_file = None
            self._modified = False
            self._update_window_title()
    
    def open_file(self, file_path=None):
        """Open a file in the editor."""
        try:
            print("Open file method called")
            
            if not self._check_save_changes():
                print("Save changes check failed")
                return False
            
            if file_path is None:
                # Get strategies directory as default location
                default_dir = self._get_strategies_dir()
                print(f"Opening file dialog with default dir: {default_dir}")
                
                file_path, _ = QFileDialog.getOpenFileName(
                    self, "Open Strategy File", default_dir,
                    "Python Files (*.py);;All Files (*.*)"
                )
                print(f"Selected file: {file_path or 'None'}")
            
            if file_path:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        print(f"Read {len(content)} characters from file")
                        self.set_text(content)
                    self._current_file = file_path
                    self._modified = False
                    self._update_window_title()
                    print("File loaded successfully")
                    return True
                except Exception as e:
                    print(f"Error opening file: {e}")
                    QMessageBox.critical(self, "Error", f"Could not open file: {str(e)}")
            else:
                print("No file selected")
            
            return False
        except Exception as e:
            import traceback
            print(f"Exception in open_file: {e}")
            traceback.print_exc()
            return False
    
    def save_file(self):
        """Save the current file."""
        if self._current_file:
            return self._save_to_file(self._current_file)
        else:
            return self.save_file_as()
    
    def save_file_as(self):
        """Save the current file with a new name."""
        # Use strategies directory as default location
        default_dir = self._get_strategies_dir()
                
        # If we have a current file in the strategies directory, use its directory
        if self._current_file:
            current_path = Path(self._current_file)
            if "strategies" in str(current_path):
                default_dir = str(current_path.parent)
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Strategy File", default_dir,
            "Python Files (*.py);;All Files (*.*)"
        )
        
        if file_path:
            return self._save_to_file(file_path)
        
        return False
    
    def _save_to_file(self, file_path):
        """Save the editor content to a file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.text())
            self._current_file = file_path
            self._modified = False
            self._update_window_title()
            self.saved.emit(file_path)
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save file: {str(e)}")
            return False
    
    def _check_save_changes(self):
        """Check if there are unsaved changes and prompt the user to save them."""
        if self._modified:
            response = QMessageBox.question(
                self, "Unsaved Changes",
                "The document has been modified. Do you want to save your changes?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            
            if response == QMessageBox.Save:
                return self.save_file()
            elif response == QMessageBox.Cancel:
                return False
        
        return True
    
    def close_editor(self):
        """Close the editor after checking for unsaved changes."""
        return self._check_save_changes()

    def _get_strategies_dir(self):
        """Get or create the strategies directory path.
        
        Returns the absolute path to the strategies directory.
        Creates the directory if it doesn't exist.
        """
        import sys
        
        # Potential locations for the strategies directory
        potential_paths = [
            Path("strategies"),                         # Current working directory
            Path("btest/strategies"),                   # Relative to project root
            Path(__file__).parent.parent / "strategies" # Relative to this module
        ]
        
        # Get current module path to check relative locations
        if hasattr(sys.modules.get('__main__'), '__file__'):
            main_dir = Path(sys.modules['__main__'].__file__).parent
            potential_paths.append(main_dir / "strategies")
            potential_paths.append(main_dir.parent / "strategies")
        
        # Check all potential paths
        for path in potential_paths:
            if path.exists() and path.is_dir():
                return str(path.absolute())
        
        # If no strategies directory exists, create one
        try:
            # Prefer creating it in the main module directory
            if hasattr(sys.modules.get('__main__'), '__file__'):
                strat_dir = Path(sys.modules['__main__'].__file__).parent / "strategies"
            else:
                strat_dir = Path("strategies")
                
            strat_dir.mkdir(exist_ok=True, parents=True)
            return str(strat_dir.absolute())
        except Exception:
            # If we can't create the directory, fall back to home
            return str(Path.home())


class EditorDockWidget(QWidget):
    """Dock widget containing the code editor with file management controls."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        
        # Add toolbar with common actions
        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create toolbar buttons - try to use icons if available
        try:
            style = QApplication.style()
            new_btn = QPushButton(style.standardIcon(QStyle.SP_FileIcon), "")
            open_btn = QPushButton(style.standardIcon(QStyle.SP_DialogOpenButton), "")
            save_btn = QPushButton(style.standardIcon(QStyle.SP_DialogSaveButton), "")
        except Exception:
            # Fallback to text buttons
            new_btn = QPushButton("New")
            open_btn = QPushButton("Open")
            save_btn = QPushButton("Save")
        
        new_btn.setToolTip("New File (Ctrl+N)")
        open_btn.setToolTip("Open File (Ctrl+O)")
        save_btn.setToolTip("Save File (Ctrl+S)")
        
        # Add file name label
        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet("color: #888888; font-style: italic;")
        
        toolbar_layout.addWidget(new_btn)
        toolbar_layout.addWidget(open_btn)
        toolbar_layout.addWidget(save_btn)
        toolbar_layout.addWidget(self.file_label, 1)  # Label takes remaining space
        
        layout.addWidget(toolbar)
        
        # Create and add the editor
        self.editor = CodeEditor(self)
        layout.addWidget(self.editor)
        
        # Connect toolbar buttons - use lambda to ensure proper connections
        new_btn.clicked.connect(lambda: self._handle_button_click("new"))
        open_btn.clicked.connect(lambda: self._handle_button_click("open"))
        save_btn.clicked.connect(lambda: self._handle_button_click("save"))
        
        # Status bar (optional)
        self.status_bar = QStatusBar()
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)
        layout.addWidget(self.status_bar)
        
        # Set initial status
        self.editor.saved.connect(self._on_file_saved)
    
    def _handle_button_click(self, action):
        """Handle toolbar button clicks with better error reporting."""
        try:
            if action == "new":
                self.editor.new_file()
                self.status_label.setText("Created new file")
                self.file_label.setText("New file (unsaved)")
                self.file_label.setStyleSheet("color: #888888; font-style: italic;")
            elif action == "open":
                if self.editor.open_file():
                    filename = os.path.basename(self.editor._current_file)
                    self.status_label.setText(f"Opened: {filename}")
                    self.file_label.setText(filename)
                    self.file_label.setStyleSheet("color: #000000; font-style: normal;")
                else:
                    self.status_label.setText("Open file cancelled or failed")
            elif action == "save":
                if self.editor.save_file():
                    filename = os.path.basename(self.editor._current_file)
                    self.status_label.setText(f"Saved: {filename}")
                    self.file_label.setText(filename)
                    self.file_label.setStyleSheet("color: #000000; font-style: normal;")
                else:
                    self.status_label.setText("Save failed or cancelled")
        except Exception as e:
            import traceback
            self.status_label.setText(f"Error: {str(e)}")
            traceback.print_exc()
    
    def _on_file_saved(self, path):
        """Update status bar when a file is saved."""
        filename = os.path.basename(path)
        self.status_label.setText(f"Saved: {filename}")
        self.file_label.setText(filename)
        self.file_label.setStyleSheet("color: #000000; font-style: normal;")
        
        # Try to update window title if we're in a dock widget
        dock = self.parent()
        if hasattr(dock, "setWindowTitle"):
            dock.setWindowTitle(f"Editor - {filename}")
            
    def set_status(self, message):
        """Set the status bar message."""
        self.status_label.setText(message) 