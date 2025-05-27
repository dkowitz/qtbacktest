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
    QTabWidget,
    QTabBar
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
            if not self._check_save_changes():
                return False
            
            if file_path is None:
                # Get strategies directory as default location
                default_dir = self._get_strategies_dir()
                
                file_path, _ = QFileDialog.getOpenFileName(
                    self, "Open Strategy File", default_dir,
                    "Python Files (*.py);;All Files (*.*)"
                )
            
            if file_path:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        self.set_text(content)
                    self._current_file = file_path
                    self._modified = False
                    self._update_window_title()
                    return True
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Could not open file: {str(e)}")
            else:
                pass
            
            return False
        except Exception as e:
            import traceback
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


class TabbedEditorWidget(QTabWidget):
    """A tabbed widget that contains multiple code editors."""
    
    saved = Signal(str)  # emitted when a file is saved: path
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setTabsClosable(True)
        self.setMovable(True)
        self.setDocumentMode(True)
        
        # Connect signals
        self.tabCloseRequested.connect(self._close_tab)
        
        # Create the "+" tab for adding new files
        self.addTab(QWidget(), "+")
        self.tabBar().setTabButton(0, QTabBar.RightSide, None)  # Hide the close button on the "+" tab
        
        self.currentChanged.connect(self._handle_tab_change)
        
    def _handle_tab_change(self, index):
        """Handle tab change events, including the special "+" tab."""
        if index == 0:  # The "+" tab
            # Create a new untitled file
            self.new_file()
            # Switch back to the newly created tab
            self.setCurrentIndex(1)  # The new tab will be at index 1
    
    def current_editor(self):
        """Return the currently active editor."""
        if self.count() <= 1:  # Only the "+" tab
            return None
            
        current_widget = self.currentWidget()
        if isinstance(current_widget, CodeEditor):
            return current_widget
        return None
    
    def new_file(self):
        """Create a new untitled file tab."""
        editor = CodeEditor(self)
        editor.saved.connect(self._on_file_saved)
        
        index = self.addTab(editor, "Untitled")
        self.setCurrentIndex(index)
        return editor
    
    def open_file(self, file_path=None):
        """Open a file in a new tab or prompt for a file to open."""
        # If no file_path is provided, show a file dialog
        if file_path is None:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open File", self._get_strategies_dir(),
                "Python Files (*.py);;All Files (*.*)"
            )
            
        if not file_path:
            return None
            
        # Check if this file is already open
        for i in range(1, self.count()):  # Skip the "+" tab
            editor = self.widget(i)
            if isinstance(editor, CodeEditor) and editor._current_file == file_path:
                self.setCurrentIndex(i)
                return editor
        
        # Create a new tab with this file
        editor = CodeEditor(self)
        editor.saved.connect(self._on_file_saved)
        
        # Open the file in the editor
        with open(file_path, 'r', encoding='utf-8') as f:
            editor.set_text(f.read())
            
        editor._current_file = file_path
        editor._modified = False
        
        # Add to tabs with filename as the tab title
        filename = Path(file_path).name
        index = self.addTab(editor, filename)
        self.setCurrentIndex(index)
        
        return editor
    
    def _close_tab(self, index):
        """Close the tab at the given index."""
        if index == 0:  # Don't close the "+" tab
            return
            
        editor = self.widget(index)
        if isinstance(editor, CodeEditor):
            if editor._modified:
                # Ask to save changes
                reply = QMessageBox.question(
                    self, "Save Changes",
                    f"Save changes to {Path(editor._current_file).name if editor._current_file else 'Untitled'}?",
                    QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
                )
                
                if reply == QMessageBox.Save:
                    saved = editor.save_file()
                    if not saved:
                        return  # Abort closing if save was cancelled
                elif reply == QMessageBox.Cancel:
                    return  # Abort closing
            
        self.removeTab(index)
    
    def _on_file_saved(self, path):
        """Update tab title when a file is saved."""
        for i in range(1, self.count()):
            editor = self.widget(i)
            if isinstance(editor, CodeEditor) and editor._current_file == path:
                self.setTabText(i, Path(path).name)
                break
                
        # Forward the signal
        self.saved.emit(path)
    
    def _get_strategies_dir(self):
        """Find the likely strategies directory."""
        # Look for strategies directory relative to the current file
        possible_paths = [
            Path("strategies"),
            Path("btest/strategies"),
            Path(__file__).parent.parent / "strategies",
        ]
        
        # Try to find current working directory
        try:
            import os
            current_dir = os.getcwd()
            possible_paths.insert(0, Path(current_dir) / "strategies")
        except Exception:
            pass
            
        for path in possible_paths:
            if path.exists() and path.is_dir():
                return str(path)
                
        # Fallback to current directory or home
        try:
            return os.getcwd()
        except Exception:
            return str(Path.home())


class EditorDockWidget(QWidget):
    """A widget with a code editor, toolbar, and status bar."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Toolbar
        toolbar = QToolBar()
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        
        # Add actions
        new_action = toolbar.addAction("New")
        new_action.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))
        new_action.triggered.connect(lambda: self._handle_button_click("new"))
        
        open_action = toolbar.addAction("Open")
        open_action.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        open_action.triggered.connect(lambda: self._handle_button_click("open"))
        
        save_action = toolbar.addAction("Save")
        save_action.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        save_action.triggered.connect(lambda: self._handle_button_click("save"))
        
        toolbar.addSeparator()
        
        run_action = toolbar.addAction("Run")
        run_action.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        run_action.triggered.connect(lambda: self._handle_button_click("run"))
        
        layout.addWidget(toolbar)
        
        # Tabbed editor
        self.editor_tabs = TabbedEditorWidget(self)
        self.editor_tabs.saved.connect(self._on_file_saved)
        layout.addWidget(self.editor_tabs)
        
        # Make sure we have at least one editor tab (after the "+" tab)
        self.editor_tabs.new_file()
        
        # Status bar
        self.status_bar = QStatusBar()
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)
        layout.addWidget(self.status_bar)
        
        self.setLayout(layout)
    
    @property
    def editor(self):
        """Return the currently active editor for backward compatibility."""
        return self.editor_tabs.current_editor()
    
    def _handle_button_click(self, action):
        """Handle toolbar button clicks."""
        if action == "new":
            self.editor_tabs.new_file()
        elif action == "open":
            self.editor_tabs.open_file()
        elif action == "save":
            editor = self.editor_tabs.current_editor()
            if editor:
                editor.save_file()
        elif action == "run":
            editor = self.editor_tabs.current_editor()
            if editor and editor._current_file:
                self.set_status(f"Running {editor._current_file}...")
                # TODO: Implement running the strategy
                # For now, just show a message
                QMessageBox.information(
                    self,
                    "Run Strategy",
                    f"Would run strategy: {editor._current_file}"
                )
            else:
                QMessageBox.warning(
                    self,
                    "Cannot Run",
                    "Please save the file first before running."
                )
    
    def _on_file_saved(self, path):
        """Handle file saved event from any editor tab."""
        self.set_status(f"Saved {path}")
    
    def set_status(self, message):
        """Set status bar message."""
        self.status_label.setText(message) 