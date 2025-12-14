#!/usr/bin/env python3
"""
WhisperX GUI - Main Application
A beautiful macOS application for transcribing audio with WhisperX
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for whisperx imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QComboBox, QProgressBar,
    QTextEdit, QGroupBox, QSpinBox, QCheckBox, QTabWidget,
    QFrame, QScrollArea, QSlider, QLineEdit, QMessageBox,
    QSplitter, QStatusBar
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve, QSize
from PyQt6.QtGui import QFont, QIcon, QColor, QPalette, QDragEnterEvent, QDropEvent

import json
import gc
from datetime import datetime


class TranscriptionWorker(QThread):
    """Worker thread for running WhisperX transcription"""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, audio_file: str, settings: dict):
        super().__init__()
        self.audio_file = audio_file
        self.settings = settings
        self._is_cancelled = False
    
    def cancel(self):
        self._is_cancelled = True
    
    def format_duration(self, seconds: float) -> str:
        """Format duration in minutes and seconds"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        if minutes > 0:
            return f"{minutes}m {secs}s"
        return f"{secs}s"
    
    def run(self):
        try:
            import whisperx
            import torch
            import os
            
            device = self.settings.get('device', 'cpu')
            model_name = self.settings.get('model', 'small')
            compute_type = self.settings.get('compute_type', 'int8')
            batch_size = self.settings.get('batch_size', 8)
            language = self.settings.get('language', None)
            do_align = self.settings.get('align', True)
            do_diarize = self.settings.get('diarize', False)
            
            # Get file info
            file_name = os.path.basename(self.audio_file)
            
            self.progress.emit(5, f"🔄 Chargement du modèle '{model_name}'...")
            
            if self._is_cancelled:
                return
            
            # Load model
            model = whisperx.load_model(
                model_name, 
                device, 
                compute_type=compute_type
            )
            
            self.progress.emit(15, f"✅ Modèle chargé • Chargement de l'audio...")
            
            if self._is_cancelled:
                return
            
            # Load audio
            audio = whisperx.load_audio(self.audio_file)
            
            # Calculate audio duration
            audio_duration = len(audio) / 16000  # whisperx uses 16kHz
            duration_str = self.format_duration(audio_duration)
            
            self.progress.emit(20, f"📁 Audio: {duration_str} • Démarrage transcription...")
            
            if self._is_cancelled:
                return
            
            # Transcribe with progress callback if available
            transcribe_options = {"batch_size": batch_size}
            if language:
                transcribe_options["language"] = language
            
            # Estimate progress based on typical transcription speed
            # Whisper processes roughly 30s of audio per batch on CPU
            estimated_batches = max(1, int(audio_duration / 30))
            
            self.progress.emit(25, f"🎙️ Transcription de {duration_str} d'audio...")
            
            result = model.transcribe(audio, **transcribe_options)
            
            # Get detected language
            detected_lang = result.get("language", "??")
            num_segments = len(result.get("segments", []))
            
            self.progress.emit(50, f"✅ Transcription terminée • {num_segments} segments • Langue: {detected_lang}")
            
            if self._is_cancelled:
                return
            
            # Align
            if do_align:
                self.progress.emit(55, f"🔄 Chargement du modèle d'alignement ({detected_lang})...")
                
                model_a, metadata = whisperx.load_align_model(
                    language_code=result["language"], 
                    device=device
                )
                
                self.progress.emit(65, f"⏱️ Alignement des mots en cours...")
                
                result = whisperx.align(
                    result["segments"], 
                    model_a, 
                    metadata, 
                    audio, 
                    device, 
                    return_char_alignments=False
                )
                
                self.progress.emit(75, "✅ Alignement terminé")
                
                # Cleanup alignment model
                del model_a
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                self.progress.emit(75, "⏭️ Alignement ignoré")
            
            if self._is_cancelled:
                return
            
            # Diarize if enabled
            if do_diarize:
                hf_token = self.settings.get('hf_token', None)
                if hf_token:
                    self.progress.emit(80, "🔄 Chargement du modèle de diarisation...")
                    
                    from whisperx.diarize import DiarizationPipeline
                    diarize_model = DiarizationPipeline(
                        use_auth_token=hf_token, 
                        device=device
                    )
                    
                    self.progress.emit(85, "👥 Identification des locuteurs...")
                    
                    diarize_segments = diarize_model(
                        audio,
                        min_speakers=self.settings.get('min_speakers'),
                        max_speakers=self.settings.get('max_speakers')
                    )
                    
                    self.progress.emit(95, "🔗 Attribution des locuteurs aux segments...")
                    
                    result = whisperx.assign_word_speakers(diarize_segments, result)
            else:
                self.progress.emit(95, "⏭️ Diarisation ignorée")
            
            self.progress.emit(98, "🧹 Nettoyage de la mémoire...")
            
            # Cleanup main model
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.progress.emit(100, f"🎉 Terminé ! {num_segments} segments transcrits")
            
            self.finished.emit(result)
            
        except Exception as e:
            self.error.emit(str(e))


class DropZone(QFrame):
    """Custom drop zone for drag and drop files"""
    fileDropped = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setup_ui()
    
    def setup_ui(self):
        self.setMinimumHeight(150)
        self.setStyleSheet("""
            DropZone {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(99, 102, 241, 0.15),
                    stop:1 rgba(139, 92, 246, 0.15));
                border: 2px dashed rgba(129, 140, 248, 0.6);
                border-radius: 16px;
            }
            DropZone:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(99, 102, 241, 0.25),
                    stop:1 rgba(139, 92, 246, 0.25));
                border-color: rgba(165, 180, 252, 0.9);
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Icon label
        icon_label = QLabel("🎤")
        icon_label.setFont(QFont(icon_label.font().family(), 48))
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon_label)
        
        # Text
        text_label = QLabel("Glissez-déposez un fichier audio ici")
        text_label.setFont(QFont(text_label.font().family(), 16, QFont.Weight.Medium))
        text_label.setStyleSheet("color: #c7d2fe;")
        text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(text_label)
        
        # Subtext
        subtext = QLabel("ou cliquez pour parcourir • MP3, WAV, M4A, FLAC")
        subtext.setFont(QFont(subtext.font().family(), 12))
        subtext.setStyleSheet("color: #94a3b8;")
        subtext.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtext)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(self.styleSheet().replace("0.5)", "1.0)"))
    
    def dragLeaveEvent(self, event):
        self.setup_ui()
    
    def dropEvent(self, event: QDropEvent):
        files = [url.toLocalFile() for url in event.mimeData().urls()]
        if files:
            self.fileDropped.emit(files[0])
        self.setup_ui()
    
    def mousePressEvent(self, event):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Sélectionner un fichier audio",
            "",
            "Fichiers audio (*.mp3 *.wav *.m4a *.flac *.ogg *.wma *.aac);;Tous les fichiers (*)"
        )
        if file_path:
            self.fileDropped.emit(file_path)


class WhisperXGUI(QMainWindow):
    """Main WhisperX GUI Application"""
    
    def __init__(self):
        super().__init__()
        self.current_file = None
        self.worker = None
        self.result = None
        self.setup_ui()
        self.apply_dark_theme()
    
    def apply_dark_theme(self):
        """Apply a beautiful dark theme with high contrast"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0a0a12;
            }
            QWidget {
                background-color: #0a0a12;
                color: #f8fafc;
                font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif;
            }
            QSplitter {
                background-color: #0a0a12;
            }
            QSplitter::handle {
                background-color: rgba(129, 140, 248, 0.2);
            }
            QGroupBox {
                background-color: rgba(22, 22, 35, 0.95);
                border: 1px solid rgba(129, 140, 248, 0.4);
                border-radius: 12px;
                margin-top: 16px;
                padding: 16px;
                font-weight: 600;
                color: #e2e8f0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 8px;
                color: #c7d2fe;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #6366f1, stop:1 #8b5cf6);
                color: #ffffff;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-weight: 600;
                font-size: 14px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #818cf8, stop:1 #a78bfa);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4f46e5, stop:1 #7c3aed);
            }
            QPushButton:disabled {
                background: rgba(99, 102, 241, 0.25);
                color: rgba(255, 255, 255, 0.4);
            }
            QPushButton#secondaryBtn {
                background: rgba(99, 102, 241, 0.25);
                border: 1px solid rgba(129, 140, 248, 0.6);
                color: #e0e7ff;
            }
            QPushButton#secondaryBtn:hover {
                background: rgba(99, 102, 241, 0.4);
                color: #ffffff;
            }
            QPushButton#dangerBtn {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ef4444, stop:1 #f87171);
            }
            QPushButton#dangerBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #dc2626, stop:1 #ef4444);
            }
            QComboBox {
                background-color: #1a1a2e;
                border: 1px solid rgba(129, 140, 248, 0.5);
                border-radius: 8px;
                padding: 8px 12px;
                padding-right: 30px;
                min-width: 150px;
                color: #ffffff;
                font-weight: 500;
            }
            QComboBox:hover {
                border-color: rgba(129, 140, 248, 0.9);
                background-color: #1f1f35;
            }
            QComboBox::drop-down {
                border: none;
                width: 24px;
                padding-right: 8px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #a5b4fc;
                margin-right: 8px;
            }
            QComboBox QAbstractItemView {
                background-color: #1a1a2e;
                border: 1px solid rgba(129, 140, 248, 0.6);
                border-radius: 8px;
                padding: 4px;
                outline: none;
            }
            QComboBox QAbstractItemView::item {
                color: #ffffff;
                padding: 8px 12px;
                border-radius: 4px;
                min-height: 24px;
            }
            QComboBox QAbstractItemView::item:hover {
                background-color: rgba(99, 102, 241, 0.4);
            }
            QComboBox QAbstractItemView::item:selected {
                background-color: rgba(99, 102, 241, 0.6);
                color: #ffffff;
            }
            QSpinBox {
                background-color: rgba(22, 22, 35, 0.95);
                border: 1px solid rgba(129, 140, 248, 0.5);
                border-radius: 8px;
                padding: 8px 12px;
                color: #f1f5f9;
            }
            QSpinBox:hover {
                border-color: rgba(129, 140, 248, 0.9);
            }
            QLineEdit {
                background-color: rgba(22, 22, 35, 0.95);
                border: 1px solid rgba(129, 140, 248, 0.5);
                border-radius: 8px;
                padding: 8px 12px;
                color: #f1f5f9;
            }
            QLineEdit:hover, QLineEdit:focus {
                border-color: rgba(129, 140, 248, 0.9);
            }
            QLineEdit::placeholder {
                color: #94a3b8;
            }
            QCheckBox {
                spacing: 8px;
                color: #e2e8f0;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border-radius: 4px;
                border: 2px solid rgba(129, 140, 248, 0.6);
                background-color: rgba(22, 22, 35, 0.8);
            }
            QCheckBox::indicator:checked {
                background-color: #6366f1;
                border-color: #818cf8;
            }
            QCheckBox::indicator:hover {
                border-color: #a5b4fc;
            }
            QProgressBar {
                background-color: rgba(22, 22, 35, 0.95);
                border: none;
                border-radius: 8px;
                height: 14px;
                text-align: center;
                color: #ffffff;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #6366f1, stop:1 #8b5cf6);
                border-radius: 8px;
            }
            QTextEdit {
                background-color: rgba(15, 15, 24, 0.98);
                border: 1px solid rgba(129, 140, 248, 0.4);
                border-radius: 12px;
                padding: 12px;
                font-family: "Menlo", Monaco, "Courier New", monospace;
                font-size: 13px;
                line-height: 1.6;
                color: #e2e8f0;
            }
            QTabWidget::pane {
                border: 1px solid rgba(129, 140, 248, 0.4);
                border-radius: 12px;
                background-color: rgba(22, 22, 35, 0.7);
            }
            QTabBar::tab {
                background-color: rgba(22, 22, 35, 0.7);
                border: 1px solid rgba(129, 140, 248, 0.4);
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding: 10px 20px;
                margin-right: 4px;
                color: #cbd5e1;
            }
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(99, 102, 241, 0.4), stop:1 rgba(22, 22, 35, 0.9));
                color: #ffffff;
                font-weight: 600;
            }
            QTabBar::tab:hover:!selected {
                background-color: rgba(99, 102, 241, 0.3);
                color: #e0e7ff;
            }
            QScrollBar:vertical {
                background-color: rgba(22, 22, 35, 0.5);
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: rgba(129, 140, 248, 0.5);
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: rgba(129, 140, 248, 0.7);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QStatusBar {
                background-color: rgba(22, 22, 35, 0.95);
                border-top: 1px solid rgba(129, 140, 248, 0.3);
                color: #cbd5e1;
                font-weight: 500;
            }
            QLabel#title {
                font-size: 28px;
                font-weight: 700;
                color: #ffffff;
            }
            QLabel#subtitle {
                font-size: 14px;
                color: #cbd5e1;
            }
            QLabel#sectionTitle {
                font-size: 13px;
                font-weight: 600;
                color: #c7d2fe;
                margin-bottom: 4px;
            }
            QLabel {
                color: #e2e8f0;
            }
        """)
    
    def setup_ui(self):
        """Setup the main UI"""
        self.setWindowTitle("WhisperX GUI")
        self.setMinimumSize(1000, 800)
        self.resize(1200, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(20)
        
        # Header
        header = self.create_header()
        main_layout.addWidget(header)
        
        # Main content area
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Settings
        left_panel = self.create_settings_panel()
        content_splitter.addWidget(left_panel)
        
        # Right panel - Results
        right_panel = self.create_results_panel()
        content_splitter.addWidget(right_panel)
        
        content_splitter.setSizes([400, 600])
        main_layout.addWidget(content_splitter, 1)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Prêt • Sélectionnez un fichier audio pour commencer")
    
    def create_header(self) -> QWidget:
        """Create the header section"""
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        # Logo and title
        title_section = QWidget()
        title_layout = QVBoxLayout(title_section)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(4)
        
        title = QLabel("✨ WhisperX GUI")
        title.setObjectName("title")
        title_layout.addWidget(title)
        
        subtitle = QLabel("Transcription rapide et précise avec horodatage au niveau des mots")
        subtitle.setObjectName("subtitle")
        title_layout.addWidget(subtitle)
        
        header_layout.addWidget(title_section)
        header_layout.addStretch()
        
        return header
    
    def create_settings_panel(self) -> QWidget:
        """Create the settings panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 12, 0)
        layout.setSpacing(16)
        
        # Drop zone
        self.drop_zone = DropZone()
        self.drop_zone.fileDropped.connect(self.on_file_selected)
        layout.addWidget(self.drop_zone)
        
        # File info
        self.file_label = QLabel("Aucun fichier sélectionné")
        self.file_label.setStyleSheet("""
            color: #cbd5e1;
            padding: 8px;
            background-color: rgba(22, 22, 35, 0.8);
            border-radius: 8px;
        """)
        self.file_label.setWordWrap(True)
        layout.addWidget(self.file_label)
        
        # Settings tabs
        tabs = QTabWidget()
        
        # Basic Settings Tab
        basic_tab = self.create_basic_settings_tab()
        tabs.addTab(basic_tab, "⚙️ Général")
        
        # Advanced Settings Tab
        advanced_tab = self.create_advanced_settings_tab()
        tabs.addTab(advanced_tab, "🔧 Avancé")
        
        # Diarization Tab
        diarization_tab = self.create_diarization_tab()
        tabs.addTab(diarization_tab, "👥 Locuteurs")
        
        layout.addWidget(tabs)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.transcribe_btn = QPushButton("🎙️ Transcrire")
        self.transcribe_btn.setMinimumHeight(48)
        self.transcribe_btn.clicked.connect(self.start_transcription)
        self.transcribe_btn.setEnabled(False)
        button_layout.addWidget(self.transcribe_btn)
        
        self.cancel_btn = QPushButton("Annuler")
        self.cancel_btn.setObjectName("dangerBtn")
        self.cancel_btn.setMinimumHeight(48)
        self.cancel_btn.clicked.connect(self.cancel_transcription)
        self.cancel_btn.setVisible(False)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
        
        layout.addStretch()
        
        return panel
    
    def create_basic_settings_tab(self) -> QWidget:
        """Create basic settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(16)
        
        # Model selection
        model_label = QLabel("Modèle")
        model_label.setObjectName("sectionTitle")
        layout.addWidget(model_label)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "tiny", "tiny.en", "base", "base.en", "small", "small.en",
            "medium", "medium.en", "large-v1", "large-v2", "large-v3"
        ])
        self.model_combo.setCurrentText("small")
        layout.addWidget(self.model_combo)
        
        # Language
        lang_label = QLabel("Langue")
        lang_label.setObjectName("sectionTitle")
        layout.addWidget(lang_label)
        
        self.language_combo = QComboBox()
        self.language_combo.addItem("Détection auto", None)
        languages = [
            ("Français", "fr"), ("Anglais", "en"), ("Allemand", "de"),
            ("Espagnol", "es"), ("Italien", "it"), ("Japonais", "ja"),
            ("Chinois", "zh"), ("Coréen", "ko"), ("Portugais", "pt"),
            ("Russe", "ru"), ("Néerlandais", "nl"), ("Polonais", "pl"),
        ]
        for name, code in languages:
            self.language_combo.addItem(name, code)
        layout.addWidget(self.language_combo)
        
        # Device
        device_label = QLabel("Processeur")
        device_label.setObjectName("sectionTitle")
        layout.addWidget(device_label)
        
        self.device_combo = QComboBox()
        # Note: MPS (Apple Silicon) is not supported by faster-whisper
        # Only show cpu and cuda options
        self.device_combo.addItems(["cpu", "cuda"])
        # Auto-detect best device
        import torch
        if torch.cuda.is_available():
            self.device_combo.setCurrentText("cuda")
        else:
            self.device_combo.setCurrentText("cpu")
        layout.addWidget(self.device_combo)
        
        # Output format
        format_label = QLabel("Format de sortie")
        format_label.setObjectName("sectionTitle")
        layout.addWidget(format_label)
        
        self.format_combo = QComboBox()
        self.format_combo.addItems(["all", "srt", "vtt", "txt", "tsv", "json"])
        layout.addWidget(self.format_combo)
        
        layout.addStretch()
        
        return tab
    
    def create_advanced_settings_tab(self) -> QWidget:
        """Create advanced settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(16)
        
        # Compute type
        compute_label = QLabel("Type de calcul")
        compute_label.setObjectName("sectionTitle")
        layout.addWidget(compute_label)
        
        self.compute_combo = QComboBox()
        self.compute_combo.addItems(["float16", "float32", "int8"])
        self.compute_combo.setCurrentText("int8")
        layout.addWidget(self.compute_combo)
        
        # Batch size
        batch_label = QLabel("Taille du lot")
        batch_label.setObjectName("sectionTitle")
        layout.addWidget(batch_label)
        
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 32)
        self.batch_spin.setValue(8)
        layout.addWidget(self.batch_spin)
        
        # Align checkbox
        self.align_check = QCheckBox("Activer l'alignement des mots")
        self.align_check.setChecked(True)
        layout.addWidget(self.align_check)
        
        # Highlight words
        self.highlight_check = QCheckBox("Surligner les mots dans les sous-titres")
        self.highlight_check.setChecked(False)
        layout.addWidget(self.highlight_check)
        
        # VAD method
        vad_label = QLabel("Méthode VAD")
        vad_label.setObjectName("sectionTitle")
        layout.addWidget(vad_label)
        
        self.vad_combo = QComboBox()
        self.vad_combo.addItems(["pyannote", "silero"])
        layout.addWidget(self.vad_combo)
        
        layout.addStretch()
        
        return tab
    
    def create_diarization_tab(self) -> QWidget:
        """Create diarization settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(16)
        
        # Enable diarization
        self.diarize_check = QCheckBox("Activer la diarisation (identification des locuteurs)")
        self.diarize_check.setChecked(False)
        self.diarize_check.stateChanged.connect(self.toggle_diarization_settings)
        layout.addWidget(self.diarize_check)
        
        # HuggingFace Token
        token_label = QLabel("Jeton HuggingFace")
        token_label.setObjectName("sectionTitle")
        layout.addWidget(token_label)
        
        self.hf_token_input = QLineEdit()
        self.hf_token_input.setPlaceholderText("hf_xxxxx...")
        self.hf_token_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.hf_token_input.setEnabled(False)
        layout.addWidget(self.hf_token_input)
        
        token_hint = QLabel("Requis pour la diarisation. Obtenez-le sur huggingface.co/settings/tokens")
        token_hint.setStyleSheet("color: #94a3b8; font-size: 11px;")
        token_hint.setWordWrap(True)
        layout.addWidget(token_hint)
        
        # Min speakers
        min_label = QLabel("Nombre minimum de locuteurs")
        min_label.setObjectName("sectionTitle")
        layout.addWidget(min_label)
        
        self.min_speakers_spin = QSpinBox()
        self.min_speakers_spin.setRange(0, 20)
        self.min_speakers_spin.setValue(0)
        self.min_speakers_spin.setSpecialValueText("Auto")
        self.min_speakers_spin.setEnabled(False)
        layout.addWidget(self.min_speakers_spin)
        
        # Max speakers
        max_label = QLabel("Nombre maximum de locuteurs")
        max_label.setObjectName("sectionTitle")
        layout.addWidget(max_label)
        
        self.max_speakers_spin = QSpinBox()
        self.max_speakers_spin.setRange(0, 20)
        self.max_speakers_spin.setValue(0)
        self.max_speakers_spin.setSpecialValueText("Auto")
        self.max_speakers_spin.setEnabled(False)
        layout.addWidget(self.max_speakers_spin)
        
        layout.addStretch()
        
        return tab
    
    def toggle_diarization_settings(self, state):
        """Toggle diarization settings enabled state"""
        enabled = state == Qt.CheckState.Checked.value
        self.hf_token_input.setEnabled(enabled)
        self.min_speakers_spin.setEnabled(enabled)
        self.max_speakers_spin.setEnabled(enabled)
    
    def create_results_panel(self) -> QWidget:
        """Create the results panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 0, 0, 0)
        layout.setSpacing(16)
        
        # Progress section (at the top of results panel)
        self.progress_group = QGroupBox("Progression")
        progress_layout = QVBoxLayout(self.progress_group)
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("En attente...")
        self.progress_label.setStyleSheet("color: #cbd5e1;")
        progress_layout.addWidget(self.progress_label)
        
        self.progress_group.setVisible(False)
        layout.addWidget(self.progress_group)
        
        # Results header
        results_header = QWidget()
        header_layout = QHBoxLayout(results_header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        results_title = QLabel("📝 Résultat de la transcription")
        results_title.setStyleSheet("font-size: 18px; font-weight: 600; color: white;")
        header_layout.addWidget(results_title)
        
        header_layout.addStretch()
        
        # Export buttons
        self.export_txt_btn = QPushButton("📄 TXT")
        self.export_txt_btn.setObjectName("secondaryBtn")
        self.export_txt_btn.clicked.connect(lambda: self.export_result("txt"))
        self.export_txt_btn.setEnabled(False)
        header_layout.addWidget(self.export_txt_btn)
        
        self.export_srt_btn = QPushButton("🎬 SRT")
        self.export_srt_btn.setObjectName("secondaryBtn")
        self.export_srt_btn.clicked.connect(lambda: self.export_result("srt"))
        self.export_srt_btn.setEnabled(False)
        header_layout.addWidget(self.export_srt_btn)
        
        self.export_json_btn = QPushButton("{ } JSON")
        self.export_json_btn.setObjectName("secondaryBtn")
        self.export_json_btn.clicked.connect(lambda: self.export_result("json"))
        self.export_json_btn.setEnabled(False)
        header_layout.addWidget(self.export_json_btn)
        
        layout.addWidget(results_header)
        
        # Results tabs
        self.results_tabs = QTabWidget()
        
        # Text view
        self.text_result = QTextEdit()
        self.text_result.setReadOnly(True)
        self.text_result.setPlaceholderText("La transcription apparaîtra ici...")
        self.results_tabs.addTab(self.text_result, "📝 Texte")
        
        # Segments view
        self.segments_result = QTextEdit()
        self.segments_result.setReadOnly(True)
        self.segments_result.setPlaceholderText("Les segments horodatés apparaîtront ici...")
        self.results_tabs.addTab(self.segments_result, "⏱️ Segments")
        
        # JSON view
        self.json_result = QTextEdit()
        self.json_result.setReadOnly(True)
        self.json_result.setPlaceholderText("La sortie JSON apparaîtra ici...")
        self.results_tabs.addTab(self.json_result, "{ } JSON")
        
        layout.addWidget(self.results_tabs)
        
        return panel
    
    def on_file_selected(self, file_path: str):
        """Handle file selection"""
        self.current_file = file_path
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        
        self.file_label.setText(f"📂 {file_name}\n📊 {file_size:.2f} MB")
        self.file_label.setStyleSheet("""
            color: #e0e7ff;
            padding: 12px;
            background-color: rgba(99, 102, 241, 0.15);
            border: 1px solid rgba(129, 140, 248, 0.5);
            border-radius: 8px;
        """)
        
        self.transcribe_btn.setEnabled(True)
        self.status_bar.showMessage(f"Prêt • {file_name} sélectionné")
    
    def get_settings(self) -> dict:
        """Get current settings from UI"""
        settings = {
            'model': self.model_combo.currentText(),
            'language': self.language_combo.currentData(),
            'device': self.device_combo.currentText(),
            'compute_type': self.compute_combo.currentText(),
            'batch_size': self.batch_spin.value(),
            'align': self.align_check.isChecked(),
            'diarize': self.diarize_check.isChecked(),
            'hf_token': self.hf_token_input.text() if self.diarize_check.isChecked() else None,
            'min_speakers': self.min_speakers_spin.value() if self.min_speakers_spin.value() > 0 else None,
            'max_speakers': self.max_speakers_spin.value() if self.max_speakers_spin.value() > 0 else None,
        }
        return settings
    
    def start_transcription(self):
        """Start the transcription process"""
        if not self.current_file:
            return
        
        # Validate diarization settings
        if self.diarize_check.isChecked() and not self.hf_token_input.text():
            QMessageBox.warning(
                self,
                "Jeton HuggingFace requis",
                "La diarisation nécessite un jeton HuggingFace.\n\n"
                "Veuillez entrer votre jeton ou désactiver la diarisation."
            )
            return
        
        # Update UI
        self.transcribe_btn.setEnabled(False)
        self.transcribe_btn.setVisible(False)
        self.cancel_btn.setVisible(True)
        self.progress_group.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Démarrage...")
        
        # Clear previous results
        self.text_result.clear()
        self.segments_result.clear()
        self.json_result.clear()
        
        # Start worker
        settings = self.get_settings()
        self.worker = TranscriptionWorker(self.current_file, settings)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_transcription_complete)
        self.worker.error.connect(self.on_transcription_error)
        self.worker.start()
        
        self.status_bar.showMessage("Transcription en cours...")
    
    def cancel_transcription(self):
        """Cancel the current transcription"""
        if self.worker:
            self.worker.cancel()
            self.worker.quit()
            self.worker.wait()
        
        self.reset_ui_after_transcription()
        self.status_bar.showMessage("Transcription annulée")
    
    def on_progress(self, value: int, message: str):
        """Update progress"""
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)
        self.status_bar.showMessage(message)
    
    def on_transcription_complete(self, result: dict):
        """Handle transcription completion"""
        self.result = result
        
        # Display text result
        if "segments" in result:
            text = " ".join([seg.get("text", "") for seg in result["segments"]])
            self.text_result.setText(text.strip())
            
            # Display segments
            segments_text = ""
            for seg in result["segments"]:
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                text = seg.get("text", "")
                speaker = seg.get("speaker", "")
                
                if speaker:
                    segments_text += f"[{self.format_time(start)} → {self.format_time(end)}] {speaker}: {text}\n"
                else:
                    segments_text += f"[{self.format_time(start)} → {self.format_time(end)}] {text}\n"
            
            self.segments_result.setText(segments_text)
        
        # Display JSON
        self.json_result.setText(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Enable export buttons
        self.export_txt_btn.setEnabled(True)
        self.export_srt_btn.setEnabled(True)
        self.export_json_btn.setEnabled(True)
        
        self.reset_ui_after_transcription()
        self.status_bar.showMessage("✅ Transcription terminée !")
    
    def on_transcription_error(self, error: str):
        """Handle transcription error"""
        QMessageBox.critical(self, "Erreur", f"La transcription a échoué :\n\n{error}")
        self.reset_ui_after_transcription()
        self.status_bar.showMessage(f"❌ Erreur : {error}")
    
    def reset_ui_after_transcription(self):
        """Reset UI after transcription"""
        self.transcribe_btn.setEnabled(True)
        self.transcribe_btn.setVisible(True)
        self.cancel_btn.setVisible(False)
        self.progress_group.setVisible(False)
        self.progress_label.setText("En attente...")
    
    def format_time(self, seconds: float) -> str:
        """Format seconds to HH:MM:SS,mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def export_result(self, format: str):
        """Export result to file"""
        if not self.result:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            f"Enregistrer le fichier {format.upper()}",
            f"transcription.{format}",
            f"Fichiers {format.upper()} (*.{format})"
        )
        
        if not file_path:
            return
        
        try:
            if format == "txt":
                text = " ".join([seg.get("text", "") for seg in self.result.get("segments", [])])
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(text.strip())
            
            elif format == "srt":
                srt_content = ""
                for i, seg in enumerate(self.result.get("segments", []), 1):
                    start = self.format_time(seg.get("start", 0))
                    end = self.format_time(seg.get("end", 0))
                    text = seg.get("text", "").strip()
                    srt_content += f"{i}\n{start} --> {end}\n{text}\n\n"
                
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(srt_content)
            
            elif format == "json":
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(self.result, f, indent=2, ensure_ascii=False)
            
            self.status_bar.showMessage(f"✅ Exporté vers {file_path}")
            QMessageBox.information(self, "Export terminé", f"Fichier enregistré :\n{file_path}")
        
        except Exception as e:
            QMessageBox.critical(self, "Erreur d'export", f"Échec de l'export :\n{e}")


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("WhisperX GUI")
    app.setOrganizationName("WhisperX")
    
    # Force dark mode with explicit palette
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.ColorRole.Window, QColor(10, 10, 18))
    dark_palette.setColor(QPalette.ColorRole.WindowText, QColor(248, 250, 252))
    dark_palette.setColor(QPalette.ColorRole.Base, QColor(15, 15, 24))
    dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(22, 22, 35))
    dark_palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(22, 22, 35))
    dark_palette.setColor(QPalette.ColorRole.ToolTipText, QColor(248, 250, 252))
    dark_palette.setColor(QPalette.ColorRole.Text, QColor(248, 250, 252))
    dark_palette.setColor(QPalette.ColorRole.Button, QColor(22, 22, 35))
    dark_palette.setColor(QPalette.ColorRole.ButtonText, QColor(248, 250, 252))
    dark_palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ColorRole.Link, QColor(165, 180, 252))
    dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(99, 102, 241))
    dark_palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(148, 163, 184))
    
    app.setPalette(dark_palette)
    app.setStyle("Fusion")  # Use Fusion style for consistent dark mode
    
    window = WhisperXGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
