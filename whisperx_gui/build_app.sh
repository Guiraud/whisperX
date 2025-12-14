#!/bin/bash
# Build WhisperX GUI as a macOS .app bundle
# Requires: pip install py2app

set -e

echo "🎤 Building WhisperX GUI macOS App..."

cd "$(dirname "$0")/.."

# Activate virtual environment if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Install py2app if not present
python -c "import py2app" 2>/dev/null || pip install py2app

# Create setup.py for py2app
cat > whisperx_gui/setup_app.py << 'EOF'
from setuptools import setup

APP = ['run.py']
DATA_FILES = []
OPTIONS = {
    'argv_emulation': True,
    'iconfile': None,  # Add icon later
    'plist': {
        'CFBundleName': 'WhisperX GUI',
        'CFBundleDisplayName': 'WhisperX GUI',
        'CFBundleGetInfoString': 'Transcribe audio with WhisperX',
        'CFBundleIdentifier': 'com.whisperx.gui',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSHighResolutionCapable': True,
        'NSRequiresAquaSystemAppearance': False,  # Support dark mode
        'CFBundleDocumentTypes': [
            {
                'CFBundleTypeName': 'Audio File',
                'CFBundleTypeRole': 'Viewer',
                'LSItemContentTypes': [
                    'public.audio',
                    'public.mp3',
                    'public.mpeg-4-audio',
                    'com.apple.m4a-audio',
                ],
            }
        ],
    },
    'packages': ['whisperx', 'torch', 'torchaudio', 'PyQt6'],
    'includes': [
        'whisperx',
        'torch',
        'torchaudio', 
        'transformers',
        'pyannote',
        'faster_whisper',
    ],
}

setup(
    app=APP,
    name='WhisperX GUI',
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
EOF

# Build the app
echo "📦 Building app bundle..."
cd whisperx_gui
python setup_app.py py2app 2>/dev/null || python setup_app.py py2app -A

echo ""
echo "✅ Build complete!"
echo "📂 App location: whisperx_gui/dist/WhisperX GUI.app"
echo ""
echo "To run: open 'whisperx_gui/dist/WhisperX GUI.app'"
