#!/bin/bash
# Consumer Psychology Engine — First-time setup script

echo "🧠 Consumer Psychology Engine Setup"
echo "======================================"

# Check Python version
python_version=$(python3 --version 2>&1)
echo "✅ Python: $python_version"

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate
echo "⚡ Activating venv..."
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip -q

# Install dependencies
echo ""
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Download NLTK data
echo ""
echo "📚 Downloading NLTK data..."
python3 -c "
import nltk
packages = ['stopwords', 'wordnet', 'punkt', 'punkt_tab', 'averaged_perceptron_tagger']
for pkg in packages:
    nltk.download(pkg, quiet=True)
    print(f'  ✅ {pkg}')
"

echo ""
echo "======================================"
echo "✅ Setup complete! Run the app with:"
echo ""
echo "   source venv/bin/activate"
echo "   streamlit run app.py"
echo ""
echo "   Then open: http://localhost:8501"
echo "======================================"
