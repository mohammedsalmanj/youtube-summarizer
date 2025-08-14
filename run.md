# 1. Clone the repository
git clone https://github.com/mohammedsalmanj/youtube-summarizer.git
cd youtube-summarizer

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# 4. Upgrade pip (optional, recommended)
python -m pip install --upgrade pip

# 5. Install dependencies
pip install -r requirements.txt

# 6. Run the application
python app.py
