"""
run.py
------
Entry point for the Marine Oil Spill Monitoring System.

Usage:
    python run.py
"""

from dotenv import load_dotenv
load_dotenv()  # Load .env before anything else imports os.environ

from backend.app import create_app
from backend import config

app = create_app()

if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
