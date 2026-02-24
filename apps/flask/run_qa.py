#!/usr/bin/env python3
"""
QA Runner for Flask App
Adds health endpoint and runs on port 8001
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from apps.flask.web_interface import app

@app.route("/health")
def health():
    """Health check endpoint for Cloudflare/monitoring"""
    return {
        "status": "healthy",
        "service": "quantagent-qa",
        "version": "0.1.0"
    }, 200

if __name__ == "__main__":
    # Run on all interfaces, port 8001
    app.run(
        debug=False,
        host="0.0.0.0",
        port=8001
    )
