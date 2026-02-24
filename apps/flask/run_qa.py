#!/usr/bin/env python3
"""
QA Runner for Flask App
Adds health endpoint and runs on port 8001
"""
import sys
from pathlib import Path
from flask import Flask, jsonify

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Create minimal Flask app for health check first
app = Flask(__name__)

@app.route("/health")
def health():
    """Health check endpoint for Cloudflare/monitoring"""
    return jsonify({
        "status": "healthy",
        "service": "quantagent-qa",
        "version": "0.1.0"
    }), 200

# NOW import the main web_interface (which needs API keys)
# This way /health works even if web_interface fails to initialize
try:
    from apps.flask import web_interface
    # Register web_interface routes
    for rule in web_interface.app.url_map.iter_rules():
        if rule.endpoint != 'static' and rule.endpoint != 'health':
            view_func = web_interface.app.view_functions[rule.endpoint]
            app.add_url_rule(
                rule.rule,
                endpoint=rule.endpoint,
                view_func=view_func,
                methods=rule.methods
            )
    print("✓ Web interface loaded successfully")
except Exception as e:
    print(f"⚠ Warning: Web interface failed to load: {e}")
    print("  Health endpoint still available")

if __name__ == "__main__":
    # Run on all interfaces, port 8001
    app.run(
        debug=False,
        host="0.0.0.0",
        port=8001
    )
