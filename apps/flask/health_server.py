#!/usr/bin/env python3
"""
Minimal health check server for QA environment
Runs independently of web_interface to avoid API key requirements
"""
from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/")
def index():
    """Root endpoint"""
    return jsonify({
        "service": "QuantAgent QA",
        "status": "running",
        "endpoints": {
            "/health": "Health check",
            "/": "Service info"
        }
    }), 200

@app.route("/health")
def health():
    """Health check endpoint for Cloudflare/monitoring"""
    return jsonify({
        "status": "healthy",
        "service": "quantagent-qa",
        "version": "0.1.0",
        "environment": "qa"
    }), 200

if __name__ == "__main__":
    # Run on all interfaces, port 8001
    print("Starting QuantAgent QA Health Server on 0.0.0.0:8001")
    app.run(
        debug=False,
        host="0.0.0.0",
        port=8001
    )
