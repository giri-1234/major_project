"""
app.py
------
Flask application factory. Keeps ``run.py`` (the entry point) trivial and
makes the app testable (multiple app instances with different configs).
"""

from __future__ import annotations

from flask import Flask, jsonify

from backend import config
from backend.routes import bp as main_blueprint
from backend.utils import get_logger

logger = get_logger(__name__)


def create_app() -> Flask:
    """Build and configure the Flask application."""
    app = Flask(
        __name__,
        static_folder=config.STATIC_DIR,
        static_url_path="/static",
        template_folder=config.TEMPLATES_DIR,
    )

    app.config["SECRET_KEY"] = config.SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    app.config["UPLOAD_FOLDER"] = config.UPLOAD_FOLDER

    app.register_blueprint(main_blueprint)

    # Ensure API routes always return JSON errors, not HTML pages
    @app.errorhandler(404)
    def not_found(e):
        from flask import request
        if request.path.startswith("/api/") or request.path.startswith("/satellite/"):
            return jsonify({"status": "error", "message": str(e)}), 404
        from flask import render_template
        return render_template("index.html", error=str(e)), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({"status": "error", "message": str(e)}), 405

    logger.info("Flask application created (debug=%s).", config.DEBUG)
    return app
