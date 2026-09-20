"""Routes serving rendered pages."""

from __future__ import annotations

from flask import Blueprint, render_template

blueprint = Blueprint("pages", __name__)


@blueprint.route("/")
def dashboard():
    """Render the single-page benchmark dashboard."""
    return render_template("index.html")
