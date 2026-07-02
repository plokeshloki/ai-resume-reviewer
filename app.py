"""
Flask web server for the AI Resume Reviewer.

This file:
1. Serves the HTML page when someone visits the site.
2. Provides an /analyze endpoint that accepts a resume PDF + job
   description, runs our analyzer.py logic, and sends back the results
   as JSON (which the webpage's JavaScript will display).
"""

from flask import Flask, request, jsonify, render_template
from analyzer import analyze

app = Flask(__name__)


@app.route("/")
def home():
    # Just shows the upload form (templates/index.html)
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze_resume():
    # 1. Check the resume file was actually uploaded
    if "resume" not in request.files:
        return jsonify({"error": "No resume file uploaded"}), 400

    resume_file = request.files["resume"]
    job_description = request.form.get("job_description", "")

    if resume_file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not job_description.strip():
        return jsonify({"error": "Please paste a job description"}), 400

    # 2. Run our analysis logic from analyzer.py
    result = analyze(resume_file, job_description)

    # 3. Send the result back as JSON
    return jsonify(result)


if __name__ == "__main__":
    # debug=True auto-reloads the server when you save changes - handy while building
    app.run(debug=True, port=5000)
