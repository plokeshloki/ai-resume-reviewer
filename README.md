# AI Resume Reviewer

Upload a resume (PDF) and paste a job description — get a match score
and a list of important keywords from the job description that are
missing from your resume.

## How it works
1. `analyzer.py` extracts text from the uploaded PDF using `pdfplumber`.
2. It compares the resume text against the job description using
   **TF-IDF + cosine similarity** (scikit-learn) — this is the "AI"
   part. TF-IDF weighs rare/meaningful words more heavily than common
   ones, and cosine similarity measures how close the two texts are.
3. It also finds keywords present in the job description but missing
   from the resume, filtering out generic filler words.
4. `app.py` is a small Flask server exposing:
   - `GET /` — the upload page
   - `POST /analyze` — accepts the resume file + job description,
     returns JSON with the score and missing keywords
5. `templates/index.html` is the frontend — plain HTML/CSS/JS,
   no framework needed.

## Running it locally
```
pip install flask pdfplumber scikit-learn
python3 app.py
```
Then open http://127.0.0.1:5000 in your browser.

## Possible upgrades (for later)
- Replace TF-IDF with sentence embeddings (`sentence-transformers`)
  for smarter, meaning-based matching instead of keyword-based.
- Support .docx resumes, not just PDF.
- Highlight WHERE in the resume matching skills were found.
- Add a "suggested bullet point" generator using an LLM API.
