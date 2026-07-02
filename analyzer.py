"""
Core logic for the AI Resume Reviewer.

What this file does:
1. Extracts text from an uploaded PDF resume.
2. Compares that text against a job description.
3. Returns a match score + which important words from the
   job description are MISSING from the resume.
"""

import re
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# A small list of common words we don't care about when comparing
# (the, and, is, etc.) - these don't tell us anything useful.
STOPWORDS = set("""
a an the and or but is are was were be been being to of in on for with
as by at from this that these those it its i you he she we they
will would should could can may might must shall
looking candidate candidates plus etc such also very
strong good great excellent ability skills experience years
work working team teams role position job company
familiar familiarity knowledge understanding
including include includes required requirements requires
responsible responsibilities responsible
using use used able
""".split())


def extract_text_from_pdf(file_stream):
    """
    Takes an uploaded PDF file and returns all the text inside it as a string.
    'file_stream' is the raw file object Flask gives us from the upload.
    """
    text = ""
    with pdfplumber.open(file_stream) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def clean_and_tokenize(text):
    """
    Lowercases the text, strips punctuation, and splits it into
    individual words, removing stopwords. Returns a set of words.
    """
    text = text.lower()
    words = re.findall(r"[a-zA-Z\+\#]+", text)  # keeps things like c++ , c#
    return set(w for w in words if w not in STOPWORDS and len(w) > 1)


def get_match_score(resume_text, job_description):
    """
    Uses TF-IDF + cosine similarity to score how well the resume
    matches the job description. Returns a percentage (0-100).

    This is the "AI" part - instead of just checking if words match
    exactly, TF-IDF weighs words by importance (rare, meaningful words
    matter more than common ones), and cosine similarity measures how
    close the two documents are in meaning-space.
    """
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    score = similarity[0][0] * 100
    return round(score, 1)


def get_missing_keywords(resume_text, job_description, top_n=15):
    """
    Finds important words that appear in the job description
    but are missing from the resume. This gives the user
    actionable feedback, not just a score.
    """
    resume_words = clean_and_tokenize(resume_text)
    jd_words = clean_and_tokenize(job_description)

    missing = jd_words - resume_words

    # Sort by length as a rough proxy for "meaningful" words
    # (short words like 'ok' are less useful than 'kubernetes')
    missing_sorted = sorted(missing, key=len, reverse=True)
    return missing_sorted[:top_n]


def analyze(resume_file_stream, job_description):
    """
    Main function that ties everything together.
    Returns a dictionary with the score and missing keywords.
    """
    resume_text = extract_text_from_pdf(resume_file_stream)

    if not resume_text.strip():
        return {
            "error": "Could not extract any text from this PDF. "
                     "Try a different file (it may be a scanned image)."
        }

    score = get_match_score(resume_text, job_description)
    missing_keywords = get_missing_keywords(resume_text, job_description)

    return {
        "score": score,
        "missing_keywords": missing_keywords,
        "resume_word_count": len(resume_text.split()),
    }
