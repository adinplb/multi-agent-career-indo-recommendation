"""
CV Parser Tool
Extracts text from PDF files and optionally fetches GitHub profile data.
"""
import os
import requests


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Uses pymupdf (fitz) to extract text from a PDF byte stream.
    Sorts text blocks by vertical position per page for correct reading order.
    Returns plain text string, truncated to 20,000 characters.
    """
    try:
        import fitz  # pymupdf
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages_text = []
        for page in doc:
            blocks = page.get_text("blocks")
            # Sort blocks top-to-bottom, left-to-right
            blocks.sort(key=lambda b: (round(b[1] / 20), b[0]))
            page_text = "\n".join(b[4].strip() for b in blocks if b[4].strip())
            pages_text.append(page_text)
        doc.close()
        full_text = "\n\n".join(pages_text)
        return full_text[:20000]
    except ImportError:
        raise RuntimeError("pymupdf not installed. Run: pip install pymupdf")
    except Exception as e:
        raise RuntimeError(f"Failed to parse PDF: {e}")


def extract_github_profile(github_url: str) -> dict:
    """
    Fetches public GitHub profile data via the GitHub REST API.
    Extracts: username, bio, top repos, languages used.
    Returns empty dict on failure (unauthenticated, rate-limited).
    """
    if not github_url:
        return {}

    try:
        # Extract username from URL
        # Handles: https://github.com/username or https://github.com/username/
        parts = github_url.rstrip("/").split("/")
        username = parts[-1]
        if not username:
            return {}

        headers = {"Accept": "application/vnd.github.v3+json"}
        base = "https://api.github.com"

        # Fetch user profile
        user_resp = requests.get(f"{base}/users/{username}", headers=headers, timeout=5)
        if user_resp.status_code != 200:
            return {}
        user_data = user_resp.json()

        # Fetch top public repos (sorted by stars)
        repos_resp = requests.get(
            f"{base}/users/{username}/repos",
            headers=headers,
            params={"sort": "stars", "per_page": 10},
            timeout=5,
        )
        repos = repos_resp.json() if repos_resp.status_code == 200 else []

        # Collect languages and repo descriptions
        languages = set()
        repo_summaries = []
        for repo in repos:
            if repo.get("language"):
                languages.add(repo["language"])
            if repo.get("description"):
                repo_summaries.append(f"{repo['name']}: {repo['description']}")

        return {
            "username": username,
            "bio": user_data.get("bio", ""),
            "public_repos": user_data.get("public_repos", 0),
            "languages": sorted(languages),
            "top_repos": repo_summaries[:5],
        }
    except Exception:
        return {}
