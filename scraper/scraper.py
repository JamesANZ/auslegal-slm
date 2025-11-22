#!/usr/bin/env python3
"""
AustLII Legal Document Scraper
Scrapes legal documents from AustLII databases using browser automation and saves them as text files.
"""

import os
import re
import time
import json
from pathlib import Path
from urllib.parse import urljoin, urlparse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# Configuration
BASE_URL = "https://www.austlii.edu.au"
DATABASES_URL = "https://www.austlii.edu.au/databases.html"
# Get project root directory (parent of scraper directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROGRESS_FILE = os.path.join(DATA_DIR, ".scraper_progress.json")
MAX_DOCUMENTS_PER_DATABASE = 1000  # Increased limit
MAX_DATABASES = None  # None = no limit, process all
DELAY_BETWEEN_REQUESTS = 0.5  # Seconds to wait between requests
PAGE_LOAD_TIMEOUT = 30  # Seconds to wait for page load


def sanitize_filename(filename):
    """Sanitize a string to be used as a filename."""
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
    # Remove leading/trailing spaces and dots
    filename = filename.strip(". ")
    # Limit length
    if len(filename) > 200:
        filename = filename[:200]
    return filename


def setup_driver():
    """Set up and return a Selenium WebDriver instance."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in background
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
    return driver


def extract_legal_text(html_content):
    """
    Extract legal text content from a document page.
    Removes navigation, headers, footers, and other non-legal content.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove script and style elements
    for element in soup(["script", "style", "nav", "header", "footer"]):
        element.decompose()

    # Try to find the main content area
    # AustLII document pages typically have content in specific areas
    content_selectors = [
        "div#content",
        "div.content",
        "div.main",
        "div#main",
        "div.document",
        "div#document",
        "pre",  # Many AustLII documents are in <pre> tags
        "body",
    ]

    content = None
    for selector in content_selectors:
        content = soup.select_one(selector)
        if content:
            break

    if not content:
        content = soup.find("body")

    if not content:
        return ""

    # Get text and clean it up
    text = content.get_text(separator="\n", strip=True)

    # Remove excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    # Remove common non-legal text patterns
    lines = text.split("\n")
    filtered_lines = []
    skip_patterns = [
        r"^AustLII",
        r"^Search",
        r"^About",
        r"^Contact",
        r"^Copyright",
        r"^Privacy",
        r"^Disclaimer",
        r"^Feedback",
        r"^Help",
        r"^Database",
        r"^Jurisdiction",
        r"^Type",
        r"^All Databases",
        r"^Home",
        r"^Back to",
        r"^Return to",
        r"^Last updated",
        r"^URL:",
        r"^AustLII:",
        r"^Database:",
        r"^Jurisdiction:",
        r"^Type:",
    ]

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip lines that match skip patterns
        should_skip = False
        for pattern in skip_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                should_skip = True
                break
        if not should_skip and len(line) > 10:  # Minimum meaningful length
            filtered_lines.append(line)

    return "\n".join(filtered_lines)


def get_database_links(driver):
    """Get all database links from the databases page."""
    print(f"Fetching database links from {DATABASES_URL}...")
    try:
        driver.get(DATABASES_URL)
        time.sleep(2)  # Wait for page to load

        # Get page source and parse with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, "html.parser")

        database_links = set()

        # Look for links that point to databases
        # Based on the page structure, database links are typically in lists
        # and point to /cgi-bin/viewdb/ or /au/ paths
        for link in soup.find_all("a", href=True):
            href = link.get("href", "")
            full_url = urljoin(BASE_URL, href)

            # Filter for database links
            # AustLII database links typically contain:
            # - /cgi-bin/viewdb/au/cases/
            # - /cgi-bin/viewdb/au/legis/
            # - /au/cases/ (directory listing pages)
            # - /au/legis/ (directory listing pages)
            # - /au/journals/
            # - /au/other/
            if BASE_URL in full_url and (
                "/cgi-bin/viewdb/" in full_url
                or (
                    "/au/" in full_url
                    and (
                        "/cases/" in full_url
                        or "/legis/" in full_url
                        or "/journals/" in full_url
                        or "/other/" in full_url
                    )
                )
            ):
                # Exclude navigation, search, help pages, and document links
                if (
                    "/databases.html" not in full_url
                    and "/search.html" not in full_url
                    and "/about.html" not in full_url
                    and "/contact.html" not in full_url
                    and "/help.html" not in full_url
                    and "/faq.html" not in full_url
                    and "/cgi-bin/viewdoc/" not in full_url  # Exclude document links
                    and "?view=" not in full_url  # Exclude view options
                    and "#" not in full_url
                    and "javascript:" not in full_url.lower()
                    and not full_url.endswith(".html")
                ):  # Database pages typically don't end with .html
                    database_links.add(full_url)

        print(f"Found {len(database_links)} database links")
        return sorted(list(database_links))
    except Exception as e:
        print(f"Error fetching database links: {e}")
        import traceback

        traceback.print_exc()
        return []


def get_document_links(driver, database_url, depth=0, max_depth=5):
    """
    Get links to individual documents from a database page.

    Args:
        driver: Selenium WebDriver instance
        database_url: URL of the database page
        depth: Current recursion depth (default: 0)
        max_depth: Maximum recursion depth to prevent stack overflow (default: 5)

    Returns:
        List of document URLs
    """
    # Prevent infinite recursion
    if depth >= max_depth:
        print(
            f"    Maximum depth ({max_depth}) reached for {database_url}, stopping recursion"
        )
        return []

    print(f"  Fetching documents from {database_url}... (depth: {depth})")
    try:
        driver.get(database_url)
        time.sleep(2)  # Wait for page to load

        soup = BeautifulSoup(driver.page_source, "html.parser")
        document_links = set()

        # Look for links to individual documents
        # AustLII document links use /cgi-bin/viewdoc/ (not viewdb)
        # Document URLs typically look like:
        # - /cgi-bin/viewdoc/au/cases/cth/HCA/2023/123.html
        # - /cgi-bin/viewdoc/au/legis/cth/consol_act/abc123/2023.html
        for link in soup.find_all("a", href=True):
            href = link.get("href", "")
            full_url = urljoin(BASE_URL, href)

            # Document links use /cgi-bin/viewdoc/ and contain year numbers or case identifiers
            if BASE_URL in full_url and "/cgi-bin/viewdoc/" in full_url:
                # Exclude navigation, search, and view pages
                if (
                    "/viewdb/" not in full_url
                    and "/search" not in full_url.lower()
                    and "/databases.html" not in full_url.lower()
                    and "/help.html" not in full_url.lower()
                    and "/faq.html" not in full_url.lower()
                    and "?view="
                    not in full_url  # Exclude view options like ?view=most_cited
                    and "#" not in full_url
                ):
                    # Check if URL looks like a document (has year or case identifier pattern)
                    # Documents typically have patterns like /2023/ or /2019/6.html
                    # Must have a year (4 digits) or end with .html with numbers
                    if (
                        re.search(r"/\d{4}/", full_url)  # Has year pattern like /2023/
                        or (
                            re.search(r"/\d+\.html$", full_url)
                            and "/cases/" in full_url
                        )  # Case document like /123.html
                        or (
                            re.search(r"/\d+\.html$", full_url)
                            and "/legis/" in full_url
                        )
                    ):  # Legislation document
                        document_links.add(full_url)

        # Also look for links in tables, lists, and divs (common structures for document listings)
        for container in soup.find_all(["table", "ul", "ol", "div"]):
            for link in container.find_all("a", href=True):
                href = link.get("href", "")
                full_url = urljoin(BASE_URL, href)
                if (
                    BASE_URL in full_url
                    and "/cgi-bin/viewdoc/" in full_url
                    and "?view=" not in full_url
                    and "#" not in full_url
                    and "/viewdb/" not in full_url
                ):
                    if (
                        re.search(r"/\d{4}/", full_url)
                        or (
                            re.search(r"/\d+\.html$", full_url)
                            and "/cases/" in full_url
                        )
                        or (
                            re.search(r"/\d+\.html$", full_url)
                            and "/legis/" in full_url
                        )
                    ):
                        document_links.add(full_url)

        # Also check for links that might be in list items with case names
        # Case names often appear as link text, and the href points to the document
        for li in soup.find_all("li"):
            link = li.find("a", href=True)
            if link:
                href = link.get("href", "")
                full_url = urljoin(BASE_URL, href)
                if (
                    BASE_URL in full_url
                    and "/cgi-bin/viewdoc/" in full_url
                    and "?view=" not in full_url
                    and "#" not in full_url
                    and "/viewdb/" not in full_url
                ):
                    if re.search(r"/\d{4}/", full_url) or (
                        re.search(r"/\d+\.html$", full_url)
                        and ("/cases/" in full_url or "/legis/" in full_url)
                    ):
                        document_links.add(full_url)

        # Check for pagination and get more links if available
        # Many AustLII databases have pagination with "Next" links
        page_links = soup.find_all(
            "a", href=True, string=re.compile(r"Next|next|More|more|»", re.I)
        )
        for page_link in page_links:
            href = page_link.get("href", "")
            next_url = urljoin(BASE_URL, href)
            if (
                BASE_URL in next_url and next_url != database_url
            ):  # Avoid infinite loops
                # Recursively get more documents (with limit and depth check)
                if len(document_links) < MAX_DOCUMENTS_PER_DATABASE:
                    more_links = get_document_links(
                        driver, next_url, depth=depth + 1, max_depth=max_depth
                    )
                    document_links.update(more_links)
                    if len(document_links) >= MAX_DOCUMENTS_PER_DATABASE:
                        break

        # Filter out any database listing pages that might have been incorrectly included
        filtered_links = []
        for link in document_links:
            # Make sure it's actually a document, not a database listing
            if "/cgi-bin/viewdoc/" in link and not link.endswith("/"):
                filtered_links.append(link)

        return filtered_links[:MAX_DOCUMENTS_PER_DATABASE]
    except Exception as e:
        print(f"    Error fetching documents from {database_url}: {e}")
        import traceback

        traceback.print_exc()
        return []


def scrape_document(driver, document_url, output_dir):
    """Scrape a single document and save it as a text file."""
    try:
        # Skip database listing pages - only process actual documents
        if "/viewdb/" in document_url or document_url.endswith("/"):
            return False

        driver.get(document_url)
        time.sleep(1)  # Wait for page to load

        # Extract legal text from page
        legal_text = extract_legal_text(driver.page_source)

        if not legal_text or len(legal_text) < 100:  # Minimum content length
            return False

        # Generate filename from URL
        parsed_url = urlparse(document_url)
        path_parts = [p for p in parsed_url.path.split("/") if p]

        # Create a meaningful filename
        if len(path_parts) >= 3:
            # Typically: /cgi-bin/viewdb/au/cases/cth/HCA/2023/123.html
            # We want: cth_HCA_2023_123
            filename_parts = (
                path_parts[-4:] if len(path_parts) >= 4 else path_parts[-3:]
            )
            filename = "_".join(filename_parts)
        else:
            filename = "_".join(path_parts) if path_parts else "document"

        # Remove .html extension if present
        filename = filename.replace(".html", "").replace(".htm", "")
        filename = sanitize_filename(filename)
        filename = f"{filename}.txt"

        filepath = os.path.join(output_dir, filename)

        # Handle duplicate filenames
        counter = 1
        while os.path.exists(filepath):
            name, ext = os.path.splitext(filename)
            filename = f"{name}_{counter}{ext}"
            filepath = os.path.join(output_dir, filename)
            counter += 1

        # Save the document
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"URL: {document_url}\n")
            f.write(f"Scraped: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            f.write(legal_text)

        return True
    except Exception as e:
        print(f"    Error scraping {document_url}: {e}")
        return False


def load_progress():
    """Load scraping progress from file."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_progress(progress):
    """Save scraping progress to file."""
    try:
        with open(PROGRESS_FILE, "w") as f:
            json.dump(progress, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save progress: {e}")


def main():
    """Main scraping function."""
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Starting AustLII legal document scraper (using browser automation)...")
    print(f"Output directory: {DATA_DIR}")
    print(f"Max databases: {MAX_DATABASES if MAX_DATABASES else 'Unlimited'}")
    print(f"Max documents per database: {MAX_DOCUMENTS_PER_DATABASE}")
    print()

    # Load progress
    progress = load_progress()
    processed_databases = set(progress.get("processed_databases", []))
    processed_documents = set(progress.get("processed_documents", []))

    # Set up browser
    print("Setting up browser...")
    driver = setup_driver()

    try:
        # Get all database links
        database_links = get_database_links(driver)

        if not database_links:
            print("No database links found. Exiting.")
            return

        # Filter out already processed databases if resuming
        if processed_databases:
            database_links = [
                db for db in database_links if db not in processed_databases
            ]
            print(f"Resuming: {len(database_links)} databases remaining to process")

        if MAX_DATABASES:
            database_links = database_links[:MAX_DATABASES]

        print(f"Processing {len(database_links)} databases...")
        print()

        total_documents = 0
        successful_documents = 0

        for i, database_url in enumerate(database_links, 1):
            print(f"[{i}/{len(database_links)}] Processing database: {database_url}")

            # Skip if already processed
            if database_url in processed_databases:
                print(f"  Already processed, skipping...")
                continue

            # Get document links from this database
            document_links = get_document_links(driver, database_url)

            if not document_links:
                print(f"  No documents found in this database")
                processed_databases.add(database_url)
                save_progress(
                    {
                        "processed_databases": list(processed_databases),
                        "processed_documents": list(processed_documents),
                    }
                )
                continue

            print(f"  Found {len(document_links)} documents")

            # Scrape each document
            for j, doc_url in enumerate(document_links, 1):
                # Skip if already processed
                if doc_url in processed_documents:
                    continue

                print(f"    [{j}/{len(document_links)}] Scraping: {doc_url[:80]}...")
                if scrape_document(driver, doc_url, DATA_DIR):
                    successful_documents += 1
                    processed_documents.add(doc_url)
                total_documents += 1

                # Be respectful - delay between requests
                time.sleep(DELAY_BETWEEN_REQUESTS)

                # Save progress periodically
                if j % 10 == 0:
                    save_progress(
                        {
                            "processed_databases": list(processed_databases),
                            "processed_documents": list(processed_documents),
                        }
                    )

            # Mark database as processed
            processed_databases.add(database_url)
            save_progress(
                {
                    "processed_databases": list(processed_databases),
                    "processed_documents": list(processed_documents),
                }
            )

            print()

        print("=" * 80)
        print(f"Scraping complete!")
        print(f"Total documents attempted: {total_documents}")
        print(f"Successfully scraped: {successful_documents}")
        print(f"Documents saved to: {DATA_DIR}")

    finally:
        driver.quit()
        print("Browser closed.")


if __name__ == "__main__":
    main()
