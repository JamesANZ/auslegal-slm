# Legal Document Scraper

This directory contains tools for scraping Australian legal documents from AustLII.

## Files

- `scraper.py` - Main scraper script that uses Selenium to scrape legal documents
- `requirements.txt` - Python dependencies for the scraper
- `scraper.log` - Scraper execution logs

## Usage

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the scraper:

```bash
python scraper.py
```

The scraper will:

- Discover databases on AustLII
- Scrape legal documents from each database
- Save documents as text files to `../data/` directory
- Track progress in `.scraper_progress.json`

## Configuration

Edit `scraper.py` to adjust:

- `MAX_DOCUMENTS_PER_DATABASE` - Limit documents per database
- `MAX_DATABASES` - Limit number of databases to process
- `DELAY_BETWEEN_REQUESTS` - Rate limiting delay

## Output

Scraped documents are saved to `../data/` directory with filenames based on the document URL structure.
