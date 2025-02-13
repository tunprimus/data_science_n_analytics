#!/usr/bin/env python3
import requests
from bs4 import BeautifulSoup
import sqlite3
import time


# URL of the job listings page
URL = "https://www.freelancer.com/jobs/data-analytics/"

# SQLite database file
DB_FILE = "freelancer_jobs.db"

def create_database():
    """
    Create a SQLite database table if it doesn't already exist, to store scraped
    job listings.

    The table has the following columns:

    - `scraped_on`: The date and time that the job listing was scraped.
    - `id`: A unique identifier for the job listing.
    - `title`: The title of the job listing.
    - `responsibilities`: The responsibilities for the job listing.
    - `days_left`: The number of days left to bid on the job listing.
    - `avg_bid`: The average bid on the job listing.
    - `url`: The URL of the job listing.
    """
    with sqlite3.connect(DB_FILE) as conn:
        cur = conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS freelancer_data_analytics_jobs (scraped_on DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL, id INTEGER PRIMARY KEY, title TEXT, responsibilities TEXT, days_left TEXT, avg_bid TEXT, url TEXT)''')
        conn.commit()
        # conn.close()

def scrape_jobs():
    """
    Scrape job listings from the data analytics section of Freelancer.com.

    The function sends a GET request to the URL of the job listings page,
    parses the HTML response with BeautifulSoup, and extracts the job title,
    responsibilities, number of days left to bid, average bid, and URL of each
    job listing into a tuple. The function then returns a list of these tuples.

    :return: A list of tuples containing scraped job listings.
    :rtype: list
    """
    response = requests.get(URL)
    soup = BeautifulSoup(response.text, "html.parser")

    jobs = []
    listings = soup.find_all("div", class_="JobSearchCard-item")

    for listing in listings:
        title = listing.find("a", class_="JobSearchCard-primary-heading-link").text.strip()
        responsibility = listing.find("p", class_="JobSearchCard-primary-description").text.strip()
        days_left = listing.find("span", class_="JobSearchCard-primary-heading-days").text.strip()
        avg_bid = listing.find("div", class_="JobSearchCard-primary-price").text.strip()
        job_url_frag = listing.find("a", class_="JobSearchCard-primary-heading-link")["href"]
        job_url = f"https://www.freelancer.com{job_url_frag}"

        jobs.append((title, responsibility, days_left, avg_bid, job_url))
    # Return the list of jobs
    return jobs

def save_jobs_to_db(jobs):
    """
    Save job listings to the database.

    This function connects to the SQLite database specified by DB_FILE, and
    inserts job listings into the 'freelancer_data_analytics_jobs' table. Each
    job listing is expected to be a tuple containing the title, responsibilities,
    days left to bid, average bid, and the URL of the job.

    :param jobs: A list of tuples, each containing job listing details.
    :type jobs: list
    """
    with sqlite3.connect(DB_FILE) as conn:
        cur = conn.cursor()
        cur.executemany("INSERT INTO freelancer_data_analytics_jobs (title, responsibilities, days_left, avg_bid, url) VALUES (?,?,?,?,?)", jobs)
        conn.commit()

if __name__ == "__main__":
    create_database()
    while True:
        jobs = scrape_jobs()
        save_jobs_to_db(jobs)
        print(f"{len(jobs)} jobs scraped and saved to database.")
        time.sleep(1553) # Wait for 243rd prime seconds before scraping again

