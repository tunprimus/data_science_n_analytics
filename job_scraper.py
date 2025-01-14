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
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS data_analytics_job_listings (id INTEGER PRIMARY KEY, title TEXT, responsibilities TEXT, days_left TEXT, avg_bid TEXT, url TEXT, scraped_on DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL)''')
    conn.commit()
    conn.close()

def scrape_jobs():
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
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.executemany("INSERT INTO data_analytics_job_listings (title, responsibilities, days_left, avg_bid, url) VALUES (?,?,?,?,?)", jobs)
    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_database()
    while True:
        jobs = scrape_jobs()
        save_jobs_to_db(jobs)
        print(f"{len(jobs)} jobs scraped and saved to database.")
        time.sleep(3600) # Wait for 1 hour before scraping again

