#!/usr/bin/env python3
import requests
from bs4 import BeautifulSoup
import sqlite3
import time


# URL of the job listings page
URL01 = "https://www.upwork.com/freelance-jobs/data-analysis/"
URL02 = "https://www.upwork.com/freelance-jobs/data-science/"

# SQLite database file
DB_FILE = "upwork_data_jobs.db"

def create_database():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS upwork_data_jobs_listings (scraped_on DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL, id INTEGER PRIMARY KEY, title TEXT, responsibilities TEXT, expertise TEXT, skills TEXT, num_of_days_posted TEXT, offer_bid TEXT, url TEXT)''')
    conn.commit()
    conn.close()

def scrape_jobs():
    response = requests.get(URL01)
    soup = BeautifulSoup(response.text, "html.parser")
    print(soup.prettify())

    jobs = []
    listings = soup.find_all("section", class_="air3-card")
    # print(listings)

    for listing in listings:
        title = listing.find("a", class_="job-title").text.strip()
        responsibility = listing.find("p", class_="job-description").text.strip()
        expertise = listing.find_all(attrs={"data-qa": "expert-level"}).strong.extract().text.strip()
        all_skills = listing.find_all("span", class_="air3-token")
        skills = ", ".join([skill.text.strip() for skill in all_skills])
        num_of_days_posted = listing.find_all("small", class_="text-muted-on-inverse")[-1].text.strip()
        offer_bid = listing.find_all(attrs={"data-qa": "hours-needed", "data-v-b6045cea": ""}).strong.extract().text.strip()
        job_url = listing.find("a", class_="up-n-link air3-btn air3-btn-primary air3-btn-sm air3-btn-block-sm mt-6x mb-4x mb-md-3x px-4x")["href"]
        job_url = listing.select("a.up-n-link.air3-btn.air3-btn-primary.air3-btn-sm.air3-btn-block-sm")["href"]

        jobs.append((title, responsibility, expertise, skills, num_of_days_posted, offer_bid, job_url))
    # Return the list of jobs
    return jobs

def save_jobs_to_db(jobs):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.executemany("INSERT INTO upwork_data_jobs_listings (title, responsibilities, expertise, skills, num_of_days_posted, offer_bid, url) VALUES (?,?,?,?,?,?,?)", jobs)
    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_database()
    while True:
        jobs = scrape_jobs()
        save_jobs_to_db(jobs)
        print(f"{len(jobs)} jobs scraped and saved to database.")
        time.sleep(1553) # Wait for 243rd prime seconds before scraping again

