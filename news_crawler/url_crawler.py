import sys
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import csv
import os
from helper_functions import configure_driver

# Get page range from command line arguments
start_page = int(sys.argv[1])
end_page = int(sys.argv[2])

# Initialize browser
driver = configure_driver()

# Set of collected links
hrefs = set()

# Iterate through pages
temp = len(hrefs)
for i in range(start_page, end_page + 1):
    url = f"https://vnexpress.net/cong-nghe/ai-p{i}" if i > 1 else "https://vnexpress.net/cong-nghe/ai"
    driver.get(url)
    time.sleep(3)  # Wait 3 seconds before stopping load

    # Find all <a> tags with data-medium and href attributes
    links = driver.find_elements(By.XPATH, "//a[@data-medium and @href]")
    
    # Save list of paths (only get links ending with ".html")
    hrefs.update(link.get_attribute("href") for link in links if link.get_attribute("href").endswith(".html"))
    
    if temp == len(hrefs):
        print(" Collected enough links. Ending...")
        break
    temp = len(hrefs)
    print(hrefs)

# Close browser
driver.quit()

# Print to screen
for href in sorted(hrefs):
    print(href)

# Save to CSV file
output_file = "data/raw/vnexpress_links.csv"
with open(output_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["URL"])
    for href in sorted(hrefs):
        writer.writerow([href])

print(f" Saved {len(hrefs)} valid links to {output_file}")