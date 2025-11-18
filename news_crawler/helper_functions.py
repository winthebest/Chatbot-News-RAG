import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service  # Still import, in case needed later

def configure_driver():
    '''Configure the webdriver'''

    chrome_options = Options()
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.set_capability('pageLoadStrategy', 'none')
    chrome_options.add_argument('--headless')
    chrome_options.add_argument("--disable-features=ScriptStreaming")
    chrome_options.add_argument("--disable-features=PreloadMediaEngagementData")
    chrome_options.add_experimental_option("prefs", {
        "profile.managed_default_content_settings.images": 2,
        "profile.managed_default_content_settings.videos": 2
    })
    chrome_options.add_experimental_option('prefs', {'intl.accept_languages': 'en,en_US'})

 
    driver = webdriver.Chrome(options=chrome_options)
    return driver
