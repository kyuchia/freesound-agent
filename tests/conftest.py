import os

# agent.py configures the Freesound client at import time and crashes on a
# missing key, so set dummy keys before any test imports it. Real keys from
# .env are never used: load_dotenv() does not override variables already set,
# and the tests block all network access anyway.
os.environ["ANTHROPIC_API_KEY"] = "test-dummy-key"
os.environ["FREESOUND_API_KEY"] = "test-dummy-key"
