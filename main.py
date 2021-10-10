# %% Import libararies
import os
import numpy as np
import pandas as pd
import requests
import nest_asyncio
import asyncpg
from dotenv import load_dotenv
load_dotenv()
nest_asyncio.apply()
# %% keys
API_KEY = os.getenv('API_KEY')

