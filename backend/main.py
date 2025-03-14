import os
import numpy as np
import pandas as pd
import random

# import cv2 as cv
from matplotlib import pyplot as plt

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welcome to the Generic API"}
