from flask import Blueprint, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import accuracy_score
from IPython.display import Markdown
import google.generativeai as genai
from IPython.display import display
import pandas as pd
import numpy as np
import PIL.Image
import textwrap
import pathlib
import imgkit
import json
import csv
from flask import Flask, render_template, request, Blueprint
from sklearn.metrics import classification_report
