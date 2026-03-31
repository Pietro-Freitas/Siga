import os
from roboflow import Roboflow
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")
rf = Roboflow(api_key=api_key)

project = rf.workspace("elementos-urbanos").project("esquina-h5grp")
dataset = project.version(2).download("yolov8")

print("Dataset salvo em:", dataset.location)