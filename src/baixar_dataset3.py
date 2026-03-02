from roboflow import Roboflow

rf = Roboflow(api_key="XNHkx2ChhCDORHfZI8fR")
project = rf.workspace("elementos-urbanos").project("esquina-h5grp")
dataset = project.version(2).download("yolov8")

print("Dataset salvo em:", dataset.location)