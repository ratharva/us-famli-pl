from ultralytics import YOLO

# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)


model.train(data="/mnt/raid/home/ayrisbud/USOD/usod.yaml", epochs=200, imgsz=600)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
success = model.export(format="torchscript")