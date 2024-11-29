from ultralytics import YOLO

# para marcar as imagens
# https://www.makesense.ai/

def main():
    # Load a model
    #model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model = YOLO("yolo11s.pt")  # load a pretrained model (recommended for training)

    # Use the model
    model.train(data=r"C:/Users/achal/Downloads/# Git/Treinamento YOLO Impressora 3D.v1.00/impressora.yaml", epochs=70, device=0)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    # path = model.export(format="onnx")  # export the model to ONNX format
    # print("path", path)


if __name__ == '__main__':
    # freeze_support()
    main()
