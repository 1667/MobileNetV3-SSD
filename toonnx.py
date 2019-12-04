import os

models_file = "models/mb3-ssd-lite-Epoch-120-Loss-2.1994679314749583.pth"
pic = "/home/grobot/mywork/cocodata/fuzi/testfu/testfu12.jpg"

os.system("python pytorchtoonnx.py onnx_mb3-ssd-lite "+ models_file +" models/voc-model-labels.txt "+pic)
