import os


outputdir = "./outputtest/"
os.system("rm -rf "+outputdir+"*")
picdir = "/home/grobot/mywork/firedetection/MobileNetV3-SSD-master/testpic/"
models_file = "models/mb3-ssd-lite-Epoch-195-Loss-2.8794922828674316.pth"
# models_file = "models/mb3-ssd-lite-Epoch-95-Loss-2.2009710584368025.pth"

pics = os.listdir(picdir)

# pics = ["/home/grobot/mywork/cocodata/fuzi/testfu/fuzi2.jpg",
#         "/home/grobot/mywork/cocodata/fuzi/testfu/fuzi1.jpg",
#         "/home/grobot/mywork/cocodata/fuzi/testfu/123.jpeg",
#         "/home/grobot/mywork/cocodata/fuzi/testfu/futest.jpg",
#         "/home/grobot/mywork/cocodata/fuzi/testfu/fuzi_221.jpg",
#         "/home/grobot/mywork/cocodata/fuzi/testfu/testfu3.png",
#         "/home/grobot/mywork/cocodata/fuzi/testfu/testfu4.png"]

for pic in pics:
    print(pic)
    os.system("python run_ssd_example.py mb3-ssd-lite "+ models_file +" models/voc-model-labels.txt "+ picdir +pic)
# python run_ssd_example.py mb3-ssd-lite models/mb3-ssd-lite-Epoch-110-Loss-2.9901715914408364.pth models/voc-model-labels.txt /home/grobot/mywork/cocodata/fuzi/testfu/123.jpeg
#  models/mb3-ssd-lite-Epoch-49-Loss-3.7712578773498535.pth  
# python run_ssd_example.py mb3-ssd-lite models/mb3-ssd-lite-Epoch-110-Loss-2.9901715914408364.pth models/voc-model-labels.txt /home/grobot/mywork/cocodata/fuzi/testfu/futest.jpg