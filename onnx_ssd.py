
import torch
import onnx
import cv2
import numpy as np
from vision.ssd.data_preprocessing import PredictionTransform,ONNXPredictionTransform
import caffe2.python.onnx.backend as backend
from vision.ssd.config import mobilenetv1_ssd_config as config

from torchsummary import summary
from vision.utils import box_utils


from vision.ssd.mobilenet_v3_ssd_lite import create_mobilenetv3_ssd_lite,create_mobilenetv3_ssd_lite_predictor

# net_type = "mb3-ssd-lite"
PRED = True
model_path = "models/mb3-ssd-lite-Epoch-120-Loss-2.1994679314749583.pth"
label_path = "models/voc-model-labels.txt"
image_path = "/home/grobot/mywork/cocodata/fuzi/testfu/testfu12.jpg"


def predict(image,c2_out,top_k=10, prob_threshold=0.4):

    cpu_device = torch.device("cpu")
    height, width, _ = image.shape
    boxes = torch.from_numpy(c2_out[1][0].astype(np.float32)) 
    scores = torch.from_numpy(c2_out[0][0].astype(np.float32)) 
    boxes = boxes.to(cpu_device)
    scores = scores.to(cpu_device)
    
    
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, scores.size(1)):
        probs = scores[:, class_index]
        print(probs.size())
        mask = probs > prob_threshold
        probs = probs[mask]
        print(probs.size(),prob_threshold)
        if probs.size(0) == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
        box_probs = box_utils.nms(box_probs, None,
                                  score_threshold=prob_threshold,
                                  iou_threshold=config.iou_threshold,
                                  sigma=0.5,
                                  top_k=top_k,
                                  candidate_size=200)
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.size(0))
    if not picked_box_probs:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])
    picked_box_probs = torch.cat(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]

class_names = [name.strip() for name in open(label_path).readlines()]

if ~PRED:

    net = create_mobilenetv3_ssd_lite(len(class_names), is_test=True)

    net.load(model_path)

    dummy_input = torch.randn(1,3,config.image_size, config.image_size)
    input_names = ["input0"]
    output_names = ["output0"]
    # print("strt input")
    torch_out = torch.onnx._export(net,             # model being run
                                dummy_input,                       # model input (or a tuple for multiple inputs)
                                "mobilenet_v3_ssd_lite.onnx", # where to save the model (can be a file or file-like object)
                                verbose=False,
                                export_params=True,
                                input_names=input_names, output_names=output_names)      # store the trained parameter weights inside the model file

if PRED:


    # pic = "/home/grobot/mywork/cocodata/fuzi/testfu/testfu12.jpg"
    # Load the ONNX model
    onnx_file = "mobilenet_v3_ssd_lite.onnx"
    model = onnx.load(onnx_file)
    
    # # Check that the IR is well formed
    onnx.checker.check_model(model)

    # # Print a human readable representation of the graph
    # onnx.helper.printable_graph(model.graph)

    transform = PredictionTransform(config.image_size, config.image_mean, config.image_std)

    orig_image = cv2.imread(image_path)
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

    image = transform(image)
    images = image.unsqueeze(0)
    x = images
    print(x.size())
    caffe2_backend = backend.prepare(model)
    # summary(caffe2_backend,(3,300,300))
    B = {"input0": x.data.numpy()}
    c2_out = caffe2_backend.run(B)
    # c2_out = caffe2_backend.run(B)
    # print(torch_out[0].detach().numpy())
    # print(c2_out[1].shape)
    # with open("onnxout_0.txt", "w+") as log_file:
    #     print(c2_out[0].shape)
    #     np.savetxt(log_file,  c2_out[0][0])
    #     print(c2_out[1].shape)
    #     np.savetxt(log_file,  c2_out[1][0])

    boxes, labels, probs = predict(orig_image,c2_out)
    print(boxes.size(),labels.size())
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
        #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        cv2.putText(orig_image, label,
                    (box[0] + 20, box[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
    outputdir = "./outputtest/"

    path = outputdir+"output_onnx_3.jpg"
    cv2.imwrite(path, orig_image)
    print(f"Found {len(probs)} objects. The output image is {path}")
    # print("==> compare torch output and caffe2 output")
    # np.testing.assert_almost_equal(torch_out[0].detach().numpy(), c2_out[0], decimal=3)
    # print("==> Passed")