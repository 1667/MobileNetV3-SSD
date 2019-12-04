import torch

from ..utils import box_utils
from .data_preprocessing import PredictionTransform
from ..utils.misc import Timer
import numpy as np
import onnx
import caffe2.python.onnx.backend as backend


class OnnxPredictor:
    def __init__(self, net, size, mean=0.0, std=1.0, nms_method=None,
                 iou_threshold=0.45, filter_threshold=0.01, candidate_size=200, sigma=0.5, device=None):
        self.net = net
        self.transform = PredictionTransform(size, mean, std)
        self.iou_threshold = iou_threshold
        self.filter_threshold = filter_threshold
        self.candidate_size = candidate_size
        self.nms_method = nms_method
        self.model = onnx.load("mobilenet_v3_ssd_lite.onnx")
        self.sigma = sigma
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.net.to(self.device)
        self.net.eval()

        self.timer = Timer()

    # def print_Tensor_encoded(self, epoch, i, tensors):
    #     message = '(epoch: %d, iters: %d)' % (epoch, i)
    #     for k, v in tensors.items():
    #         with open(self.log_tensor_name, "a") as log_file:
    #             v_cpu = v.cpu()
    #             log_file.write('%s: ' % message)
    #             np.savetxt(log_file,  v_cpu.detach().numpy())

    def predict(self, image, top_k=-1, prob_threshold=None):
        cpu_device = torch.device("cpu")
        height, width, _ = image.shape

        image = self.transform(image)
        

        print(image.size())
        images = image.unsqueeze(0)
        print(image.size())
        # print("image==== ",image)
        # with open("input_0.txt", "a") as log_file:
        #     print(type(image.numpy()))
        #     np.savetxt(log_file,  image.numpy()[0])
        #     np.savetxt(log_file,  image.numpy()[1])
        #     np.savetxt(log_file,  image.numpy()[2])

        images = images.to(self.device)
        with torch.no_grad():
            self.timer.start()
            # scores, boxes = self.net.forward(images)
            caffe2_backend = backend.prepare(self.model)
            B = {"input0": images.data.numpy()}
            scores, boxes = caffe2_backend.run(B)
            boxes = torch.from_numpy(boxes.astype(np.float32)) 
            scores = torch.from_numpy(scores.astype(np.float32)) 
            # with open("preout_0.txt", "w+") as log_file:
            #     print(scores.size())
            #     np.savetxt(log_file,  scores.numpy()[0])
            #     print(boxes.size())
            #     np.savetxt(log_file,  boxes.numpy()[0])
            print("Inference time: ", self.timer.end())
        boxes = boxes[0]
        scores = scores[0]
        if not prob_threshold:
            prob_threshold = self.filter_threshold
        # this version of nms is slower on GPU, so we move data to CPU.
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
            box_probs = box_utils.nms(box_probs, self.nms_method,
                                      score_threshold=prob_threshold,
                                      iou_threshold=self.iou_threshold,
                                      sigma=self.sigma,
                                      top_k=top_k,
                                      candidate_size=self.candidate_size)
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