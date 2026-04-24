from mmpose.apis import init_model, inference_topdown

class HrNetPose:
    def __init__(self, model_cfg, ckpt, device):
        # config path, model weights, device where the anchors will be put
        self.model = init_model(model_cfg, ckpt, device)

    def infer(self, frame, bboxes):
        if len(bboxes) == 0:
            print("Empty bboxes detected while trying to infer poses...")
            return []
        
        # model, frame, bboxes, bboxes format
        batch_results = inference_topdown(self.model, frame, bboxes, "xyxy")
        return batch_results