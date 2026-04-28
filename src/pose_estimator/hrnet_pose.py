from mmpose.apis import init_model, inference_topdown

# https://mmpose.readthedocs.io/en/latest/model_zoo/body_2d_keypoint.html#topdown-heatmap-hrnet-on-coco
# https://mmpose.readthedocs.io/en/latest/model_zoo/body_2d_keypoint.html#topdown-heatmap-hrnet-udp-on-coco

# First test with pose_hrnet_w32_256x192 then move to the 384x288 then test with the udp models
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
        # data_sample.pred_instances.keypoints and data_sample.pred_instances.keypoint_scores
        return batch_results