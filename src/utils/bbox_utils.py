class BboxUtils:
    def get_iou(self, innerbbox, outerbbox):
        # Get bbox coords
        ix1, iy1, ix2, iy2 = [int(i) for i in innerbbox]
        ox1, oy1, ox2, oy2 = [int(i) for i in outerbbox]
        # top-left (1) and bottom right (2)

        # Determine the intersection box coords
        # Get the largest coordinates from the top left of the bboxes
        x1 = max(ix1, ox1)
        y1 = max(iy1, oy1)
        # Get the smallest coordinates from the bottom right of the bboxes
        x2 = min(ix2, ox2)
        y2 = min(iy2, oy2)

        # Get the width and height of the intersection box
        interWidth = max(0, x2 - x1) # Subtracting the bottom right by the top left to get the width
        interHeight = max(0, y2 - y1)

        # Get the area of the intersection box
        interArea = interWidth * interHeight

        # Get the area of the bboxes
        innerBoxArea = (ix2 - ix1) * (iy2 - iy1) # Get the width and height of the bbox then multiply with each other
        outerBoxArea = (ox2 - ox1) * (oy2 - oy1)

        # Calculate the union area
        unionArea = (innerBoxArea + outerBoxArea) - interArea
        # union is the area of both boxes combined, minus the area that intersects between both boxes

        # Checking if is divisible
        if unionArea == 0:
            return 0

        # Return the IoU
        return interArea / unionArea
    
    def xyxy_to_xywh(xyxy):
        x1, y1, x2, y2 = xyxy
        
        if x2 < x1 or y2 < y1:
            raise ValueError(f"Invalid bbox: {xyxy}")
        
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        
        return [x1, y1, w, h]
    
    # top-left
    def xywh_to_xyxy_yolo(xywh):
        x1, y1, w, h = xywh
        y2 = y1 + h
        x2 = x1 + w
        return [x1, y1, x2, y2]
    
    # center
    def xywh_to_xyxy_coco(xywh):
        x_c, y_c, w, h = xywh
        x1 = x_c - w / 2
        y1 = y_c - h / 2
        x2 = x_c + w / 2
        y2 = y_c + h / 2
        return [x1, y1, x2, y2]