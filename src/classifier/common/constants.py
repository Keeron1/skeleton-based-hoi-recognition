PERSON_CAT_ID = 1
OBJECT_CAT_IDS = [2, 3, 4]  # laptop, cell phone, book
OBJECT_CAT_NAMES = ["laptop", "cell phone", "book"]

# YOLO class name to dataset category id. Used by the live pipeline to convert
# tracker outputs into the id space the GNN's one-hot expects.
NAME_TO_CAT_ID = dict(zip(OBJECT_CAT_NAMES, OBJECT_CAT_IDS))

NUM_KEYPOINTS = 17
OBJECT_NODE_IDX = 17
NUM_NODES = NUM_KEYPOINTS + 1

ACTION_NAMES = ["idle", "using_laptop", "using_phone", "reading"]


# Maps dataset category id to 0-indexed object class used by the bbox feature extractor
def object_class_to_index(cat_id):
    if cat_id not in OBJECT_CAT_IDS:
        return None
    return OBJECT_CAT_IDS.index(cat_id)
