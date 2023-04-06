from pinferencia import Server, task
from PIL import Image
from io import BytesIO
import base64
import json
from typing import List

import layoutparser as lp
import cv2

model = lp.AutoLayoutModel("lp://detectron2/PrimaLayout/mask_rcnn_R_50_FPN_3x",
                           label_map = {1:"TextRegion", 2:"ImageRegion", 3:"TableRegion",
                                        4:"MathsRegion", 5:"SeparatorRegion", 6:"OtherRegion"},
                           extra_config = ["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.85,
                                           "MODEL.ROI_HEADS.NMS_THRESH_TEST", 0.75])


label_mapping = {'TextRegion': 'text', 'ImageRegion': 'image', 'TableRegion': 'table',
                 'MathRegion': 'equation', 'SeparatorRegion': 'separator', 'OtherRegion': 'other'}


# base64 -> Image
# In test file, turn image to base64

def parse_layout(layout, ocr_agent, image, ocr_selected=False):
    layout_collections = list()
    for ob in layout:
        layout_dic = dict()
        layout_dic['id'] = ob.id
        layout_dic['detect_type'] = ob.type
        #if ocr_selected:
        #    layout_dic['text'] = detect_text(ocr_agent, ob, image)
        layout_dic['parent'] = ob.parent
        layout_dic['rect_left'] = ob.block.coordinates[0]
        layout_dic['rect_right'] = ob.block.coordinates[1]

        layout_dic['confidence'] = ob.score
        # layout_dic['']
        layout_collections.append(layout_dic)
    detected_info = pd.DataFrame(layout_collections)

    return detected_info

def transform_and_detect(image_base64_str):
    '''

    :param image_base64_str: string of image in base64 format
    :return: parsed_layout: layout parsed into list
    '''
    input_img = Image.open(BytesIO(base64.b64decode(image_base64_str)))
    layout = model.detect(input_img)

    parsed_layout = parse_layout_api(layout, label_mapping)
    return parsed_layout


def predict(image_base64_str: str)-> str:
    '''

    :param image_base64_str: string of image in base64 format
    :return:  json_layout: layout response
    '''
    parsed_layout = transform_and_detect(image_base64_str)
    json_layout = json.dumps(parsed_layout, indent=2)

    return json_layout


service = Server()
service.register(
    model_name="layout",
    model=predict,
    metadata={"task": task.IMAGE_CLASSIFICATION,
              "display_name": 'Layout Detection'})
