import os
import numpy as np
import torch
from typing import List, Dict, Optional
from uuid import uuid4

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path
from label_studio_sdk.converter import brush
from PIL import Image
from transformers import Sam3Processor, Sam3Model

DEVICE = os.getenv('DEVICE', 'cuda')
MODEL_NAME = os.getenv('MODEL_NAME', 'facebook/sam3')
TEXT_PROMPT = os.getenv('TEXT_PROMPT', 'food')
SCORE_THRESHOLD = float(os.getenv('SCORE_THRESHOLD', '0.25'))
# Label name → prompt mode mapping
# "food" (or TEXT_PROMPT value) → text+box, "box" → box only, "text" → text only
BOX_ONLY_LABEL = os.getenv('BOX_ONLY_LABEL', 'box')
TEXT_ONLY_LABEL = os.getenv('TEXT_ONLY_LABEL', 'text')

if DEVICE == 'cuda' and not torch.cuda.is_available():
    print("WARNING: CUDA requested but not available. Falling back to CPU.")
    DEVICE = 'cpu'

model = Sam3Model.from_pretrained(MODEL_NAME).to(DEVICE)
processor = Sam3Processor.from_pretrained(MODEL_NAME)


class Sam3Backend(LabelStudioMLBase):

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        from_name, to_name, value = self.get_first_tag_occurence('BrushLabels', 'Image')

        if not context or not context.get('result'):
            return ModelResponse(predictions=[])

        image_width = context['result'][0]['original_width']
        image_height = context['result'][0]['original_height']

        # collect rectangle boxes and determine prompt mode from label
        boxes = []
        selected_label = None
        for ctx in context['result']:
            ctx_type = ctx['type']
            if ctx_type == 'rectanglelabels':
                x = ctx['value']['x'] * image_width / 100
                y = ctx['value']['y'] * image_height / 100
                w = ctx['value']['width'] * image_width / 100
                h = ctx['value']['height'] * image_height / 100
                boxes.append([int(x), int(y), int(x + w), int(y + h)])
                selected_label = ctx['value'].get('rectanglelabels', [None])[0]

        if not boxes:
            return ModelResponse(predictions=[])

        # determine prompt mode from selected label
        if selected_label == BOX_ONLY_LABEL:
            prompt_mode = 'box'
        elif selected_label == TEXT_ONLY_LABEL:
            prompt_mode = 'text'
        else:
            prompt_mode = 'text+box'

        print(f'Prompt mode: {prompt_mode} (label={selected_label})')

        # load image
        img_url = tasks[0]['data'][value]
        image_path = get_local_path(img_url, task_id=tasks[0].get('id'))
        image = Image.open(image_path).convert("RGB")

        # run model with prompt mode determined by label
        proc_kwargs = dict(images=image, return_tensors="pt")
        if prompt_mode in ('text+box', 'text'):
            proc_kwargs['text'] = TEXT_PROMPT
        if prompt_mode in ('text+box', 'box'):
            proc_kwargs['input_boxes'] = [boxes]
            proc_kwargs['input_boxes_labels'] = [[1] * len(boxes)]
        inputs = processor(**proc_kwargs).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        # post-process
        sam_results = processor.post_process_instance_segmentation(
            outputs,
            threshold=SCORE_THRESHOLD,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]

        # build bbox mask for IoU computation
        bbox_mask = np.zeros((image_height, image_width), dtype=bool)
        for box in boxes:
            x1, y1, x2, y2 = box
            bbox_mask[y1:y2, x1:x2] = True
        bbox_area = bbox_mask.sum()

        # select mask with highest IoU against the drawn bbox
        best_mask = None
        best_score = 0
        best_iou = 0
        for mask_tensor, score_tensor in zip(sam_results['masks'], sam_results['scores']):
            mask = (mask_tensor.cpu().numpy() > 0).astype(np.uint8)
            intersection = (mask & bbox_mask).sum()
            union = mask.sum() + bbox_area - intersection
            iou = intersection / max(union, 1)
            if iou > best_iou:
                best_iou = iou
                best_mask = mask
                best_score = float(score_tensor.cpu().item())

        results = []
        if best_mask is not None:
            rle = brush.mask2rle(best_mask * 255)
            results.append({
                'id': str(uuid4())[:9],
                'from_name': from_name,
                'to_name': to_name,
                'original_width': image_width,
                'original_height': image_height,
                'image_rotation': 0,
                'value': {
                    'format': 'rle',
                    'rle': rle,
                    'brushlabels': [TEXT_PROMPT],
                },
                'score': best_score,
                'type': 'brushlabels',
                'readonly': False
            })

        predictions = [{
            'result': results,
            'model_version': self.get('model_version'),
            'score': best_score
        }]

        return ModelResponse(predictions=predictions)
