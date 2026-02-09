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

        # collect rectangle boxes from context
        boxes = []
        for ctx in context['result']:
            ctx_type = ctx['type']
            if ctx_type == 'rectanglelabels':
                x = ctx['value']['x'] * image_width / 100
                y = ctx['value']['y'] * image_height / 100
                w = ctx['value']['width'] * image_width / 100
                h = ctx['value']['height'] * image_height / 100
                boxes.append([int(x), int(y), int(x + w), int(y + h)])

        if not boxes:
            return ModelResponse(predictions=[])

        # load image
        img_url = tasks[0]['data'][value]
        image_path = get_local_path(img_url, task_id=tasks[0].get('id'))
        image = Image.open(image_path).convert("RGB")

        # run model
        inputs = processor(
            images=image,
            text=TEXT_PROMPT,
            input_boxes=[boxes],
            input_boxes_labels=[[1] * len(boxes)],
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        # post-process
        sam_results = processor.post_process_instance_segmentation(
            outputs,
            threshold=SCORE_THRESHOLD,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]

        # build predictions
        results = []
        total_score = 0
        for mask_tensor, score_tensor in zip(sam_results['masks'], sam_results['scores']):
            mask = (mask_tensor.cpu().numpy() > 0).astype(np.uint8) * 255
            score = float(score_tensor.cpu().item())

            label_id = str(uuid4())[:9]
            rle = brush.mask2rle(mask)

            results.append({
                'id': label_id,
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
                'score': score,
                'type': 'brushlabels',
                'readonly': False
            })
            total_score += score

        predictions = [{
            'result': results,
            'model_version': self.get('model_version'),
            'score': total_score / max(len(results), 1)
        }]

        return ModelResponse(predictions=predictions)
