# SAM3 Image Segmentation Backend

Interactive food segmentation using SAM3 (Segment Anything 3). User draws rectangle in Label Studio → SAM3 returns brush mask for "food" within that region.

## Quick Start

```bash
# Set required env vars
export HF_TOKEN=your_huggingface_token
export LABEL_STUDIO_URL=http://host.docker.internal:8080
export LABEL_STUDIO_API_KEY=your_api_key

# Build and run
docker compose up --build
```

Backend runs at `http://localhost:9090`.

## Label Studio Config

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="rect" toName="image">
    <Label value="food"/>
  </RectangleLabels>
  <BrushLabels name="brush" toName="image">
    <Label value="food"/>
  </BrushLabels>
</View>
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DEVICE` | `cuda` | `cuda` or `cpu` |
| `MODEL_NAME` | `facebook/sam3` | HuggingFace model ID |
| `TEXT_PROMPT` | `food` | Text prompt for segmentation |
| `SCORE_THRESHOLD` | `0.25` | Minimum confidence score |
| `HF_TOKEN` | - | HuggingFace token (required for gated model) |
| `LABEL_STUDIO_URL` | - | Label Studio instance URL |
| `LABEL_STUDIO_API_KEY` | - | Label Studio API key |

## How It Works

1. Tasks already have preannotated masks (done externally)
2. User opens task, sees existing masks
3. User draws rectangle around missed food item
4. Backend runs SAM3 with text="food" + box prompt
5. Returns brush mask for that region
6. User can draw more rectangles for additional items
