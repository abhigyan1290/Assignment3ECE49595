import numpy as np

# prompt templates for ensembling, {} gets the item name.
# idea from the CLIP paper's imagenet prompt ensemble, trimmed down for lost & found photos
PROMPT_TEMPLATES = (
    "a photo of a {}.",
    "a photo of a lost {}.",
    "a close-up photo of a {}.",
    "a {} left on a table.",
    "a {} on the floor.",
    "a photo of someone's {}.",
)


def l2_normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def _as_tensor(features):
    # newer transformers can hand back an output object instead of a tensor
    import torch
    if isinstance(features, torch.Tensor):
        return features
    if getattr(features, "image_embeds", None) is not None:
        return features.image_embeds
    if getattr(features, "text_embeds", None) is not None:
        return features.text_embeds
    return features.pooler_output


class ClipEmbedder:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()

    def encode_images(self, images, batch_size=16):
        import torch
        out = []
        with torch.inference_mode():
            for i in range(0, len(images), batch_size):
                pix = self.processor(images=images[i:i + batch_size], return_tensors="pt")["pixel_values"]
                feats = _as_tensor(self.model.get_image_features(pixel_values=pix.to(self.device)))
                out.append(feats.float().cpu().numpy())
        return l2_normalize(np.concatenate(out))

    def encode_texts(self, texts, batch_size=64):
        import torch
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                tok = self.processor(text=list(texts[i:i + batch_size]), return_tensors="pt",
                                     padding=True, truncation=True)
                tok = {k: v.to(self.device) for k, v in tok.items()}
                feats = _as_tensor(self.model.get_text_features(**tok))
                out.append(feats.float().cpu().numpy())
        return l2_normalize(np.concatenate(out))


def encode_query(embedder, query, ensemble=False):
    """Single normalized text vector. ensemble=True averages the query over PROMPT_TEMPLATES."""
    query = query.strip()
    if not query:
        raise ValueError("empty query")
    if not ensemble:
        return embedder.encode_texts([query])[0]
    vecs = embedder.encode_texts([t.format(query) for t in PROMPT_TEMPLATES])
    return l2_normalize(vecs.mean(axis=0))
