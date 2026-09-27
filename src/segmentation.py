# import cv2
# import numpy as np
# import torch
# from torchvision.models.detection import (
#     MaskRCNN_ResNet50_FPN_Weights,
#     maskrcnn_resnet50_fpn,
# )

# PERSON_CLASS_ID = 1  # COCO category id for "person"
# SCORE_THRESHOLD = 0.7
# MASK_THRESHOLD = 0.5

# def load_model() -> torch.nn.Module:
#     weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
#     model = maskrcnn_resnet50_fpn(weights=weights)
#     model.eval()
#     return model

# model = load_model()

# def segment_people(image_bgr: np.ndarray, model: torch.nn.Module = model) -> np.ndarray:
#     """Return a single combined binary mask (H, W) covering all detected people.
    
#     Return BGR
#     """
#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0

#     with torch.no_grad():
#         output = model([tensor])[0]

#     h, w = image_bgr.shape[:2]
#     combined_mask = np.zeros((h, w), dtype=np.uint8)

#     for label, score, mask in zip(output["labels"], output["scores"], output["masks"]):
#         if label.item() != PERSON_CLASS_ID or score.item() < SCORE_THRESHOLD:
#             continue
#         person_mask = (mask[0].numpy() > MASK_THRESHOLD).astype(np.uint8) * 255
#         combined_mask = np.maximum(combined_mask, person_mask)

#     mask = combined_mask
#     b, g, r = cv2.split(image_bgr)
#     b_masked = np.where(mask, 1, b)
#     g_masked = np.where(mask, 1, g)
#     r_masked = np.where(mask, 1, r)

#     img = np.zeros((*image_bgr.shape[:2], 3), dtype=np.uint8)

#     img[:, :, 0] = b_masked
#     img[:, :, 1] = g_masked
#     img[:, :, 2] = r_masked

#     return img, mask


# import torch

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = load_model().to(device)

# def preprocess(image_bgr: np.ndarray, max_dim: int = 640):
#     h, w = image_bgr.shape[:2]
#     scale = max_dim / max(h, w)
#     resized = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
#     rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
#     tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device) / 255.0
#     return tensor, scale

# def segment_people_batch(frames, model=model, batch_size=4):
#     frames_masked = []
#     masks = []
#     for i in range(0, len(frames), batch_size):
#         batch = frames[i : i + batch_size]
#         tensors, scales = zip(*(preprocess(f) for f in batch))
#         with torch.inference_mode():
#             outputs = model(list(tensors))  # single forward pass for the whole batch
#         for frame, output, scale in zip(batch, outputs, scales):
#             h, w = frame.shape[:2]
#             combined_mask = np.zeros((h, w), dtype=np.uint8)
#             for label, score, mask in zip(output["labels"], output["scores"], output["masks"]):
#                 if label.item() != PERSON_CLASS_ID or score.item() < SCORE_THRESHOLD:
#                     continue
#                 small_mask = (mask[0].cpu().numpy() > MASK_THRESHOLD).astype(np.uint8) * 255
#                 full_mask = cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_NEAREST)
#                 combined_mask = np.maximum(combined_mask, full_mask)
#             masks.append(~combined_mask)
#             mask = combined_mask
#             b, g, r = cv2.split(frame)
#             b_masked = np.where(mask, 1, b)
#             g_masked = np.where(mask, 1, g)
#             r_masked = np.where(mask, 1, r)

#             img = np.zeros((*frame.shape[:2], 3), dtype=np.uint8)

#             img[:, :, 0] = b_masked
#             img[:, :, 1] = g_masked
#             img[:, :, 2] = r_masked
#             frames_masked.append(img)

#     return np.asarray(frames_masked), np.array(np.asarray(masks) / 255.0, dtype=np.int8) # binary masks














# import cv2
# import numpy as np
# import torch
# from torchvision.models.segmentation import (
#     LRASPP_MobileNet_V3_Large_Weights,
#     lraspp_mobilenet_v3_large,
# )

# PERSON_CLASS_ID = 15  # "person" index in this model's label set

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def load_model():
#     weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
#     model = lraspp_mobilenet_v3_large(weights=weights)
#     model.eval()
#     return model.to(device)

# model = load_model()

# def preprocess(image_bgr: np.ndarray) -> torch.Tensor:
#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
#     return tensor.to(device)

# def segment_people_batch(frames, model=model, batch_size=8) -> np.ndarray:
#     """Return a (T, H, W) uint8 array of binary person masks (255 = person)."""
#     masks = []

#     for i in range(0, len(frames), batch_size):
#         batch = frames[i : i + batch_size]
#         tensors = torch.stack([preprocess(f) for f in batch])  # (B, 3, H, W)

#         with torch.inference_mode():
#             output = model(tensors)["out"]  # (B, num_classes, H, W)

#         class_maps = output.argmax(dim=1)                       # (B, H, W)
#         mask = (class_maps == PERSON_CLASS_ID).cpu().numpy().astype(np.uint8) * 255
#         masks.extend(mask)

#     frames_masked = []
#     for frame, mask in zip(frames, masks):
#         b, g, r = cv2.split(frame)
#         b_masked = np.where(mask, 1, b)
#         g_masked = np.where(mask, 1, g)
#         r_masked = np.where(mask, 1, r)

#         img = np.zeros((*frame.shape[:2], 3), dtype=np.uint8)

#         img[:, :, 0] = b_masked
#         img[:, :, 1] = g_masked
#         img[:, :, 2] = r_masked
#         frames_masked.append(img)

#     return np.asarray(frames_masked), np.array(np.asarray(masks) / 255.0, dtype=np.uint8) # binary masks



import cv2
import numpy as np
import torch
from torchvision.models.segmentation import (
    LRASPP_MobileNet_V3_Large_Weights,
    lraspp_mobilenet_v3_large,
)

PERSON_CLASS_ID = 15  # "person" index in this model's label set

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
    model = lraspp_mobilenet_v3_large(weights=weights)
    model.eval()
    return model.to(device)

model = load_model()

def preprocess(image_bgr: np.ndarray) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    return tensor.to(device)


def segment_people_batch(frames, model=model, batch_size=8, dilate_kernel_size=15) -> np.ndarray:
    """Return frames with people zeroed out, and a (T, H, W) mask where
    1 = background (valid for reconstruction), 0 = person (excluded)."""
    person_masks = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size))

    for i in range(0, len(frames), batch_size):
        batch = frames[i : i + batch_size]
        tensors = torch.stack([preprocess(f) for f in batch])

        with torch.inference_mode():
            output = model(tensors)["out"]

        class_maps = output.argmax(dim=1)
        batch_masks = (class_maps == PERSON_CLASS_ID).cpu().numpy().astype(np.uint8) * 255

        for mask in batch_masks:
            dilated = cv2.dilate(mask, kernel, iterations=1)
            person_masks.append(dilated)

    frames_masked = []
    background_masks = []
    for frame, person_mask in zip(frames, person_masks):
        background_mask = (person_mask == 0).astype(np.uint8)  # 1 = background, 0 = person
        frame_masked = frame * background_mask[..., None]      # zero out person pixels
        frames_masked.append(frame_masked)
        background_masks.append(background_mask.astype(np.int8))

    return np.asarray(frames_masked), np.asarray(background_masks)
