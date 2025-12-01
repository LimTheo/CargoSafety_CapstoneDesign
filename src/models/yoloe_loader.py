import os
from ultralytics import YOLOE

# YOLOE 모델 경로
MODEL_PATH = "/home/pi/Desktop/CargoSafety_CapstoneDesign/yoloe-v8s-seg.pt"

names = [
    "cardboard_box_front", "cardboard_box_diagonal", "cardboard_box_tilted",
    "cardboard_box_heavily_tilted", "cardboard_box_stacked", "cardboard_box_collapsed",
    "cardboard_box_damaged",
    "plastic_container_front", "plastic_container_tilted", "plastic_container_stacked",
    "plastic_container_damaged",
    "metal_case_front", "metal_case_tilted", "metal_case_damaged",
    "wooden_crate_front", "wooden_crate_tilted", "wooden_crate_damaged",
    "stacked_boxes", "leaning_box", "displaced_box", "collapsed_box",
    "wrapped_cargo", "open_cargo", "pallet_wrapped", "pallet_open",
    "other_cargo"
]

# 모델 로드 함수
def load_yoloe_model():
    print(f"Loading YOLOE Model: {MODEL_PATH}")

    model = YOLOE(MODEL_PATH)
    model.set_classes(names, model.get_text_pe(names))

    print("YOLOE model loaded.")
    return model