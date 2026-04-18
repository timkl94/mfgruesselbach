#!/usr/bin/env python3
"""
Erkennungs-Test-Werkzeug
========================
Lädt das aktuelle Webcam-Bild herunter, erkennt Fahrzeuge und
zeichnet die Erkennungsrahmen direkt ins Bild ein.

Das Ergebnis wird als 'detection_result.jpg' gespeichert und
sofern möglich automatisch im Standard-Bildbetrachter geöffnet.

Benutzung:
    python show_detections.py
"""

import io
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# Gemeinsame Einstellungen aus car_detector importieren
from car_detector import (
    CAR_CONFIDENCE_THRESHOLD,
    IMAGE_URL,
    TIMEZONE,
    VEHICLE_CLASSES,
    YOLO_MODEL,
)

# Farbe pro Fahrzeugklasse (R, G, B)
CLASS_COLORS = {
    "Auto": (255, 50, 50),       # Rot
    "Motorrad": (255, 165, 0),   # Orange
    "Bus": (50, 200, 50),        # Grün
    "LKW": (80, 130, 255),       # Blau
}
DEFAULT_COLOR = (255, 255, 0)    # Gelb für unbekannte Klassen

OUTPUT_FILE = "detection_result.jpg"


def load_font(size: int = 16) -> ImageFont.ImageFont:
    """Versucht eine TrueType-Schrift zu laden; fällt auf Standardschrift zurück."""
    for font_path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ]:
        try:
            return ImageFont.truetype(font_path, size)
        except (OSError, IOError):
            pass
    return ImageFont.load_default()


def draw_detections(image: Image.Image, detections: list[dict]) -> Image.Image:
    """Zeichnet Bounding-Boxes und Beschriftungen in das Bild."""
    result = image.copy()
    draw = ImageDraw.Draw(result)
    font = load_font(16)

    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["bbox"])
        label = f"{det['class']} {det['confidence']:.0%}"
        color = CLASS_COLORS.get(det["class"], DEFAULT_COLOR)

        # Rahmen (3 Pixel Stärke)
        for offset in range(3):
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=color,
            )

        # Beschriftungs-Hintergrund
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        label_y = max(0, y1 - text_h - 6)
        draw.rectangle(
            [x1, label_y, x1 + text_w + 6, label_y + text_h + 6],
            fill=color,
        )

        # Beschriftungstext
        draw.text((x1 + 3, label_y + 3), label, fill=(0, 0, 0), font=font)

    return result


def main() -> None:
    print("Lade YOLOv8-Modell …")
    model = YOLO(YOLO_MODEL)

    print(f"Rufe Bild ab: {IMAGE_URL}")
    try:
        response = requests.get(IMAGE_URL, timeout=15)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
    except requests.RequestException as exc:
        print(f"FEHLER: Bild konnte nicht abgerufen werden: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Erkenne Fahrzeuge …")
    results = model(image, verbose=False)
    detections: list[dict] = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            if class_id in VEHICLE_CLASSES and conf >= CAR_CONFIDENCE_THRESHOLD:
                detections.append(
                    {
                        "class": VEHICLE_CLASSES[class_id],
                        "confidence": conf,
                        "bbox": box.xyxy[0].tolist(),
                    }
                )

    timestamp = datetime.now(tz=TIMEZONE).strftime("%Y-%m-%d %H:%M:%S %Z")

    print()
    print("=" * 60)
    print(f"  Zeitpunkt : {timestamp}")
    print(f"  Erkannt   : {len(detections)} Fahrzeug(e)")
    if detections:
        print("-" * 60)
        for i, det in enumerate(detections, start=1):
            print(f"  {i}. {det['class']}  (Konfidenz: {det['confidence']:.0%})")
    print("=" * 60)
    print()

    annotated = draw_detections(image, detections)
    annotated.save(OUTPUT_FILE, quality=90)
    print(f"Ergebnis gespeichert: {OUTPUT_FILE}")

    # Bild automatisch öffnen (klappt auf Desktop-Systemen)
    try:
        annotated.show(title="Fahrzeugerkennung – Ergebnis")
    except Exception:
        pass  # Kein Fehler, falls kein Bildbetrachter verfügbar


if __name__ == "__main__":
    main()
