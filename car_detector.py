#!/usr/bin/env python3
"""
Webcam-Fahrzeugerkennung
========================
Ruft alle 15 Minuten das Bild der Webcam ab und benachrichtigt,
wenn ein Fahrzeug erkannt wird.

Benutzung:
    python car_detector.py

Erweiterungen:
    Für E-Mail- oder App-Benachrichtigungen die Funktion `notify()`
    am Ende dieser Datei anpassen.
"""

import io
import logging
import time
from datetime import datetime

import requests
import schedule
from PIL import Image
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

IMAGE_URL = "http://raar.myds.me/webcam/rb/rb.jpg"
CHECK_INTERVAL_MINUTES = 15
CAR_CONFIDENCE_THRESHOLD = 0.5  # Mindest-Konfidenz (0–1) für eine Erkennung

# Fahrzeug-Klassen aus dem COCO-Datensatz (wird von YOLOv8 verwendet)
VEHICLE_CLASSES = {
    2: "Auto",
    3: "Motorrad",
    5: "Bus",
    7: "LKW",
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def fetch_image(url: str) -> Image.Image:
    """Lädt das Bild von der angegebenen URL herunter."""
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def detect_vehicles(
    image: Image.Image,
    model: YOLO,
    confidence: float = CAR_CONFIDENCE_THRESHOLD,
) -> list[dict]:
    """
    Erkennt Fahrzeuge im Bild mit YOLOv8.

    Gibt eine Liste von Erkennungen zurück, jeweils mit:
    - ``class``:      Fahrzeugtyp (z. B. "Auto")
    - ``confidence``: Erkennungs-Konfidenz (0–1)
    - ``bbox``:       Bounding Box [x1, y1, x2, y2]
    """
    results = model(image, verbose=False)
    detections: list[dict] = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])

            if class_id in VEHICLE_CLASSES and conf >= confidence:
                detections.append(
                    {
                        "class": VEHICLE_CLASSES[class_id],
                        "confidence": conf,
                        "bbox": box.xyxy[0].tolist(),
                    }
                )

    return detections


def notify(detections: list[dict]) -> None:
    """
    Benachrichtigung ausgeben, wenn Fahrzeuge erkannt wurden.

    Hier kann später eine E-Mail- oder App-Benachrichtigung eingebunden
    werden (z. B. über smtplib, Pushover, Telegram …).
    """
    count = len(detections)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "",
        "=" * 60,
        f"  TESTMITTEILUNG – {timestamp}",
        f"  {count} Fahrzeug(e) im Webcam-Bild erkannt!",
        "-" * 60,
    ]
    for i, det in enumerate(detections, start=1):
        lines.append(
            f"  {i}. {det['class']}  "
            f"(Konfidenz: {det['confidence']:.0%})"
        )
    lines.append("=" * 60)
    lines.append("")

    print("\n".join(lines))
    logger.info("Benachrichtigung: %d Fahrzeug(e) erkannt.", count)


# ---------------------------------------------------------------------------
# Haupt-Prüfschleife
# ---------------------------------------------------------------------------


def check_for_vehicles(model: YOLO) -> None:
    """Bild abrufen, analysieren und ggf. benachrichtigen."""
    logger.info("Überprüfe Webcam-Bild (%s) …", IMAGE_URL)

    try:
        image = fetch_image(IMAGE_URL)
    except requests.RequestException as exc:
        logger.error("Bild konnte nicht abgerufen werden: %s", exc)
        return

    detections = detect_vehicles(image, model)

    if detections:
        notify(detections)
    else:
        logger.info("Keine Fahrzeuge erkannt.")


# ---------------------------------------------------------------------------
# Einstiegspunkt
# ---------------------------------------------------------------------------


def main() -> None:
    logger.info("Starte Webcam-Fahrzeugerkennung …")
    logger.info("Lade YOLOv8-Modell (yolov8n.pt) …")
    model = YOLO("yolov8n.pt")

    # Sofortige erste Prüfung beim Start
    check_for_vehicles(model)

    # Regelmäßige Prüfung alle 15 Minuten
    schedule.every(CHECK_INTERVAL_MINUTES).minutes.do(
        check_for_vehicles, model=model
    )
    logger.info(
        "Nächste Prüfung in %d Minuten. Drücke Strg+C zum Beenden.",
        CHECK_INTERVAL_MINUTES,
    )

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Programm beendet.")


if __name__ == "__main__":
    main()
