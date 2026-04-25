#!/usr/bin/env python3
"""
Webcam-Fahrzeugerkennung
========================
Ruft alle 15 Minuten das Bild der Webcam ab und benachrichtigt,
wenn ein Fahrzeug erkannt wird.

Startet außerdem automatisch den Bild-Viewer-Server (aus server.py)
im Hintergrund, sodass die Ergebnisse unter http://localhost:8080
eingesehen werden können.

Benutzung:
    python car_detector.py

Erweiterungen:
    Für E-Mail- oder App-Benachrichtigungen die Funktion `notify()`
    am Ende dieser Datei anpassen.
    Den Server-Port kann über die Konstante SERVER_PORT angepasst werden.
"""

import glob
import io
import json
import logging
import os
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
import schedule
from PIL import Image, ImageDraw
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

# Hinweis: Die URL verwendet HTTP (kein HTTPS), da dies die vom Betreiber
# bereitgestellte Adresse ist. Falls der Server HTTPS unterstützt, bitte
# das Protokoll entsprechend anpassen.
IMAGE_URL = "http://raar.myds.me/webcam/rb/rb.jpg"
CHECK_INTERVAL_MINUTES = 15
CAR_CONFIDENCE_THRESHOLD = 0.3  # Mindest-Konfidenz (0–1) für eine Erkennung

# YOLO-Modellvariante: yolov8s.pt (small) bietet deutlich bessere Erkennung
# kleiner/weit entfernter Fahrzeuge als das nano-Modell;
# für noch höhere Genauigkeit: yolov8m.pt, yolov8l.pt oder yolov8x.pt
YOLO_MODEL = "yolov8s.pt"

# Zeitzone für Benachrichtigungen
TIMEZONE = ZoneInfo("Europe/Berlin")

# ntfy-Kanal für Push-Benachrichtigungen ans iPhone
NTFY_TOPIC_URL = "https://ntfy.sh/mfgRuesselbachAktuellePiloten"

# Fahrzeug-Klassen aus dem COCO-Datensatz (wird von YOLOv8 verwendet)
VEHICLE_CLASSES = {
    2: "Auto",
    3: "Motorrad",
    5: "Bus",
    7: "LKW",
}

# Nachtruhe: Keine Bildabrufe zwischen 22 Uhr und 6 Uhr (Berliner Ortszeit,
# DST-bewusst dank ZoneInfo – gilt also sowohl für Sommer- als auch Winterzeit)
QUIET_HOURS_START = 22   # ab 22:00 Uhr keine Abrufe
QUIET_HOURS_END = 6      # bis 06:00 Uhr keine Abrufe

# Port für den integrierten Bild-Viewer-Server (http://localhost:PORT)
SERVER_PORT = 8080

# Ordner für gespeicherte Bilder (absolut, relativ zum Skriptverzeichnis)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTION_FOLDER = os.path.join(_SCRIPT_DIR, "Fahrzeug erkannt")
EMPTY_FOLDER     = os.path.join(_SCRIPT_DIR, "Kein Fahrzeug")
MAX_DETECTION_IMAGES = 20               # Maximale Anzahl gespeicherter Erkennungsbilder
MAX_EMPTY_IMAGES = 30                   # Maximale Anzahl gespeicherter Leerbilder

# Farben der Rahmen pro Fahrzeugklasse (R, G, B).
# Hinweis: Diese Werte müssen mit den CSS-Variablen --c-auto, --c-motorrad,
# --c-bus und --c-lkw in viewer.html übereinstimmen.
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "Auto":     (255,  50,  50),   # Rot
    "Motorrad": (255, 165,   0),   # Orange
    "Bus":      ( 50, 200,  50),   # Grün
    "LKW":      ( 80, 130, 255),   # Blau
}
DEFAULT_BBOX_COLOR = (255, 255,   0)   # Gelb für unbekannte Klassen
BBOX_WIDTH = 3              # Linienbreite in Pixeln
BBOX_TEXT_OFFSET = 14       # Pixel-Abstand des Beschriftungstexts oberhalb der Box

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
    results = model(image, verbose=False, imgsz=1280)
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


def _prune_folder(folder: str, max_files: int) -> None:
    """Löscht die ältesten Bilder (und zugehörige JSON-Metadaten), wenn die Ordner-Grenze überschritten wird."""
    files = sorted(
        glob.glob(os.path.join(folder, "*.jpg")),
        key=os.path.getmtime,
    )
    while len(files) > max_files:
        jpg_path = files.pop(0)
        os.remove(jpg_path)
        json_path = jpg_path[:-4] + ".json"
        if os.path.exists(json_path):
            os.remove(json_path)


def _save_metadata_json(
    folder: str,
    timestamp: str,
    detections: list[dict],
    notification_title: str | None,
    notification_message: str | None,
) -> None:
    """Speichert eine JSON-Metadatendatei neben dem Bild."""
    dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").replace(tzinfo=TIMEZONE)
    meta = {
        "timestamp": timestamp,
        "datetime_display": dt.strftime("%d.%m.%Y %H:%M:%S %Z"),
        "datetime_iso": dt.isoformat(),
        "has_vehicles": bool(detections),
        "vehicle_count": len(detections),
        "detections": detections,
        "notification_title": notification_title,
        "notification_message": notification_message,
    }
    json_path = os.path.join(folder, f"{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    logger.info("Metadaten gespeichert: %s", json_path)


def save_detection_image(
    image: Image.Image,
    detections: list[dict],
    notification_title: str | None = None,
    notification_message: str | None = None,
) -> None:
    """
    Speichert das Bild mit farbigen Bounding-Boxen in ``DETECTION_FOLDER``.
    Hält maximal ``MAX_DETECTION_IMAGES`` Bilder vor.
    Neben dem Bild wird eine JSON-Metadaten-Datei gespeichert.
    """
    os.makedirs(DETECTION_FOLDER, exist_ok=True)

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = CLASS_COLORS.get(det["class"], DEFAULT_BBOX_COLOR)
        draw.rectangle(
            [x1, y1, x2, y2],
            outline=color,
            width=BBOX_WIDTH,
        )
        draw.text(
            (x1, max(y1 - BBOX_TEXT_OFFSET, 0)),
            f"{det['class']} {det['confidence']:.0%}",
            fill=color,
        )

    timestamp = datetime.now(tz=TIMEZONE).strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(DETECTION_FOLDER, f"{timestamp}.jpg")
    annotated.save(filepath, format="JPEG")
    logger.info("Erkennungsbild gespeichert: %s", filepath)

    _save_metadata_json(
        DETECTION_FOLDER, timestamp, detections,
        notification_title, notification_message,
    )

    _prune_folder(DETECTION_FOLDER, MAX_DETECTION_IMAGES)


def save_empty_image(
    image: Image.Image,
    notification_title: str | None = None,
    notification_message: str | None = None,
) -> None:
    """
    Speichert das Bild ohne Erkennungen in ``EMPTY_FOLDER``.
    Hält maximal ``MAX_EMPTY_IMAGES`` Bilder vor.
    Neben dem Bild wird eine JSON-Metadaten-Datei gespeichert.
    """
    os.makedirs(EMPTY_FOLDER, exist_ok=True)

    timestamp = datetime.now(tz=TIMEZONE).strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(EMPTY_FOLDER, f"{timestamp}.jpg")
    image.save(filepath, format="JPEG")
    logger.info("Leerbild gespeichert: %s", filepath)

    _save_metadata_json(
        EMPTY_FOLDER, timestamp, [],
        notification_title, notification_message,
    )

    _prune_folder(EMPTY_FOLDER, MAX_EMPTY_IMAGES)


def send_startup_notification() -> None:
    """Sendet eine Push-Benachrichtigung beim Skriptstart via ntfy."""
    try:
        requests.post(
            NTFY_TOPIC_URL,
            data="Webcam-Fahrzeugerkennung gestartet.".encode("utf-8"),
            headers={
                "Title": "MFG Rüsselbach - Skript gestartet",
                "Priority": "default",
                "Tags": "white_check_mark",
            },
            timeout=10,
        )
        logger.info("Start-Benachrichtigung via ntfy gesendet.")
    except requests.RequestException as exc:
        logger.warning("Start-Benachrichtigung fehlgeschlagen: %s", exc)


def notify(detections: list[dict]) -> None:
    """
    Benachrichtigung ausgeben, wenn Fahrzeuge erkannt wurden.

    Wird nur aufgerufen, wenn sich die Fahrzeuganzahl geändert hat.
    Hier kann später eine E-Mail- oder App-Benachrichtigung eingebunden
    werden (z. B. über smtplib, Pushover, Telegram …).
    """
    count = len(detections)
    timestamp = datetime.now(tz=TIMEZONE).strftime("%Y-%m-%d %H:%M:%S %Z")

    if count == 0:
        lines = [
            "",
            "=" * 60,
            f"  TESTMITTEILUNG – {timestamp}",
            "  Kein Fahrzeug mehr im Webcam-Bild erkennbar.",
            "=" * 60,
            "",
        ]
        print("\n".join(lines))
        logger.info("Benachrichtigung: Kein Fahrzeug mehr erkannt.")

        ntfy_message = "Kein Fahrzeug mehr erkannt"
        ntfy_title = "MFG Rüsselbach - Kein Fahrzeug"
        ntfy_tags = "white_check_mark"
    else:
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

        ntfy_message = f"{count} Fahrzeug(e) erkannt"
        ntfy_title = f"MFG Rüsselbach - {count} Fahrzeug(e) erkannt"
        ntfy_tags = "car"

    # Push-Benachrichtigung via ntfy
    try:
        requests.post(
            NTFY_TOPIC_URL,
            data=ntfy_message.encode("utf-8"),
            headers={
                "Title": ntfy_title,
                "Priority": "default",
                "Tags": ntfy_tags,
            },
            timeout=10,
        )
        logger.info("Push-Benachrichtigung via ntfy gesendet.")
    except requests.RequestException as exc:
        logger.warning("ntfy-Benachrichtigung fehlgeschlagen: %s", exc)


# ---------------------------------------------------------------------------
# Haupt-Prüfschleife
# ---------------------------------------------------------------------------

# Zuletzt gemeldete Fahrzeuganzahl; -1 = noch keine Meldung gesendet
_last_notified_count: int = -1


def is_quiet_hours() -> bool:
    """Gibt True zurück, wenn die aktuelle Berliner Ortszeit in der Nachtruhe liegt.

    Die Nachtruhe dauert von QUIET_HOURS_START (22 Uhr) bis QUIET_HOURS_END (6 Uhr).
    ZoneInfo("Europe/Berlin") berücksichtigt automatisch Sommer- und Winterzeit,
    sodass die Uhrzeit immer der lokalen Berliner Zeit entspricht.
    """
    hour = datetime.now(tz=TIMEZONE).hour
    return hour >= QUIET_HOURS_START or hour < QUIET_HOURS_END


def check_for_vehicles(model: YOLO) -> None:
    """Bild abrufen, analysieren und ggf. benachrichtigen."""
    global _last_notified_count

    if is_quiet_hours():
        logger.info(
            "Nachtruhe (%d–%d Uhr Berliner Zeit) – kein Bildabruf.",
            QUIET_HOURS_START,
            QUIET_HOURS_END,
        )
        return

    logger.info("Überprüfe Webcam-Bild (%s) …", IMAGE_URL)

    try:
        image = fetch_image(IMAGE_URL)
    except requests.RequestException as exc:
        logger.error("Bild konnte nicht abgerufen werden: %s", exc)
        return

    detections = detect_vehicles(image, model)
    count = len(detections)

    # Benachrichtigungsinhalt vorab bestimmen, damit er in den Metadaten landet
    notif_title: str | None = None
    notif_message: str | None = None
    if count != _last_notified_count:
        if count > 0:
            notif_title = f"MFG Rüsselbach - {count} Fahrzeug(e) erkannt"
            notif_message = f"{count} Fahrzeug(e) erkannt"
        elif _last_notified_count > 0:
            notif_title = "MFG Rüsselbach - Kein Fahrzeug"
            notif_message = "Kein Fahrzeug mehr erkannt"

    # Bilder immer speichern
    if detections:
        save_detection_image(image, detections, notif_title, notif_message)
    else:
        save_empty_image(image, notif_title, notif_message)

    # Benachrichtigung nur senden, wenn sich die Anzahl geändert hat;
    # beim ersten Durchlauf (_last_notified_count == -1) nur bei Fahrzeugen.
    if count != _last_notified_count:
        if count > 0:
            notify(detections)
            _last_notified_count = count
        elif _last_notified_count > 0:
            # Anzahl ist auf 0 gesunken → "kein Fahrzeug mehr"-Meldung
            notify([])
            _last_notified_count = 0
        else:
            # _last_notified_count war -1 und count ist 0: keine Meldung
            _last_notified_count = 0
            logger.info("Keine Fahrzeuge erkannt.")
    else:
        logger.info(
            "Fahrzeuganzahl unverändert (%d) – keine neue Benachrichtigung.", count
        )


# ---------------------------------------------------------------------------
# Integrierter Viewer-Server
# ---------------------------------------------------------------------------


def _start_viewer_server() -> None:
    """Startet den HTTP-Viewer-Server aus server.py in einem Hintergrund-Thread."""
    from http.server import HTTPServer
    from server import Handler  # Handler aus server.py importieren

    server = HTTPServer(("0.0.0.0", SERVER_PORT), Handler)
    logger.info(
        "Viewer-Server gestartet: http://localhost:%d", SERVER_PORT
    )
    server.serve_forever()


# ---------------------------------------------------------------------------
# Einstiegspunkt
# ---------------------------------------------------------------------------


def main() -> None:
    logger.info("Starte Webcam-Fahrzeugerkennung …")
    logger.info("Lade YOLOv8-Modell (%s) …", YOLO_MODEL)
    model = YOLO(YOLO_MODEL)

    # Viewer-Server im Hintergrund starten
    server_thread = threading.Thread(target=_start_viewer_server, daemon=True)
    server_thread.start()

    # Push-Benachrichtigung beim Start senden
    send_startup_notification()

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
