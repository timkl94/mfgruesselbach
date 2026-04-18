# mfgruesselbach – Webcam-Fahrzeugerkennung

Dieses Projekt überwacht automatisch eine Webcam und erkennt Fahrzeuge mithilfe von **YOLOv8** (künstliche Intelligenz). Bei einer Erkennung wird eine Benachrichtigung ausgegeben. Die Uhrzeiten werden immer in **deutscher Zeit** (Europe/Berlin, inkl. Sommer-/Winterzeit) angezeigt.

---

## Voraussetzungen

- **Python 3.9 oder neuer** – [python.org/downloads](https://www.python.org/downloads/)
- Internetverbindung (für den ersten Download des KI-Modells und den Webcam-Abruf)

---

## Installation – Schritt für Schritt

### Schritt 1 – Repository herunterladen

```bash
git clone https://github.com/timkl94/mfgruesselbach.git
cd mfgruesselbach
```

### Schritt 2 – Virtuelle Umgebung erstellen (empfohlen)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Schritt 3 – Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

> Beim ersten Start wird das YOLOv8-Modell (~6 MB) automatisch heruntergeladen.

---

## Programme

### `car_detector.py` – Dauerüberwachung

Läuft im Hintergrund und prüft alle **15 Minuten** das Webcam-Bild.  
Bei erkannten Fahrzeugen erscheint eine Testmitteilung in der Konsole.

```bash
python car_detector.py
```

**Beispielausgabe:**

```
============================================================
  TESTMITTEILUNG – 2026-04-18 20:15:00 CEST
  1 Fahrzeug(e) im Webcam-Bild erkannt!
------------------------------------------------------------
  1. Auto  (Konfidenz: 87%)
============================================================
```

Mit **Strg + C** wird das Programm beendet.

---

### `show_detections.py` – Testmodus mit Bild-Visualisierung

Lädt das aktuelle Webcam-Bild **einmalig** herunter, erkennt Fahrzeuge und  
zeichnet farbige Rahmen (Bounding-Boxes) direkt ins Bild ein.

```bash
python show_detections.py
```

Das Ergebnis wird als **`detection_result.jpg`** im gleichen Ordner gespeichert  
und – sofern ein Bildbetrachter verfügbar ist – automatisch geöffnet.

**Farbcode der Rahmen:**

| Farbe  | Fahrzeugtyp |
|--------|-------------|
| 🔴 Rot    | Auto        |
| 🟠 Orange | Motorrad    |
| 🟢 Grün   | Bus         |
| 🔵 Blau   | LKW         |

---

## Konfiguration

In `car_detector.py` können folgende Werte oben in der Datei angepasst werden:

| Einstellung               | Standard          | Beschreibung                                     |
|---------------------------|-------------------|--------------------------------------------------|
| `IMAGE_URL`               | `http://raar.…`   | URL des Webcam-Bildes                            |
| `CHECK_INTERVAL_MINUTES`  | `15`              | Prüfintervall in Minuten                         |
| `CAR_CONFIDENCE_THRESHOLD`| `0.5`             | Mindest-Konfidenz (0–1) für eine Erkennung       |
| `YOLO_MODEL`              | `yolov8n.pt`      | KI-Modell (`n`=schnell, `s/m/l/x`=genauer)      |

---

## Erweiterungen

Für **E-Mail- oder App-Benachrichtigungen** die Funktion `notify()` in `car_detector.py` anpassen –  
z. B. mit `smtplib` (E-Mail), Pushover oder Telegram.
