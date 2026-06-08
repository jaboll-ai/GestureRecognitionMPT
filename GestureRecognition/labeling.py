import pickle
from pathlib import Path


def data_labeling(times: int, label: str):
    """
    TODO: data_labeling: Datenerfassung für Gesten (SignalHub)

    Ziel:
    -----
    Implementiere eine Funktion, mit der Trainingsdaten für eine bestimmte
    Geste aufgenommen und gespeichert werden können.

    Anforderungen / Ideen:
    ----------------------

    1. Aufnahme starten

       - Starte SignalHub über einen Subprocess mit ``--mode record``
       - Übergib einen Dateipfad für die Aufnahme ``--recorder <path_to_save_recording_at>.pkl``
       - Überlege, welche Module aufgenommen werden sollen
       - Nimm entsprechende Änderungen in der ``config.yaml`` vor

    2. Interaktive Steuerung (optional)

       - Implementiere eine einfache Benutzerinteraktion:
         - Aufnahme speichern
         - Aufnahme verwerfen
         - Programm beenden

    .. tip::

       Die Funktion ``getch()`` (Aus dem Modul Linux :mod:`getch` oder bei Windows :mod:`msvcrt`) ist sehr hilfreich, um einzelne Tastendrücke
       direkt auszulesen (ohne Enter). Damit kannst du dir ein schnelles
       Labeling-Interface bauen.

       Beispiel:

       .. code-block:: text

           ESC → speichern
           andere Taste → verwerfen

    3. Daten sichten und bereinigen

       - Lade die aufgenommenen Daten
       - Überlege:
         - Welche Teile sind relevant?
         - Welche Frames sind leer oder unbrauchbar?
         - Sollten gewisse Sequenzen evtl. gar nicht benutzt werden?
       - Entferne unnötige Anteile (z. B. keine erkannte Hand am Anfang/Ende)

    4. Speicherung

       - Speichere Daten strukturiert nach Labels (z. B. Ordnerstruktur)
       - Jede Aufnahme sollte einzeln gespeichert werden

    .. note::

       Die konkrete Umsetzung (Dateiformat, Struktur, Ablauf) ist bewusst offen.
       Entwickle ein System, das für dich sinnvoll ist und sich gut weiterverarbeiten lässt.

    .. warning::

       Ziel ist nicht nur, dass es „funktioniert“, sondern ein sauberer und
       effizienter Workflow für Datensammlung.

    Parameters
    ----------
    times : int
        Wie viele Aufnahmen gemacht werden sollen.
        Kann frei angepasst werden (z. B. Endlosschleife oder interaktive Steuerung).

    label : str
        Name der Geste / Klasse.
        Kann ebenfalls frei gestaltet werden (z. B. dynamische Labels, mehrere Klassen gleichzeitig).
    """
    pass

def extract_trajectory(pkl_path, finger_idx=8):
    with open(pkl_path, "rb") as f:
        rec = pickle.load(f)

    trajectory = []
    for frame in rec["detector"]:
        result = frame["detector"]
        if not result.hand_landmarks:
            continue
        tip = result.hand_landmarks[0][finger_idx]
        trajectory.append((tip.x, tip.y))
    return trajectory


def dataset_building(output_path):
    """
    TODO: dataset_building: Trainingsdatensatz aus aufgenommenen Gesten erstellen

    Ziel:
    -----
    Implementiere eine Funktion, die alle aufgenommenen Daten lädt,
    verarbeitet und in eine Form bringt, die von eurem
    Hidden-Markov-Modell (HMM) Classifier verwendet werden kann.

    Anforderungen / Ideen:
    ----------------------

    1. Daten laden

       - Durchsuche deinen Trainingsdaten-Ordner
       - Organisiere Daten nach Labels

    2. Feature-Extraktion / Preprocessing

       - Überlege:
         - Welche Features braucht dein Modell?
         - Wie transformierst du die Rohdaten sinnvoll?
       - Wende eine konsistente Verarbeitung auf alle Sequenzen an

    3. Umgang mit Sequenzen

       - Daten sind zeitliche Sequenzen
       - Achte auf:
         - Unterschiedliche Längen
         - Konsistente Struktur

    4. Validierung

       - Entferne unbrauchbare Daten
         (z. B. zu kurze oder fehlerhafte Sequenzen)

    5. Ausgabeformat

       - Baue den Datensatz so, dass dein HMM direkt damit arbeiten kann
       - Das Format sollst du selbst definieren

    .. note::

       Es gibt hier keine vorgegebene „richtige“ Lösung.
       Wichtig ist, dass dein Datensatz konsistent und nutzbar ist.

    .. tip::

       Denke wie ein System-Designer:
       Wie müssen Daten aussehen, damit Training und Inferenz sauber funktionieren?

    .. warning::

       Inkonsistente Datenstrukturen sind eine der häufigsten Fehlerquellen
       beim Training von Sequenzmodellen.

    Erweiterung (optional):
    -----------------------

    - Normalisierung der Daten
    - Datenaugmentation
    - Debug-Ausgaben oder Visualisierung

    Parameters
    ----------
    output_path : Path or str
        Zielpfad für den erzeugten Trainingsdatensatz.
    """
    
    recordings_dir = Path("data/recordings")

    sequences = []
    labels = []

    for pkl_file in sorted(recordings_dir.glob("*/*.pkl")):
        label = pkl_file.parent.name
        trajectory = extract_trajectory(pkl_file)

        if len(trajectory) < 2:
            continue

        sequences.append(trajectory)
        labels.append(label)

    print(f"{len(sequences)} Sequenzen geladen, Klassen: {sorted(set(labels))}")
    return sequences, labels