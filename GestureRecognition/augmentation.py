"""
Erzeugt künstliche (augmentierte) Trainingsdaten aus den Aufnahmen in ``recordings/``.

Für jede echte Aufnahme wird die normalisierte Fingertrajektorie (wie sie auch
``HMMClassifier._extract_sequences_from_folder`` verwendet) extrahiert und
mehrfach leicht verändert (Rotation, Skalierung, Zeitverzerrung, Rauschen).
Die Ergebnisse werden als eigenständige .pkl-Dateien in einem separaten
Ordner (Standard: ``recordings_augmented/<Label>/``) gespeichert, im selben
Format, das die bestehende Trainingspipeline erwartet.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np


def extract_trajectory(recording):
    """Holt die längste Trajektorie aus einer geladenen Aufnahme (siehe hmmclassifier.py)."""
    if not isinstance(recording, dict) or "preprocessor" not in recording:
        return None

    best_trajectory = None
    best_len = 0

    for frame in recording["preprocessor"]:
        if frame is None:
            continue

        sequence = frame.get("preprocessor")
        if sequence is not None and len(sequence) > best_len:
            best_len = len(sequence)
            best_trajectory = sequence

    return best_trajectory


def augment_trajectory(trajectory, rng):
    """
    Wendet eine zufällige Kombination von Transformationen auf eine
    normalisierte (zentrierte, auf max. Distanz 1 skalierte) 2D-Trajektorie an:

    1. Zeitverzerrung (Geste schneller/langsamer ausgeführt)
    2. Rotation (leichte Winkelabweichung)
    3. Anisotrope Skalierung (Geste etwas breiter/schmaler oder höher/flacher)
    4. Geglättetes Rauschen (leichtes Zittern der Hand)

    Am Ende wird erneut zentriert und normalisiert, damit die Trajektorie
    dieselbe Konvention wie ``Preprocessor.step`` erfüllt.
    """
    traj = np.asarray(trajectory, dtype=np.float32)
    T = len(traj)

    # 1. Zeitverzerrung: Sequenz auf eine zufällig andere Länge resamplen
    time_factor = rng.uniform(0.85, 1.15)
    new_T = max(5, int(round(T * time_factor)))
    orig_idx = np.linspace(0.0, 1.0, T)
    new_idx = np.linspace(0.0, 1.0, new_T)
    traj = np.stack(
        [np.interp(new_idx, orig_idx, traj[:, dim]) for dim in range(traj.shape[1])],
        axis=1,
    )

    # 2. Rotation um einen kleinen zufälligen Winkel
    angle = np.deg2rad(rng.uniform(-15, 15))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    traj = traj @ rotation.T

    # 3. Anisotrope Skalierung (x und y unabhängig leicht stauchen/strecken)
    scale = rng.uniform(0.85, 1.15, size=2).astype(np.float32)
    traj = traj * scale

    # 4. Geglättetes Rauschen, damit die Trajektorie eine plausible,
    #    weiche Handbewegung bleibt (kein Frame-für-Frame-Zittern)
    noise = rng.normal(0, 0.03, size=traj.shape).astype(np.float32)
    kernel_size = max(1, (len(traj) // 10) | 1)
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    smooth_noise = np.stack(
        [np.convolve(noise[:, dim], kernel, mode="same") for dim in range(traj.shape[1])],
        axis=1,
    )
    traj = traj + smooth_noise

    # 5. Erneut zentrieren und normalisieren (wie im Preprocessor)
    center = traj.mean(axis=0)
    traj = traj - center
    max_dist = np.max(np.linalg.norm(traj, axis=1))
    if max_dist > 0:
        traj = traj / max_dist

    return traj.astype(np.float32)


def augment_recording(file_path, output_dir="recordings_augmented", n_per_recording=5, rng=None, verbose=True):
    """
    Augmentiert genau eine Aufnahme, statt den kompletten Label-Ordner neu
    zu verarbeiten - gedacht für den Aufruf direkt nach dem Speichern einer
    einzelnen neuen Aufnahme (z. B. aus labeling.py oder GestureRecorder).

    Das Label wird aus dem übergeordneten Ordnernamen abgeleitet
    (``recordings/<Label>/<Datei>.pkl``).

    Parameters
    ----------
    rng : int, np.random.Generator oder None
        Zufallsquelle. ``None`` (Standard) erzeugt bei jedem Aufruf frische,
        unabhängige Zufallszahlen - passend für einzelne, interaktive Aufrufe.

    Returns
    -------
    int
        Anzahl der erzeugten künstlichen Aufnahmen (0 bei fehlender Trajektorie).
    """
    file_path = Path(file_path)
    output_dir = Path(output_dir)

    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    with open(file_path, "rb") as f:
        recording = pickle.load(f)

    trajectory = extract_trajectory(recording)
    if trajectory is None:
        if verbose:
            print(f"Übersprungen (keine Trajektorie): {file_path}")
        return 0

    out_label_dir = output_dir / file_path.parent.name
    out_label_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    for i in range(n_per_recording):
        synthetic_trajectory = augment_trajectory(trajectory, rng)
        synthetic_recording = {"preprocessor": [{"preprocessor": synthetic_trajectory}]}

        out_path = out_label_dir / f"{file_path.stem}_aug{i + 1}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(synthetic_recording, f)

        created += 1

    if verbose:
        print(f"{created} künstliche Aufnahmen für {file_path.name} erzeugt")
    return created


def augment_dataset(input_dir="recordings", output_dir="recordings_augmented", n_per_recording=5, seed=42, labels=None):
    """
    Liest alle Aufnahmen aus ``input_dir/<Label>/*.pkl``, erzeugt pro Aufnahme
    ``n_per_recording`` künstliche Varianten und speichert sie unter
    ``output_dir/<Label>/``.

    Parameters
    ----------
    labels : str oder list[str], optional
        Falls angegeben, werden nur diese Label-Ordner augmentiert (z. B. nach
        einer einzelnen Labeling-Session), statt des gesamten Datensatzes.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    rng = np.random.default_rng(seed)

    if isinstance(labels, str):
        labels = [labels]

    if not input_dir.exists():
        print(f"Eingabeordner existiert nicht: {input_dir}")
        return 0

    total_created = 0
    total_skipped = 0

    label_dirs = (
        [input_dir / label for label in labels]
        if labels is not None
        else sorted(input_dir.iterdir())
    )

    for label_dir in label_dirs:
        if not label_dir.is_dir():
            continue

        label = label_dir.name
        label_created = 0

        for file_path in sorted(label_dir.glob("*.pkl")):
            if file_path.name.startswith("_temp_"):
                continue

            created = augment_recording(
                file_path, output_dir=output_dir, n_per_recording=n_per_recording, rng=rng, verbose=False
            )

            if created == 0:
                print(f"Übersprungen (keine Trajektorie): {file_path}")
                total_skipped += 1
                continue

            label_created += created

        total_created += label_created
        print(f"{label}: {label_created} künstliche Aufnahmen erzeugt")

    print(f"\nFertig. {total_created} künstliche Aufnahmen unter '{output_dir}' gespeichert.")
    if total_skipped:
        print(f"{total_skipped} Aufnahmen ohne verwertbare Trajektorie übersprungen.")

    return total_created


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Erzeugt künstliche Trainingsdaten aus vorhandenen Aufnahmen."
    )
    parser.add_argument("--input", default="recordings", help="Ordner mit den echten Aufnahmen")
    parser.add_argument("--output", default="recordings_augmented", help="Zielordner für künstliche Aufnahmen")
    parser.add_argument("--n-per-recording", type=int, default=5, help="Anzahl künstlicher Varianten pro Aufnahme")
    parser.add_argument("--seed", type=int, default=42, help="Zufalls-Seed für Reproduzierbarkeit")
    args = parser.parse_args()

    augment_dataset(
        input_dir=args.input,
        output_dir=args.output,
        n_per_recording=args.n_per_recording,
        seed=args.seed,
    )
