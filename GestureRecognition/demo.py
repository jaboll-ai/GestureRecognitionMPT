from SignalHub import Engine, ConfigParser, Webcam
from GestureRecognition.modules import *
import argparse

def run(parser: argparse.ArgumentParser):
    parser.add_argument("--mode", action="store", default="none")
    parser.add_argument("--recorder.file", action="store")
    parser.add_argument("--engine.singlestep", action="store_true", default=False)
    parser.add_argument("--webcam.width", required=False)
    parser.add_argument(
        "--trigger", action="store_true", default=False,
        help="Aufnahme/Erkennung per Taste 'a' steuern: 1. Druck = Start, 2. Druck = Stop",
    )
    parser.add_argument(
        "--label", action="store", default=None,
        help="Nur mit --trigger: speichert jede Aufnahme als neue Trainingsaufnahme für dieses Label, statt vorherzusagen",
    )
    parser.add_argument(
        "--kuerzel", action="store", default=None,
        help="Nur mit --label: Kürzel der aufnehmenden Person für den Dateinamen",
    )

    config_parser = ConfigParser(parser)

    # Frühes Parsen nur um zu entscheiden, welche Module gebraucht werden
    args, _ = parser.parse_known_args()

    modules = [
        config_parser,
        Webcam(),
        HandDetector(),
        TrailMarker(),
    ]

    if args.trigger:
        modules.append(GestureTrigger())

    modules.append(Preprocessor())

    if args.trigger and args.label:
        modules.append(GestureRecorder(label=args.label, kuerzel=args.kuerzel))
    else:
        modules.append(HMMModule())

    engine = Engine(modules=modules, signals={})
    signals = engine.run({})

if __name__ == "__main__":
    parser = argparse.ArgumentParser("GestureRecognition")
    run(parser)