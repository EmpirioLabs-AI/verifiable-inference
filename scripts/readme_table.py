"""Write the table of fingerprints into README.md, or check that it is current.

    python scripts/readme_table.py          write it
    python scripts/readme_table.py --check  exit 1 when README.md does not match fingerprints/

The table sits between the two marker comments in README.md and is derived from the fingerprint files only, so a
fingerprint added, changed or removed shows in the README in the same commit or CI fails.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(ROOT, "README.md")
FINGERPRINTS = os.path.join(ROOT, "fingerprints")
START = "<!-- fingerprints:start -->"
END = "<!-- fingerprints:end -->"


def table():
    lines = ["| Model ID | Public weights | Revision | Made |", "| --- | --- | --- | --- |"]
    for name in sorted(os.listdir(FINGERPRINTS)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(FINGERPRINTS, name), encoding="utf-8") as handle:
            data = json.load(handle)
        weights = data["weights"]
        lines.append("| [`%s`](fingerprints/%s) | [%s](https://huggingface.co/%s/tree/%s) | `%s` | %s |" % (
            data["model"], name, weights, weights, data["revision"], data["revision"][:12], data["made_at"][:10]))
    return "\n".join(lines)


def render(text):
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    return head + START + "\n" + table() + "\n" + END + tail


def main(argv):
    with open(README, encoding="utf-8") as handle:
        current = handle.read()
    wanted = render(current)
    if "--check" in argv:
        if wanted != current:
            print("README.md does not list the fingerprints in fingerprints/: run python scripts/readme_table.py")
            return 1
        print("README.md lists every fingerprint")
        return 0
    with open(README, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(wanted)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
