import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from models.sonic_model_specs import SONIC_MODEL_SPECS, print_model_spec


def main():
    for name in ["M1", "M2", "C1", "C2"]:
        spec = SONIC_MODEL_SPECS[name]
        print_model_spec(spec)


if __name__ == "__main__":
    main()