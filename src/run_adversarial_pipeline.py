import subprocess
import sys

print(f"{len(sys.argv[1:])} Arguments >> {sys.argv}")

python_files = [
    'train_classifiers.py',
    'train_autoencoders.py',
    'generate_adversarial_examples.py',
    'detect_adversarial_examples.py',
    'analyze_adversarial_results.py',
    ]

for python_file in python_files:
    if len(sys.argv) > 1:
        print(f"Running {python_file} with args: ")
        subprocess.run(['python', python_file] + sys.argv[1:])
    else:
        print(f"Running {python_file} without args: ")
        subprocess.run(['python', python_file])