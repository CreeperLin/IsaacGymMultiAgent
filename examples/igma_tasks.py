import importlib
import traceback

task_paths = [
    'igma.tasks.joust',
    'igma.tasks.legged_gym',
    'igma.tasks.legged',
]

for path in task_paths:
    try:
        importlib.import_module(path)
    except ImportError:
        traceback.print_exc()
