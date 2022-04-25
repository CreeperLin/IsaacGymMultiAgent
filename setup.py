import os
import sys
from setuptools import setup

root_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(root_dir)
import igma
readme = open(os.path.join(root_dir, 'README.md')).read()
version = igma.__version__
requirements = [
    name.rstrip()
    for name in open(os.path.join(root_dir, 'requirements.txt')).readlines()
]
try:
    git_head = open(os.path.join(root_dir, '.git', 'HEAD')).read().split()[1]
    git_version = open(os.path.join(root_dir, '.git', git_head)).read()[:7]
    version += ('+git' + git_version)
except Exception:
    pass

setup(
    name='IsaacGymMultiAgents',
    version=version,
    author='Yunfeng Lin',
    author_email='linyunfeng@sjtu.edu.cn',
    url='https://github.com/CreeperLin/IsaacGymMultiAgents',
    description='Event system with hooks included',
    long_description=readme,
    long_description_content_type="text/markdown",
    install_requires=requirements,
    tests_require=["pytest"],
    py_modules=['igma'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
