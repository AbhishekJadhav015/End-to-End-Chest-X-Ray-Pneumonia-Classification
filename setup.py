from setuptools import find_packages, setup

setup(
    name="pneumonia-classifier",
    version="0.0.1",
    author="Abhishek Jadav",
    author_email="abhishekjadhav0015@gmail.com",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "pillow",
        "tqdm",
    ],
)