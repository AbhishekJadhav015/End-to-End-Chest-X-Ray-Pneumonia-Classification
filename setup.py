from typing import List
from setuptools import find_packages, setup

def get_requirements(file_path: str) -> List[str]: 
    '''
    This function will return list of requirements
    '''
    HYPEN_E_DOT = "-e ."
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        # Clean up newlines and spaces
        requirements = [req.strip() for req in requirements]
        
        # Remove the -e . flag
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
            
        # THE FIX: Ignore any pip flags like --extra-index-url
        requirements = [req for req in requirements if not req.startswith("-")]
        
    return requirements

setup(
    name="pneumonia-classifier",
    version="0.0.1",
    author="Your Name",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)