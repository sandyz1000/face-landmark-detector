from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

requirements = open("requirements.txt", 'r').readlines()

setup(
    name="face-landmark-detector",
    entry_points={
        'console_script': ["face_landmark_detect=scripts.cli:main", "face_landmark_prepare=scripts.prepare_dataset"],
    },
    version="0.1.0",
    description="Face keypoints detector",
    long_description=long_description,
    license='GNU License',
    long_description_content_type="text/markdown",
    author="Sandip Dey",
    author_email="sandip.dey1988@yahoo.com",
    packages=['src'],
    include_package_data=True,
    install_requires=requirements,
    platforms=["linux", "unix"],
    python_requires=">3.5.2",
)
