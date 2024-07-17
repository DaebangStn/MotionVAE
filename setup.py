from setuptools import find_packages, setup

setup(
    name="MotionVAE",
    version="0.1",
    description="Character Controllers using Motion VAEs / "
                "fixed to implement ACE (Adversarial Correspondence Embedding)",
    author="original: Hung Yu Ling, Fabio Zinno / modifier: Geonho Leem",
    author_email="geonholeem@imo.snu.ac.kr",
    install_requires=[
        "torch==2.3.1",
        "tensorboard",
        "gitpython",
        "numpy==1.23.1",
        "imageio",
        "matplotlib",
        "tqdm",
        "gym",
        "pybullet",
    ],
    packages=find_packages(include=["mvae*"], exclude=["res", "scripts", ]),
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
