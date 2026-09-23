from setuptools import find_namespace_packages, setup

setup(
    name="colab-shim",
    version="0.1.0",
    description="google.colab의 로컬 대체 shim (cv2_imshow 등)",
    packages=find_namespace_packages(include=["google.*"]),
    install_requires=["opencv-python", "matplotlib"],
)
