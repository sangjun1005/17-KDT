import cv2
import matplotlib.pyplot as plt


def cv2_imshow(image):
    """Colab의 cv2_imshow와 동일하게 동작하는 로컬(Jupyter) 대체 함수."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.axis("off")
    plt.show()
