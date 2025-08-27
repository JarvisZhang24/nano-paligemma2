import time
import cv2
import re
import numpy as np

################################### Detection ###################################

# Works only with the fine-tuned models like PaliGemma-3b-mix-224
# Prompt: 'Detect <entity>'
# <loc[value]> is the token used to detect objects in the image
# There are 1024 total loc tokens
# Each detection is represented by a bounding box with 4 values (in order): y_min, x_min, y_max, x_max
# To convert x values to coordinate, use the following formula: value * image_width / 1024
# To convert y values to coordinate, use the following formula: value * image_height / 1024


def display_detection(decoded, image_file_path):
    image = cv2.imread(image_file_path)
    
    # 改进的正则表达式，确保标签和边界框的对应关系
    # 匹配完整的检测模式：4个loc标记后跟一个标签
    pattern = r"<loc(\d+)><loc(\d+)><loc(\d+)><loc(\d+)>\s+(\w+)"
    matches = re.findall(pattern, decoded)
    
    if not matches:
        print("No detection results found in the output.")
        return
    
    # 现在matches中的每个元素都包含(y_min, x_min, y_max, x_max, label)
    colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(len(matches))]
    height, width, _ = image.shape
    
    for idx, (y_min_str, x_min_str, y_max_str, x_max_str, label) in enumerate(matches):
        y_min = int(y_min_str) * height // 1024
        x_min = int(x_min_str) * width // 1024
        y_max = int(y_max_str) * height // 1024
        x_max = int(x_max_str) * width // 1024
        
        color = colors[idx]
        
        # the order is always y_min, x_min, y_max, x_max
        overlay = image.copy()
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, -1)
        alpha = 0.5  # opacity
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        # Bounding box outline
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

        # Label text
        cv2.putText(
            image,
            label,
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
    cv2.imshow("image", image)
    cv2.waitKey(0)  # Wait for any key to close the window
    cv2.destroyAllWindows()

    # Save the image
    cv2.imwrite(f"/Users/jarviszhang/CV_Project/PaliGemma Vision Language Model/examples/output_image_{time.time()}.jpg", image)
    print(f"Image saved to examples/output_image_{time.time()}.jpg")

