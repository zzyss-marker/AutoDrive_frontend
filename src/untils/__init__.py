import cv2


def print_available_cameras():
    """
    尝试打开不同的摄像头索引来检查哪些摄像头是可用的，并打印出这些索引。
    """
    index = 0
    while index < 10:
        # 尝试通过当前索引创建VideoCapture对象
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            # 如果无法打开摄像头，假定没有更多的摄像头可用，退出循环
            cap.release()
            break
        print(f"Camera index {index} is available")
        cap.release()
        index += 1

    if index == 0:
        print("No available cameras found.")
    else:
        print(f"Total available cameras: {index}")
