import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.utils import custom_object_scope

# ================== COMPATIBILITY FIX ==================
class CustomDepthwiseConv2D(DepthwiseConv2D):
    @classmethod
    def from_config(cls, config):
        config.pop('groups', None)  # Remove problematic parameter
        return super().from_config(config)

# ================== INITIALIZATION ==================
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
labels = ["Hello", "I love you", "No", "Okay", "Please", "Thank you", "Yes"]

# ================== MODEL LOADING WITH FIX ==================
# model path
model_path = "C:/Users/272749/Downloads/Sign-Language-detection-main (2)/Sign-Language-detection-main/Model/keras_model.h5"

with custom_object_scope({'DepthwiseConv2D': CustomDepthwiseConv2D}):
    classifier = Classifier(model_path, labels_path)

# ================== CONFIGURATION ==================
offset = 20
imgSize = 300
min_confidence = 0.8

# ================== MAIN LOOP ==================
while True:
    try:
        success, img = cap.read()
        if not success or img is None:
            continue

        img_output = img.copy()
        hands, _ = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Ensure valid crop dimensions
            x1 = max(x - offset, 0)
            y1 = max(y - offset, 0)
            x2 = min(x + w + offset, img.shape[1])
            y2 = min(y + h + offset, img.shape[0])
            
            img_crop = img[y1:y2, x1:x2]
            if img_crop.size == 0:
                continue

            # Create white background
            img_white = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            aspect_ratio = h / w

            try:
                # Resize and center the hand image
                if aspect_ratio > 1:
                    k = imgSize / h
                    w_cal = math.ceil(k * w)
                    img_resize = cv2.resize(img_crop, (w_cal, imgSize))
                    w_gap = math.ceil((imgSize - w_cal) / 2)
                    img_white[:, w_gap:w_cal + w_gap] = img_resize
                else:
                    k = imgSize / w
                    h_cal = math.ceil(k * h)
                    img_resize = cv2.resize(img_crop, (imgSize, h_cal))
                    h_gap = math.ceil((imgSize - h_cal) / 2)
                    img_white[h_gap:h_cal + h_gap, :] = img_resize

                # Get prediction
                prediction, index = classifier.getPrediction(img_white)
                
                # Only show predictions with sufficient confidence
                if prediction[index] > min_confidence:
                    # Dynamic text positioning
                    label = labels[index]
                    (text_width, text_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_COMPLEX, 1, 2
                    )

                    # Ensure text stays within frame bounds
                    text_x = max(x1, 10)
                    text_y = max(y1 - 10, text_height + 20)

                    # Draw background and text
                    cv2.rectangle(img_output,
                                (text_x - 5, text_y - text_height - 10),
                                (text_x + text_width + 5, text_y + 5),
                                (0, 255, 0), cv2.FILLED)
                    cv2.putText(img_output, label,
                                (text_x, text_y),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

                    # Draw hand bounding box
                    cv2.rectangle(img_output,(x1, y1), (x2, y2),(0, 255, 0), 4)

            except Exception as e:
                print(f"Processing error: {str(e)}")

        # Display output
        cv2.imshow('Sign Language Detection', img_output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except KeyboardInterrupt:
        break

# ================== CLEANUP ==================
cap.release()
cv2.destroyAllWindows()