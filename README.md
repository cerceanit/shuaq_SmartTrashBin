# shuaq — Smart Waste Sorting Bin

An IoT-powered automated waste classifier that uses computer vision to sort waste into recyclable categories in real time. Built with an Arduino, a webcam, and a MobileNet-based model trained on a hybrid Kaggle + custom dataset — achieving **96% real-time classification accuracy**.

---

## How It Works

Every 2 seconds, the system captures a frame from the webcam, crops it to a square, resizes it to 224×224, and runs inference through a MobileNet model. If the confidence for any class exceeds 90%, it sends a signal to the Arduino to physically sort the item.

```
Webcam → Frame capture → MobileNet inference → Arduino signal → Bin sorting
```

**Waste categories:**
| Label | Signal to Arduino |
|-------|------------------|
| `plastic` | `1` |
| `paper` | `2` |
| `trash` | `3` |

---

## Repository Structure

```
shuaq_SmartTrashBin/
│
├── TeachableMachineaArduino.py   # Main script: webcam → model → Arduino
├── arduino.py                    # Serial image receiver from Arduino (RLE-encoded)
├── keras_model.h5                # Trained MobileNet model (Google Teachable Machine)
├── labels.txt                    # Class labels
└── codeof.py                     # (placeholder)
```

---

## Setup & Usage

### Requirements

```bash
pip install numpy opencv-python tensorflow pyserial
```

### Run

1. Connect your Arduino to the correct COM port.
2. Update the serial port in `TeachableMachineaArduino.py`:
   ```python
   arduino = serial.Serial(port='COM9', baudrate=9600, timeout=.1)
   ```
   *(uncomment the arduino lines once your port is configured)*
3. Make sure `keras_model.h5` and `labels.txt` are in the same directory as the script.
4. Run:
   ```bash
   python TeachableMachineaArduino.py
   ```

The webcam window will open and display the current classification label at the bottom of the frame.

---

## Model

The model was trained using [Google Teachable Machine](https://teachablemachine.withgoogle.com/) on a hybrid dataset combining:
- A public waste classification dataset from Kaggle
- A custom-collected dataset

**Architecture:** MobileNet-V2 (exported as `.h5`)  
**Input:** 224×224 RGB image, normalized to `[-1, 1]`  
**Confidence threshold:** 90%  
**Accuracy:** 96% real-time

---

## Hardware

- Arduino (any model with serial support)
- USB webcam
- Servo motor(s) for physical bin sorting mechanism

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

*Built by [Uais Amangeldi](https://github.com/cerceanit)*
