import os
import sys
import cv2
import numpy as np
import threading
import time
import queue
import ctypes

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'src')))
from yolov5_trt import YoLov5TRT

# ============ CONFIGURATION ============
PARAMS_PATH = "/home/minhthong/Desktop/code/farmbot/calib-camera/camera_params.npz"
EXTRINSIC_PATH = "/home/minhthong/Desktop/code/farmbot/calib-camera/camera_extrinsic.npz"
ENGINE_PATH = "../../models/nano/farmbot_seg_model.engine"

# Real physical dimensions of the rectangular bag (in mm)
BAG_REAL_X_MM = 123.0  # Real Width along X axis (mm)
BAG_REAL_Y_MM = 86.0   # Real Height along Y axis (mm)

# Hardware GStreamer Pipeline
GSTREAMER_PIPELINE = (
    "v4l2src device=/dev/video0 ! "
    "image/jpeg, width=640, height=480, framerate=30/1 ! "
    "jpegdec ! videoconvert ! video/x-raw, format=BGR ! appsink drop=true"
)


class RayCastingTester:
    """Calculates Z-depth estimation and 3D Ray-Casting target points in Robot World Frame."""
    def __init__(self, params_path, extrinsic_path):
        cam_data = np.load(params_path)
        self.K = cam_data["K"].astype(np.float64)
        self.dist = cam_data["dist"].astype(np.float64)

        self.fx = float(self.K[0, 0])
        self.fy = float(self.K[1, 1])
        self.cx = float(self.K[0, 2])
        self.cy = float(self.K[1, 2])

        ext_data = np.load(extrinsic_path)
        self.R = ext_data["R"].astype(np.float64)
        self.t = ext_data["t"].astype(np.float64).reshape(3, 1)

        self.C_robot = (-self.R.T @ self.t).ravel()

        print("--- Intrinsic Parameters ---")
        print(f"fx: {self.fx:.2f}, fy: {self.fy:.2f}, cx: {self.cx:.2f}, cy: {self.cy:.2f}")
        print("--- Extrinsic Parameters ---")
        print(f"Camera World Pos (C_robot): X={self.C_robot[0]:.1f}, Y={self.C_robot[1]:.1f}, Z={self.C_robot[2]:.1f} mm")

    def estimate_depth_from_bbox(self, bbox_w_px, bbox_h_px):
        if bbox_w_px <= 0 or bbox_h_px <= 0:
            return float(self.C_robot[2])

        z_from_x = (self.fx * BAG_REAL_X_MM) / float(bbox_w_px)
        z_from_y = (self.fy * BAG_REAL_Y_MM) / float(bbox_h_px)

        z_estimated = (z_from_x + z_from_y) / 2.0
        return z_estimated

    def pixel_to_world_3d(self, u, v, z_estimated_mm):
        pts = np.array([[[u, v]]], dtype=np.float64)
        undistorted = cv2.undistortPoints(pts, self.K, self.dist)
        x_norm, y_norm = undistorted[0, 0]

        P_cam = -z_estimated_mm * np.array([x_norm, y_norm, 1.0], dtype=np.float64)
        p_world = self.R.T @ P_cam + self.C_robot
        #print(p_world)

        return p_world, z_estimated_mm

class CameraRayCastRunner(threading.Thread):
    """Camera Capture, TensorRT Inference, and Ray-Casting Execution Thread."""
    def __init__(self, engine_path, params_path, extrinsic_path, enable_display=True):
        super().__init__(daemon=True)
        
        self.tester = RayCastingTester(params_path, extrinsic_path)

        self.cap = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            print("[WARN] GStreamer failed. Trying VideoCapture(0) fallback...")
            self.cap = cv2.VideoCapture(0)

        self.INPUT_SIZE = 640
        self.IOU_THRESH = 0.45
        self.CONF_THRESH = 0.8
        self.classes = ["bag", "strawberry", "sweet potato"]
        
        print(f"[TRT] Loading TensorRT model from {engine_path}...")
        try:
            self.model = YoLov5TRT(engine_path, self.classes, self.CONF_THRESH, self.IOU_THRESH)
            print("[TRT] Model loaded successfully.")
        except Exception as e:
            print(f"[TRT ERROR] Failed to initialize YoLov5TRT: {e}")
            self.model = None

        self.enable_display = enable_display
        self.running = False
        self.draw_mask = False  # Disable color filling mask overlay to optimize FPS
        self.frame_queue = queue.Queue(maxsize=2)
        self.lock = threading.Lock()

        self.prev_time = time.time()
        self.capture_thread = None
        self.display_thread = None

    def start_pipeline(self):
        if self.running:
            return
        self.running = True
        
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

        if self.enable_display:
            self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
            self.display_thread.start()

    def stop_pipeline(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        if self.model is not None:
            self.model.destroy()

    def _capture_loop(self):
        print("[CAMERA] Hardware capture loop started.")
        while self.running:
            if self.frame_queue.full():
                self.cap.grab()
                time.sleep(0.005)
                continue

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.005)
                continue

            self.frame_queue.put(frame)

        print("[CAMERA] Capture loop stopped.")

    def _display_loop(self):
        print("[DISPLAY] Display loop started.")
        cv2.namedWindow("Ray Casting TRT Test", cv2.WINDOW_NORMAL)

        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            draw = self.process_and_raycast(frame)

            if draw is not None:
                cv2.imshow("Ray Casting TRT Test", draw)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break

        cv2.destroyAllWindows()
        print("[DISPLAY] Display loop stopped.")

    def process_and_raycast(self, frame):
        draw = frame.copy()
        h_frame, w_frame = frame.shape[:2]

        boxes = None
        masks = None
        infer_time = 0.0

        if self.model is not None:
            infer_outputs = self.model.infer([frame], self.draw_mask)
            if len(infer_outputs) == 4:
                batch_results, infer_time, boxes, masks = infer_outputs
            else:
                batch_results, infer_time, boxes = infer_outputs[:3]
                masks = None
            
            # Clean image without default AABB bounding boxes
            draw = frame.copy()

        # Draw optical center (cx, cy)
        cx_i, cy_i = int(self.tester.cx), int(self.tester.cy)
        cv2.circle(draw, (cx_i, cy_i), 5, (255, 0, 0), -1)

        if boxes is not None and len(boxes) > 0:
            for idx, box in enumerate(boxes):
                class_id = int(box[5])

                if class_id >= len(self.classes):
                    continue

                label = self.classes[class_id]

                if label == "bag":
                    # Default values fallback
                    u_px = float((box[0] + box[2]) / 2.0)
                    v_px = float((box[1] + box[3]) / 2.0)
                    w_px = float(box[2] - box[0])
                    h_px = float(box[3] - box[1])
                    
                    text_x = int(box[0])
                    text_y = int(box[1])

                    # --- OBB EXTRACTION VIA MASK SEGMENTATION ---
                    if masks is not None and len(masks) > idx:
                        raw_mask = masks[idx]
                        if raw_mask is not None and raw_mask.size > 0:
                            mask_gray = (raw_mask * 255).astype(np.uint8) if raw_mask.max() <= 1 else raw_mask.astype(np.uint8)
                            
                            kernel = np.ones((3, 3), np.uint8)
                            mask_clean = cv2.morphologyEx(mask_gray, cv2.MORPH_OPEN, kernel)

                            contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            if contours:
                                valid_contours = [c for c in contours if cv2.contourArea(c) > 500]
                                if valid_contours:
                                    c_max = max(valid_contours, key=cv2.contourArea)
                                    
                                    # Extract Minimum Area Rectangle (OBB)
                                    rect = cv2.minAreaRect(c_max)
                                    (u_obb, v_obb), (w_obb, h_obb), angle = rect

                                    u_px, v_px = float(u_obb), float(v_obb)
                                    
                                    # Assign longest edge to Real X, shortest edge to Real Y
                                    w_px = float(max(w_obb, h_obb))
                                    h_px = float(min(w_obb, h_obb))

                                    # Generate 4 corner points for OBB
                                    box_points = cv2.boxPoints(rect)
                                    box_points = np.int0(box_points)

                                    # Draw Red OBB Box
                                    cv2.drawContours(draw, [box_points], 0, (0, 0, 255), 2)

                                    # Text display anchor position (top-leftmost point of OBB)
                                    text_x = int(np.min(box_points[:, 0]))
                                    text_y = int(np.min(box_points[:, 1]))

                    # 1. Estimate depth Z from OBB pixel dimensions
                    dist_to_bag = self.tester.estimate_depth_from_bbox(w_px, h_px)

                    # 2. Compute 3D world position in Robot Frame
                    p_world, _ = self.tester.pixel_to_world_3d(u_px, v_px, dist_to_bag)
                    bag_height = p_world[2]

                    # Render target markers
                    u_i, v_i = int(u_px), int(v_px)
                    cv2.circle(draw, (u_i, v_i), 5, (0, 0, 255), -1)
                    cv2.line(draw, (cx_i, cy_i), (u_i, v_i), (0, 255, 255), 2)

                    # Display metrics aligned with OBB position
                    text_z = f"Dist: {dist_to_bag:.1f}mm | Height: {bag_height:.1f}mm"
                    text_world = f"World: [{p_world[0]:.1f}, {p_world[1]:.1f}, {p_world[2]:.1f}]"
                    
                    cv2.putText(draw, text_z, (text_x, max(15, text_y - 25)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(draw, text_world, (text_x, max(30, text_y - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # FPS calculation
        curr_time = time.time()
        fps = 1.0 / (curr_time - self.prev_time) if (curr_time - self.prev_time) > 0 else 0.0
        self.prev_time = curr_time

        cv2.putText(draw, f"FPS: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(draw, f"Infer: {infer_time*1000:.1f}ms", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return draw


def main():
    try:
        ctypes.CDLL("../../src/libmyplugins.so")
    except Exception as e:
        print(f"Warning: Could not load libmyplugins.so. Error: {e}")

    runner = CameraRayCastRunner(
        engine_path=ENGINE_PATH,
        params_path=PARAMS_PATH,
        extrinsic_path=EXTRINSIC_PATH,
        enable_display=True
    )

    runner.start_pipeline()

    try:
        while runner.running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[INFO] Stopping test pipeline...")
    finally:
        runner.stop_pipeline()


if __name__ == "__main__":
    main()
