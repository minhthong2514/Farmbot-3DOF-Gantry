import cv2
import numpy as np
import os

# ============ CONFIGURATION — ADJUST TO MATCH YOUR PHYSICAL SETUP ============
IMAGES_DIR = "./images"
PARAMS_PATH = "../camera_params.npz"          # Source file for K and dist
OUTPUT_PATH = "../camera_extrinsic.npz"       # Target file to save R and t

PATTERN_SIZE = (10, 8)                         # Inner corners count (cols, rows)
SQUARE_SIZE_X_MM = 58.5 / 11                   # DO NOT round
SQUARE_SIZE_Y_MM = 48.1 / 9                    # DO NOT round

# Reference anchor corner (0-indexed): col 6, row 5 (i.e., "7,6" 1-based index)
ANCHOR_COL = 6
ANCHOR_ROW = 5
ANCHOR_WORLD_X = 80.0     # mm, Current_X when aligned with cx, cy
ANCHOR_WORLD_Y = 50.0     # mm, Current_Y when aligned with cx, cy

# Axis sign mapping between image frame and robot frame
SIGN_X = +1   # Change to -1 if +image_col corresponds to -X_robot
SIGN_Y = +1   # Change to -1 if +image_row corresponds to -Y_robot

# Image pair list: (filename, robot Z-lowered value in mm)
IMAGE_Z_PAIRS = [
    ("img-0.jpg",   0),
    ("img-25.jpg",  25),
    ("img-50.jpg",  50),
    ("img-75.jpg",  75),
    ("img-100.jpg", 100),
    ("img-125.jpg", 125),
    ("img-150.jpg", 150),
]
# ==============================================================================


def build_object_points(anchor_row, anchor_col, anchor_x, anchor_y, sign_x, sign_y, z_robot):
    cols, rows = PATTERN_SIZE
    obj_pts = np.zeros((rows * cols, 3), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            world_x = anchor_x + sign_x * (c - anchor_col) * SQUARE_SIZE_X_MM
            world_y = anchor_y + sign_y * (r - anchor_row) * SQUARE_SIZE_Y_MM
            obj_pts[idx] = [world_x, world_y, float(z_robot)]   # Z level set according to z_robot
    return obj_pts


def main():
    params = np.load(PARAMS_PATH)
    K = params["K"]
    dist = params["dist"]

    all_object_points = []   # List of (80, 3) arrays - Object points per image
    all_image_points = []    # List of (80, 2) arrays - Image points per image

    for fname, z_val in IMAGE_Z_PAIRS:
        path = os.path.join(IMAGES_DIR, fname)
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, PATTERN_SIZE,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if not found:
            raise RuntimeError(f"Failed to detect chessboard in {fname}")

        # Refine corner locations to sub-pixel accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        obj_pts = build_object_points(
            ANCHOR_ROW, ANCHOR_COL, ANCHOR_WORLD_X, ANCHOR_WORLD_Y,
            SIGN_X, SIGN_Y, z_val
        )

        all_object_points.append(obj_pts)
        all_image_points.append(corners.reshape(-1, 2))
        print(f"[OK] {fname}: detected all {len(corners)} corners, Z_robot={z_val}mm")

    # Concatenate 7 images x 80 points = 560 point pairs, solve once via solvePnP
    obj_all = np.concatenate(all_object_points, axis=0).astype(np.float64)
    img_all = np.concatenate(all_image_points, axis=0).astype(np.float64)

    success, rvec, tvec = cv2.solvePnP(
        obj_all, img_all, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        raise RuntimeError("solvePnP failed!")

    R, _ = cv2.Rodrigues(rvec)   # Convert rotation vector -> 3x3 rotation matrix

    # Calculate reprojection error to verify calibration accuracy
    proj_pts, _ = cv2.projectPoints(obj_all, rvec, tvec, K, dist)
    proj_pts = proj_pts.reshape(-1, 2)
    errors = np.linalg.norm(proj_pts - img_all, axis=1)
    
    print(f"\n--- Calibration Results ---")
    print(f"Reprojection error: mean={errors.mean():.3f}px, max={errors.max():.3f}px")
    print(f"R =\n{R}")
    print(f"t (camera position, mm, at Z_robot=0/home) =\n{tvec.ravel()}")
    
    cam_pos_world = (-R.T @ tvec).ravel()
    print("Camera position in robot frame:", cam_pos_world)

    np.savez(OUTPUT_PATH, R=R, t=tvec, rvec=rvec,
             reprojection_error_mean=errors.mean(),
             reprojection_error_max=errors.max())
    print(f"\nSaved successfully to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()