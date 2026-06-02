# ============================================================
#   SMART DROWSINESS DETECTION SYSTEM — AUTO CALIBRATION
#   File     : drowsiness_detector_calibrated.py
#   Feature  : Calibrates EAR threshold to YOUR eyes at startup
#              No more manual threshold tuning needed
# ============================================================
0
import cv2
import mediapipe as mp
import numpy as np
import serial
import time

# ─────────────────────────────────────────``
#  CONFIGURATION
# ─────────────────────────────────────────

SERIAL_PORT       = 'COM5'
BAUD_RATE         = 9600
ALERT_SECONDS     = 7.0
WARNING_SECONDS   = 3.0

CALIBRATION_SECS  = 5            # How long to collect open-eye samples
THRESHOLD_FACTOR  = 0.75         # Threshold = open_ear * this factor
                                 # 0.75 means 75% of your natural open EAR
                                 # If still triggering falsely → lower to 0.70
                                 # If not triggering when drowsy → raise to 0.78

DISPLAY_W         = 640
DISPLAY_H         = 480
FACE_PAD          = 0.55
ZOOM_SMOOTHING    = 0.12

# ─────────────────────────────────────────
#  COLOURS
# ─────────────────────────────────────────

C_GREEN  = (0,   210,  80)
C_ORANGE = (0,   165, 255)
C_RED    = (45,   45, 215)
C_WHITE  = (255, 255, 255)
C_BLACK  = (0,     0,   0)
C_DARK   = (18,   18,  18)
C_PANEL  = (28,   28,  28)
C_BLUE   = (200, 140,  30)

# ─────────────────────────────────────────
#  EYE LANDMARKS
# ─────────────────────────────────────────

RIGHT_EYE = [33,  160, 158, 133, 153, 144]
LEFT_EYE  = [362, 385, 387, 263, 373, 380]

# ─────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────

def filled_rect(frame, x1, y1, x2, y2, color, alpha=0.55):
    x1,y1,x2,y2 = max(0,x1),max(0,y1),min(frame.shape[1],x2),min(frame.shape[0],y2)
    if x2<=x1 or y2<=y1: return
    sub  = frame[y1:y2, x1:x2]
    rect = np.full(sub.shape, color, dtype=np.uint8)
    frame[y1:y2, x1:x2] = cv2.addWeighted(rect, alpha, sub, 1-alpha, 0)

def shadow_text(frame, text, x, y, scale, color, thickness=1):
    cv2.putText(frame, text, (x+1,y+1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, C_BLACK, thickness+1, cv2.LINE_AA)
    cv2.putText(frame, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)

def progress_bar(frame, x, y, bw, bh, progress, bg, fill, label=""):
    cv2.rectangle(frame, (x,y), (x+bw,y+bh), bg, -1)
    cv2.rectangle(frame, (x,y), (x+bw,y+bh), (70,70,70), 1)
    fw = int(progress * bw)
    if fw > 0:
        cv2.rectangle(frame, (x,y), (x+fw,y+bh), fill, -1)
    if label:
        cv2.putText(frame, label, (x+6, y+bh-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_WHITE, 1, cv2.LINE_AA)

def draw_eye_hull(frame, pts_dict, indices, color, alpha=0.28):
    pts  = np.array([pts_dict[i] for i in indices], dtype=np.int32)
    hull = cv2.convexHull(pts)
    ov   = frame.copy()
    cv2.fillConvexPoly(ov, hull, color)
    cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)
    cv2.polylines(frame, [hull], True, color, 2, cv2.LINE_AA)

def calc_ear(pts_dict, indices):
    p = [pts_dict[i] for i in indices]
    A = np.linalg.norm(np.array(p[1])-np.array(p[5]))
    B = np.linalg.norm(np.array(p[2])-np.array(p[4]))
    C = np.linalg.norm(np.array(p[0])-np.array(p[3]))
    return (A+B)/(2.0*C) if C!=0 else 0.0

def get_scaled_pts(lms, cx1, cy1, crop_w, crop_h, fw, fh):
    sx = DISPLAY_W / max(crop_w, 1)
    sy = DISPLAY_H / max(crop_h, 1)
    return {
        i: (int((int(lm.x*fw)-cx1)*sx), int((int(lm.y*fh)-cy1)*sy))
        for i, lm in enumerate(lms)
    }

def get_face_box(lms, fw, fh):
    xs = [int(lm.x*fw) for lm in lms]
    ys = [int(lm.y*fh) for lm in lms]
    return min(xs), min(ys), max(xs), max(ys)

def padded_crop(fx1,fy1,fx2,fy2,fw,fh):
    pw = int((fx2-fx1)*FACE_PAD)
    ph = int((fy2-fy1)*FACE_PAD)
    x1 = max(0,  fx1-pw);  y1 = max(0,  fy1-ph)
    x2 = min(fw, fx2+pw);  y2 = min(fh, fy2+ph)
    cw,ch = x2-x1, y2-y1
    if cw>ch:
        d=cw-ch; y1=max(0,y1-d//2); y2=min(fh,y2+d//2)
    else:
        d=ch-cw; x1=max(0,x1-d//2); x2=min(fw,x2+d//2)
    return x1,y1,x2,y2

# ─────────────────────────────────────────
#  CONNECT ARDUINO
# ─────────────────────────────────────────

print("[INFO] Connecting to Arduino...")
try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print("[INFO] Arduino connected.")
except Exception as e:
    print(f"[WARNING] Arduino not connected: {e}")
    arduino = None

# ─────────────────────────────────────────
#  MEDIAPIPE INIT
# ─────────────────────────────────────────

mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.6, min_tracking_confidence=0.6)

cap = cv2.VideoCapture(0)
time.sleep(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DISPLAY_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_H)

# ─────────────────────────────────────────
#  ★  CALIBRATION PHASE
#  Ask user to keep eyes open for 5 seconds
#  Collect EAR samples → compute threshold
# ─────────────────────────────────────────

print("\n[CALIBRATION] Keep your eyes OPEN and look at the camera.")
print(f"[CALIBRATION] Collecting samples for {CALIBRATION_SECS} seconds...\n")

ear_samples   = []
cal_start     = time.time()
cal_complete  = False
EAR_THRESHOLD = 0.22          # fallback default

while True:
    ret, frame = cap.read()
    if not ret: break

    frame     = cv2.flip(frame, 1)
    fh, fw    = frame.shape[:2]
    elapsed   = time.time() - cal_start
    remaining = max(0, CALIBRATION_SECS - elapsed)
    progress  = min(elapsed / CALIBRATION_SECS, 1.0)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    res = face_mesh.process(rgb)
    rgb.flags.writeable = True

    display = frame.copy()

    # Dark overlay
    filled_rect(display, 0, 0, fw, fh, C_DARK, 0.45)

    # Title
    filled_rect(display, 0, 0, fw, 55, C_DARK, 0.85)
    shadow_text(display, "CALIBRATION", fw//2-80, 32, 0.80, C_BLUE, 2)
    cv2.line(display, (0,55),(fw,55),(55,55,55),1)

    if res.multi_face_landmarks:
        lms    = res.multi_face_landmarks[0].landmark
        fx1,fy1,fx2,fy2 = get_face_box(lms, fw, fh)
        cx1,cy1,cx2,cy2 = padded_crop(fx1,fy1,fx2,fy2,fw,fh)
        spts   = get_scaled_pts(lms, cx1, cy1, cx2-cx1, cy2-cy1, fw, fh)

        # Draw face box
        cv2.rectangle(display, (fx1,fy1),(fx2,fy2), C_BLUE, 2)

        ear = (calc_ear(spts, LEFT_EYE) + calc_ear(spts, RIGHT_EYE)) / 2.0

        # Only collect if EAR looks valid (eyes actually open)
        if ear > 0.15 and elapsed > 0.5:
            ear_samples.append(ear)

        # Live EAR reading
        shadow_text(display, f"Live EAR: {ear:.3f}", fw//2-70, 100,
                    0.65, C_GREEN if ear>0.18 else C_ORANGE, 2)
        shadow_text(display, f"Samples: {len(ear_samples)}", fw//2-55, 128, 0.50, (170,170,170))

    else:
        shadow_text(display, "No face found — look at camera", fw//2-160, 100,
                    0.60, C_ORANGE, 1)

    # Instruction
    filled_rect(display, fw//2-220, fh//2-25, fw//2+220, fh//2+25, C_DARK, 0.80)
    shadow_text(display, "Keep eyes OPEN and face the camera",
                fw//2-195, fh//2+8, 0.58, C_WHITE, 1)

    # Countdown bar
    bar_color = C_GREEN if len(ear_samples) > 5 else C_ORANGE
    filled_rect(display, 60, fh-80, fw-60, fh-40, C_PANEL, 0.80)
    progress_bar(display, 68, fh-76, fw-136, 28,
                 progress, (50,50,50), bar_color,
                 f"Calibrating...  {remaining:.1f}s remaining")

    # Sample count confirmation
    if len(ear_samples) > 10:
        shadow_text(display, f"✓  {len(ear_samples)} samples collected",
                    fw//2-105, fh-95, 0.50, C_GREEN)

    cv2.imshow("Smart Drowsiness Detection System", display)

    if elapsed >= CALIBRATION_SECS:
        cal_complete = True
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── Compute threshold from samples ───────
if cal_complete and len(ear_samples) >= 5:
    open_ear_avg  = float(np.mean(ear_samples))
    open_ear_min  = float(np.min(ear_samples))
    EAR_THRESHOLD = round(open_ear_avg * THRESHOLD_FACTOR, 4)

    print(f"\n[CALIBRATION] ✓ Complete!")
    print(f"  Your average open-eye EAR : {open_ear_avg:.4f}")
    print(f"  Your minimum open-eye EAR : {open_ear_min:.4f}")
    print(f"  Threshold set to          : {EAR_THRESHOLD:.4f}  "
          f"({int(THRESHOLD_FACTOR*100)}% of avg)\n")

    # Show result screen for 3 seconds
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        fh2, fw2 = frame.shape[:2]
        filled_rect(frame, 0, 0, fw2, fh2, C_DARK, 0.6)
        filled_rect(frame, fw2//2-240, fh2//2-80, fw2//2+240, fh2//2+80, C_PANEL, 0.90)
        shadow_text(frame, "CALIBRATION COMPLETE", fw2//2-155, fh2//2-42, 0.72, C_GREEN, 2)
        shadow_text(frame, f"Your EAR (open eyes): {open_ear_avg:.3f}",
                    fw2//2-165, fh2//2-8, 0.55, C_WHITE)
        shadow_text(frame, f"Alert threshold set:  {EAR_THRESHOLD:.3f}",
                    fw2//2-165, fh2//2+22, 0.55, C_ORANGE)
        shadow_text(frame, "Starting detection in 3 seconds...",
                    fw2//2-185, fh2//2+58, 0.52, (160,160,160))
        cv2.imshow("Smart Drowsiness Detection System", frame)
        cv2.waitKey(3000)
else:
    print(f"[WARNING] Not enough samples. Using default threshold: {EAR_THRESHOLD}")

# ─────────────────────────────────────────
#  DETECTION STATE
# ─────────────────────────────────────────

eyes_closed_start = None
alert_active      = False
total_alerts      = 0
session_start     = time.time()
smooth_x1 = smooth_y1 = 0.0
smooth_x2 = float(DISPLAY_W)
smooth_y2 = float(DISPLAY_H)
face_found_prev = False

print(f"[INFO] Detection running. Threshold = {EAR_THRESHOLD:.4f}  |  Press Q to quit.\n")

# ─────────────────────────────────────────
#  MAIN DETECTION LOOP
# ─────────────────────────────────────────

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Cannot read from webcam.")
        break

    fh, fw        = frame.shape[:2]
    frame         = cv2.flip(frame, 1)
    current_time  = time.time()
    session_dur   = current_time - session_start
    closed_dur    = 0.0
    avg_ear       = 0.0
    state         = "no_face"
    scaled_pts    = {}

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = face_mesh.process(rgb)
    rgb.flags.writeable = True

    if results.multi_face_landmarks:
        lms = results.multi_face_landmarks[0].landmark
        fx1,fy1,fx2,fy2 = get_face_box(lms, fw, fh)
        tx1,ty1,tx2,ty2 = padded_crop(fx1,fy1,fx2,fy2,fw,fh)

        if not face_found_prev:
            smooth_x1,smooth_y1,smooth_x2,smooth_y2 = float(tx1),float(ty1),float(tx2),float(ty2)
        else:
            s = ZOOM_SMOOTHING
            smooth_x1 += s*(tx1-smooth_x1); smooth_y1 += s*(ty1-smooth_y1)
            smooth_x2 += s*(tx2-smooth_x2); smooth_y2 += s*(ty2-smooth_y2)

        face_found_prev = True
        cx1,cy1 = int(smooth_x1),int(smooth_y1)
        cx2,cy2 = int(smooth_x2),int(smooth_y2)
        scaled_pts = get_scaled_pts(lms, cx1, cy1, cx2-cx1, cy2-cy1, fw, fh)

        avg_ear = (calc_ear(scaled_pts, LEFT_EYE) + calc_ear(scaled_pts, RIGHT_EYE)) / 2.0

        if avg_ear < EAR_THRESHOLD:
            if eyes_closed_start is None:
                eyes_closed_start = current_time
            closed_dur = current_time - eyes_closed_start
            state = "alert" if closed_dur>=ALERT_SECONDS else \
                    "warning" if closed_dur>=WARNING_SECONDS else "closing"
        else:
            if eyes_closed_start is not None:
                print(f"[INFO] Eyes open. Closed for {current_time-eyes_closed_start:.1f}s")
                eyes_closed_start = None
            if alert_active:
                alert_active = False
                if arduino: arduino.write(b'0')
                print("[INFO] Alert cleared.")
            state = "monitoring"
    else:
        face_found_prev = False
        smooth_x1 += ZOOM_SMOOTHING*(0 -smooth_x1); smooth_y1 += ZOOM_SMOOTHING*(0 -smooth_y1)
        smooth_x2 += ZOOM_SMOOTHING*(fw-smooth_x2); smooth_y2 += ZOOM_SMOOTHING*(fh-smooth_y2)
        state = "alert" if alert_active else "no_face"

    if state == "alert" and not alert_active:
        alert_active = True; total_alerts += 1
        print(f"[ALERT] Drowsiness detected! Event #{total_alerts}")
        if arduino: arduino.write(b'1')

    # ── Build zoomed display ──────────────
    cx1=max(0,int(smooth_x1)); cy1=max(0,int(smooth_y1))
    cx2=min(fw,int(smooth_x2)); cy2=min(fh,int(smooth_y2))
    crop = frame[cy1:cy2, cx1:cx2] if cx2>cx1 and cy2>cy1 else frame
    display = cv2.resize(crop, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_LINEAR)
    dh, dw  = display.shape[:2]

    # Eye hulls
    if scaled_pts:
        ec = C_GREEN if state=="monitoring" else C_ORANGE if state in("closing","warning") else C_RED
        draw_eye_hull(display, scaled_pts, LEFT_EYE,  ec)
        draw_eye_hull(display, scaled_pts, RIGHT_EYE, ec)

    # PiP
    pip_w,pip_h = 160,120
    pip_frame   = cv2.resize(frame,(pip_w,pip_h))
    if results.multi_face_landmarks:
        px,py = pip_w/fw, pip_h/fh
        cv2.rectangle(pip_frame,(int(cx1*px),int(cy1*py)),(int(cx2*px),int(cy2*py)),
                      C_RED if alert_active else C_GREEN, 1)
    pip_x, pip_y = dw-pip_w-10, 62
    cv2.rectangle(pip_frame,(0,0),(pip_w-1,pip_h-1),(100,100,100),1)
    display[pip_y:pip_y+pip_h, pip_x:pip_x+pip_w] = pip_frame
    shadow_text(display,"Full view", pip_x, pip_y+pip_h+14, 0.35, (160,160,160))

    # Zoom factor
    if results.multi_face_landmarks:
        zf  = 1.0/max((cx2-cx1)/fw,0.01)
        zc  = C_ORANGE if zf>2.5 else C_GREEN
        filled_rect(display, pip_x, pip_y+pip_h+18, pip_x+pip_w, pip_y+pip_h+34, C_PANEL, 0.7)
        shadow_text(display, f"ZOOM  {zf:.1f}x", pip_x+6, pip_y+pip_h+31, 0.40, zc)

    # Top bar
    filled_rect(display,0,0,dw,52,C_DARK,0.78)
    shadow_text(display,"DROWSINESS DETECTION SYSTEM",10,22,0.52,C_WHITE)
    mins,secs = int(session_dur//60),int(session_dur%60)
    shadow_text(display,f"{mins:02d}:{secs:02d}",10,44,0.40,(160,160,160))
    filled_rect(display,140,32,300,50,C_PANEL,0.75)
    shadow_text(display,f"Alerts: {total_alerts}",148,46,0.40,
                C_ORANGE if total_alerts>0 else (130,130,130))

    # Threshold info pill
    filled_rect(display,310,32,500,50,C_PANEL,0.75)
    shadow_text(display,f"Threshold: {EAR_THRESHOLD:.3f}",316,46,0.38,(130,180,130))

    cv2.line(display,(0,52),(dw,52),(55,55,55),1)

    # EAR panel
    if scaled_pts:
        filled_rect(display,8,dh-72,195,dh-8,C_PANEL,0.72)
        ear_c = C_GREEN if avg_ear>=EAR_THRESHOLD else C_RED
        shadow_text(display,"EAR",16,dh-54,0.38,(150,150,150))
        shadow_text(display,f"{avg_ear:.3f}",16,dh-28,0.75,ear_c,2)
        bv = min(avg_ear/(EAR_THRESHOLD*1.8),1.0)
        progress_bar(display,16,dh-18,170,8,bv,(55,55,55),ear_c)
        tp = int(16+(EAR_THRESHOLD/(EAR_THRESHOLD*1.8))*170)
        cv2.line(display,(tp,dh-22),(tp,dh-6),C_WHITE,1)
        shadow_text(display,"▲",tp-5,dh-4,0.30,(200,200,200))

    # Progress bar
    if state in("closing","warning","alert") and eyes_closed_start:
        prog = min(closed_dur/ALERT_SECONDS,1.0)
        bc   = C_ORANGE if state in("closing","warning") else C_RED
        lbl  = f"Eyes closed  {closed_dur:.1f}s / {int(ALERT_SECONDS)}s"
        filled_rect(display,8,dh-100,dw-8,dh-80,C_PANEL,0.72)
        progress_bar(display,12,dh-96,dw-28,14,prog,(55,55,55),bc,lbl)

    # State banners
    if state=="monitoring":
        filled_rect(display,8,58,225,82,C_PANEL,0.72)
        shadow_text(display,"●  MONITORING",18,75,0.50,C_GREEN)
    elif state=="closing":
        filled_rect(display,8,58,275,82,C_PANEL,0.72)
        shadow_text(display,"●  EYES CLOSING...",18,75,0.50,C_ORANGE)
    elif state=="warning":
        filled_rect(display,0,56,dw-pip_w-20,90,C_ORANGE,0.22)
        shadow_text(display,"⚠  WARNING — KEEP EYES OPEN!",18,80,0.60,C_ORANGE,2)
    elif state=="alert":
        flash = int(current_time*3)%2==0
        filled_rect(display,0,54,dw-pip_w-20,96,C_RED,0.85 if flash else 0.60)
        shadow_text(display,"DROWSINESS ALERT!  WAKE UP!",14,83,0.75,C_WHITE,2)
    elif state=="no_face":
        filled_rect(display,8,58,262,82,C_PANEL,0.72)
        shadow_text(display,"●  NO FACE DETECTED",18,75,0.50,C_ORANGE)

    # Status dot
    sc = C_RED if alert_active else C_GREEN
    filled_rect(display,dw-110,dh-38,dw-8,dh-8,C_PANEL,0.72)
    cv2.circle(display,(dw-96,dh-22),6,sc,-1,cv2.LINE_AA)
    shadow_text(display,"ALERT" if alert_active else "SAFE",dw-86,dh-16,0.50,sc)

    cv2.imshow("Smart Drowsiness Detection System", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Quit.")
        break

if arduino:
    arduino.write(b'0'); arduino.close()
face_mesh.close(); cap.release(); cv2.destroyAllWindows()
print(f"\n[INFO] Done. Total alerts: {total_alerts}")
