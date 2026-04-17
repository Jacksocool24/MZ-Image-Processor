import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class ProcessingResult:
    ocr_raw_text: str
    ocr_size_token: Optional[str]
    content_rotation_deg: int
    production_rotation_deg: int
    total_rotation_deg: int
    needs_align: bool
    needs_physical_rotate: bool
    canvas_size_px: Tuple[int, int]
    rendered_image_bgr: np.ndarray


def ocr_size_from_demo_bottom(
    demo_image_bgr: np.ndarray,
    reader: Optional[object] = None,
) -> Tuple[str, Optional[str]]:
    del demo_image_bgr, reader
    return "", None


_WHITE_BGR = (255, 255, 255)


def _rotate_with_white_border(image_bgr: np.ndarray, angle_ccw_deg: float) -> np.ndarray:
    """getRotationMatrix2D + warpAffine；angle_ccw_deg 为 OpenCV 约定（逆时针为正）。"""
    h, w = image_bgr.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_ccw_deg, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nW = int(h * sin + w * cos)
    nH = int(h * cos + w * sin)
    M[0, 2] += (nW / 2.0) - center[0]
    M[1, 2] += (nH / 2.0) - center[1]
    return cv2.warpAffine(
        image_bgr,
        M,
        (nW, nH),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=_WHITE_BGR,
    )


def rotate_image_cw(image_bgr: np.ndarray, degree: int) -> np.ndarray:
    """顺时针 0/90/180/270；旋转产生的空隙用纯白填充（与 cv2.rotate 默认黑边区分）。"""
    degree = degree % 360
    if degree == 0:
        return image_bgr.copy()
    if degree == 90:
        return _rotate_with_white_border(image_bgr, -90.0)
    if degree == 180:
        return _rotate_with_white_border(image_bgr, 180.0)
    if degree == 270:
        return _rotate_with_white_border(image_bgr, 90.0)
    raise ValueError("degree must be one of: 0, 90, 180, 270")


def _resize_long_edge_bgr(image_bgr: np.ndarray, long_edge: int = 800) -> np.ndarray:
    """等比例缩小，长边不超过指定像素；纠偏只看轮廓，减轻 SIFT/ORB 负担。"""
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    h, w = image_bgr.shape[:2]
    m = max(h, w)
    if m <= long_edge:
        return image_bgr
    scale = long_edge / float(m)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_AREA)


def detect_actual_content_box(
    image_bgr: np.ndarray,
    color_delta_l2: float = 28.0,
    std_gate: float = 6.0,
) -> Tuple[int, int, int, int]:
    """
    边缘向内收缩探测：在「探测图」坐标系下返回内容包围盒 (left, top, right, bottom)，
    right/bottom 为开区间边界（与 Python 切片 [left:right, top:bottom] 一致）。

    从四边向中心逐行/列扫描：以该侧最外一行/列为参照色（BGR 均值），当该行/列与参照的
    欧氏距离超过阈值，或行内像素标准差明显大于「边缘带」典型值时，视为进入画面内容。
    底色可为任意纯色（土黄等），只要与画面区颜色不同即可检出。
    """
    if image_bgr is None or image_bgr.size == 0:
        return 0, 0, 1, 1
    h, w = image_bgr.shape[:2]
    if h < 3 or w < 3:
        return 0, 0, w, h

    img = image_bgr.astype(np.float32)

    def row_mean_std(y: int) -> Tuple[np.ndarray, float]:
        row = img[y, :, :]
        return row.mean(axis=0), float(row.std())

    def col_mean_std(x: int) -> Tuple[np.ndarray, float]:
        col = img[:, x, :]
        return col.mean(axis=0), float(col.std())

    ref_top, std_edge_top = row_mean_std(0)
    ref_bottom, std_edge_bot = row_mean_std(h - 1)
    ref_left, std_edge_left = col_mean_std(0)
    ref_right, std_edge_right = col_mean_std(w - 1)

    def row_changed(y: int, ref: np.ndarray, std_edge: float) -> bool:
        m, s = row_mean_std(y)
        if float(np.linalg.norm(m - ref)) > color_delta_l2:
            return True
        if s > std_edge + std_gate:
            return True
        return False

    def col_changed(x: int, ref: np.ndarray, std_edge: float) -> bool:
        m, s = col_mean_std(x)
        if float(np.linalg.norm(m - ref)) > color_delta_l2:
            return True
        if s > std_edge + std_gate:
            return True
        return False

    top = 0
    for y in range(1, h):
        if row_changed(y, ref_top, std_edge_top):
            top = y
            break

    bottom = h
    for y in range(h - 2, -1, -1):
        if row_changed(y, ref_bottom, std_edge_bot):
            bottom = y + 1
            break

    left = 0
    for x in range(1, w):
        if col_changed(x, ref_left, std_edge_left):
            left = x
            break

    right = w
    for x in range(w - 2, -1, -1):
        if col_changed(x, ref_right, std_edge_right):
            right = x + 1
            break

    left = max(0, min(left, w - 1))
    top = max(0, min(top, h - 1))
    right = max(left + 1, min(right, w))
    bottom = max(top + 1, min(bottom, h))

    return left, top, right, bottom


def _probe_box_to_original_rect(
    probe_shape: Tuple[int, int],
    orig_shape: Tuple[int, int],
    box_ltrb: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    """probe (ph, pw) 上的 [l,t,r,b) 映射回原图坐标。"""
    ph, pw = probe_shape
    oh, ow = orig_shape
    l, t, r, b = box_ltrb
    sx = ow / float(pw)
    sy = oh / float(ph)
    L = int(round(l * sx))
    T = int(round(t * sy))
    R = int(round(r * sx))
    B = int(round(b * sy))
    L = max(0, min(L, ow - 1))
    T = max(0, min(T, oh - 1))
    R = max(L + 1, min(R, ow))
    B = max(T + 1, min(B, oh))
    return L, T, R, B


def _safe_crop_demo_core(
    demo_image_bgr: np.ndarray,
    rect_ltrb: Tuple[int, int, int, int],
    min_side_px: int = 8,
) -> np.ndarray:
    L, T, R, B = rect_ltrb
    crop = demo_image_bgr[T:B, L:R]
    if crop.size == 0 or crop.shape[0] < min_side_px or crop.shape[1] < min_side_px:
        return demo_image_bgr.copy()
    return crop


def _build_feature_detector():
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(), "SIFT"
    return cv2.ORB_create(nfeatures=3000), "ORB"


def _feature_match_score(
    ref_image_bgr: np.ndarray,
    query_image_bgr: np.ndarray,
) -> float:
    detector, kind = _build_feature_detector()
    ref_gray = cv2.cvtColor(ref_image_bgr, cv2.COLOR_BGR2GRAY)
    query_gray = cv2.cvtColor(query_image_bgr, cv2.COLOR_BGR2GRAY)

    kp1, des1 = detector.detectAndCompute(ref_gray, None)
    kp2, des2 = detector.detectAndCompute(query_gray, None)
    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return 0.0

    if kind == "SIFT":
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        ratio = 0.75
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        ratio = 0.8

    knn = matcher.knnMatch(des1, des2, k=2)
    good = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    if not good:
        return 0.0
    return float(len(good))


def sift_best_rotation_deg(
    demo_image_bgr: np.ndarray,
    original_image_bgr: np.ndarray,
    match_long_edge_px: int = 800,
) -> Tuple[int, Dict[int, float]]:
    """
    demo_image_bgr 建议为去掉底色的示意图核心裁剪区；特征匹配前将示意图与原图（各旋转）
    缩至长边 match_long_edge_px，加速且减少边框干扰。
    """
    demo_small = _resize_long_edge_bgr(demo_image_bgr, match_long_edge_px)
    scores: Dict[int, float] = {}
    for degree in (0, 90, 180, 270):
        rotated = rotate_image_cw(original_image_bgr, degree)
        rotated_small = _resize_long_edge_bgr(rotated, match_long_edge_px)
        scores[degree] = _feature_match_score(demo_small, rotated_small)
    best_degree = max(scores, key=scores.get)
    return best_degree, scores


def inch_token_to_pixels(size_token: str, dpi: int = 100) -> Tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)\s*", size_token)
    if not match:
        raise ValueError(f"Invalid size token: {size_token}")
    w_in = float(match.group(1))
    h_in = float(match.group(2))
    w_px = max(1, int(round(w_in * dpi)))
    h_px = max(1, int(round(h_in * dpi)))
    return w_px, h_px


def parse_inch_wh(size_token: str) -> Tuple[float, float]:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)\s*", size_token)
    if not match:
        raise ValueError(f"Invalid size token: {size_token}")
    return float(match.group(1)), float(match.group(2))


def render_cover(
    image_bgr: np.ndarray,
    canvas_size_px: Tuple[int, int],
) -> np.ndarray:
    """
    将 aligned 图像等比放大/缩小至完全覆盖画布，再中心裁剪为 canvas 尺寸（无留白）。
    scale = max(canvas_w/img_w, canvas_h/img_h)，与 contain 的 min 相反。
    """
    canvas_w, canvas_h = canvas_size_px
    if canvas_w <= 0 or canvas_h <= 0:
        raise ValueError("Canvas size must be positive.")

    img_h, img_w = image_bgr.shape[:2]
    if img_h <= 0 or img_w <= 0:
        raise ValueError("Image size must be positive.")

    scale = max(canvas_w / float(img_w), canvas_h / float(img_h))
    target_w = max(1, int(round(img_w * scale)))
    target_h = max(1, int(round(img_h * scale)))

    ip = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image_bgr, (target_w, target_h), interpolation=ip)

    x0 = (target_w - canvas_w) // 2
    y0 = (target_h - canvas_h) // 2
    x0 = max(0, min(x0, max(0, target_w - canvas_w)))
    y0 = max(0, min(y0, max(0, target_h - canvas_h)))
    return resized[y0 : y0 + canvas_h, x0 : x0 + canvas_w].copy()


def run_pipeline(
    demo_image_bgr: np.ndarray,
    original_image_bgr: np.ndarray,
    canvas_size_token: str,
    reader: Optional[object] = None,
    dpi: int = 100,
    perform_ocr: bool = True,
) -> ProcessingResult:
    """内容先纠正（SIFT）→ 物理强匹配画布英寸横竖；无 XOR，不以原图未纠偏尺寸参与判定。"""
    if perform_ocr:
        raw_text, parsed_size = ocr_size_from_demo_bottom(demo_image_bgr, reader=reader)
    else:
        raw_text, parsed_size = "", None

    # 示意图：长边 400 探测 + 边缘向内收缩 → 仅用于 SIFT 参照（去边后的画面核心），不取原图 shape 参与判定
    probe_long_edge_px = 400
    demo_probe = _resize_long_edge_bgr(demo_image_bgr, probe_long_edge_px)
    pl, pt, pr, pb = detect_actual_content_box(demo_probe)
    ph, pw = demo_probe.shape[:2]
    if (pr - pl) * (pb - pt) < max(1, int(0.01 * pw * ph)):
        pl, pt, pr, pb = 0, 0, pw, ph

    orig_h, orig_w = demo_image_bgr.shape[:2]
    L, T, R, B = _probe_box_to_original_rect((ph, pw), (orig_h, orig_w), (pl, pt, pr, pb))
    demo_for_sift = _safe_crop_demo_core(demo_image_bgr, (L, T, R, B))

    # 马克18·步骤一：SIFT 内容纠偏（无 XOR）—— 0/90/180/270 中选最优，使原图与探测后示意图核心同向
    best_deg, _scores = sift_best_rotation_deg(demo_for_sift, original_image_bgr)
    content_aligned = rotate_image_cw(original_image_bgr, best_deg)

    # 马克18·步骤二：物理强制对齐 —— 纠偏后原图横竖 vs 画布英寸横竖；不一致则顺时针再转 90°
    # （仅用 content_aligned 的尺寸，禁止用 original_image_bgr.shape 参与判定）
    ah, aw = content_aligned.shape[:2]
    aligned_landscape = aw > ah
    cw_in, ch_in = parse_inch_wh(canvas_size_token)
    canvas_landscape = cw_in > ch_in
    composition_need_90 = aligned_landscape != canvas_landscape

    if composition_need_90:
        final_aligned = rotate_image_cw(content_aligned, 90)
    else:
        final_aligned = content_aligned

    total_deg = (best_deg + (90 if composition_need_90 else 0)) % 360

    # 与画布同向后 cover：撑满画布比例并中心裁剪，无白边
    canvas_w, canvas_h = inch_token_to_pixels(canvas_size_token, dpi=dpi)
    rendered = render_cover(final_aligned, (canvas_w, canvas_h))

    return ProcessingResult(
        ocr_raw_text=raw_text,
        ocr_size_token=parsed_size,
        content_rotation_deg=best_deg,
        production_rotation_deg=90 if composition_need_90 else 0,
        total_rotation_deg=total_deg,
        needs_align=best_deg != 0,
        needs_physical_rotate=composition_need_90,
        canvas_size_px=(canvas_w, canvas_h),
        rendered_image_bgr=rendered,
    )
