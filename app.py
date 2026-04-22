import base64
import html
import io
import os
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from Picture import run_pipeline


st.set_page_config(page_title="一键改图系统", layout="wide")

# 马克18：批量并行上限强制为 4（不按 CPU 减半）
BATCH_MAX_WORKERS_CAP = 4

# 看板：加工后预览物理 400×400、画芯长边 330（Base64）；孪生底按 demo_rows 行序斑马线 A/B；前端 200×200
# 示意图列：长边高清采样后以同尺寸显示；ZIP 仍为全尺寸成品（Cover 由 Picture.py，与此无关）
PREVIEW_ENCODE_HEIGHT_PX = 300  # resize_bgr_fixed_height 默认
PREVIEW_JPEG_QUALITY = 95
# 系统成品 ZIP 内 JPEG：元数据锁定 100 DPI（与 pipeline dpi 一致）
OUTPUT_JPEG_DPI = (100, 100)
# 仅加工后预览孪生容器（不写 ZIP）
PROCESSED_PREVIEW_CANVAS_PX = 400
PROCESSED_PREVIEW_CONTENT_LONG_EDGE_PX = 330
# 示意图：高清源长边（再经 HTML 缩至逻辑格，避免马赛克）
SCHEMATIC_PREVIEW_ENCODE_LONG_EDGE_PX = 400
# 表格内两预览列逻辑显示边长（像素，与 <img> style 一致）
PREVIEW_GRID_DISPLAY_PX = 200
# 加工后预览斑马线：Color_A RGB(231,219,196)、Color_B RGB(220,208,185) → OpenCV BGR；偶数行 A / 奇数行 B
ZEBRA_PREVIEW_AMBIENT_A_BGR: Tuple[int, int, int] = (196, 219, 231)
ZEBRA_PREVIEW_AMBIENT_B_BGR: Tuple[int, int, int] = (185, 208, 220)
# 行高与预览逻辑格对齐；文字列垂直居中
TABLE_CELL_ROW_HEIGHT_PX = 200
PREVIEW_COL_MAX_WIDTH_PX = 200
# 左预览 | 文本×3 | 右预览 | 状态 —— 两列预览等宽，逻辑显示均为 PREVIEW_GRID_DISPLAY_PX
COL_WEIGHT_PREVIEW = 1.32
COL_WEIGHT_TEXT = 1.0
TABLE_COL_WEIGHTS: List[float] = [
    COL_WEIGHT_PREVIEW,
    COL_WEIGHT_TEXT,
    COL_WEIGHT_TEXT,
    COL_WEIGHT_TEXT,
    COL_WEIGHT_PREVIEW,
    COL_WEIGHT_TEXT,
]


def normalize_size_token(text: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*[xX×]\s*(\d+(?:\.\d+)?)", str(text))
    if not match:
        return None
    return f"{match.group(1)}x{match.group(2)}"


def extract_order_item_id(filename: str) -> Tuple[str, str]:
    base = os.path.splitext(os.path.basename(filename))[0]

    if "_" in base:
        order_id, item_id = base.split("_", 1)
        order_id = order_id.strip()
        item_id = item_id.strip()
        return order_id, item_id

    return base, ""


def normalize_order_key(order_id: str, item_id: str) -> str:
    o = order_id.strip().lower()
    i = item_id.strip().lower()
    return f"{o}_{i}"


def normalize_sku_key(sku: str) -> str:
    return str(sku).strip().lower()


def make_unique_key(order_id: str, item_id: str) -> str:
    """与示意图文件名解析一致：一行一图，禁止与 SKU 混用。"""
    return f"{str(order_id).strip()}_{str(item_id).strip()}"


def to_bgr(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("无法解码图片，请确认文件格式正确。")
    return image


def bgr_to_rgb_for_streamlit(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def finalize_image_with_dpi(cv2_img_bgr: np.ndarray) -> bytes:
    """OpenCV BGR → PIL JPEG 字节流，并写入 100 DPI（JFIF/EXIF）。"""
    img_rgb = cv2.cvtColor(cv2_img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil_img.save(
        buf,
        format="JPEG",
        quality=PREVIEW_JPEG_QUALITY,
        dpi=OUTPUT_JPEG_DPI,
    )
    return buf.getvalue()


def zebra_preview_ambient_bgr(zebra_row_index: int) -> Tuple[int, int, int]:
    """偶数行 Color_A，奇数行 Color_B（与 df.iterrows 默认 0,1,2… 及 demo_rows 顺序对齐）。"""
    return (
        ZEBRA_PREVIEW_AMBIENT_A_BGR
        if zebra_row_index % 2 == 0
        else ZEBRA_PREVIEW_AMBIENT_B_BGR
    )


def pack_processed_preview_bgr(
    image_bgr: np.ndarray,
    ambient_bgr: Tuple[int, int, int],
) -> np.ndarray:
    """
    仅「加工后预览」列：400×400 斑马线孪生底 ambient_bgr，成品等比缩放使长边 = PROCESSED_PREVIEW_CONTENT_LONG_EDGE_PX（INTER_AREA），居中贴合。
    不用于示意图列、不用于 ZIP。
    """
    canvas_n = PROCESSED_PREVIEW_CANVAS_PX
    lg_target = PROCESSED_PREVIEW_CONTENT_LONG_EDGE_PX
    bg = ambient_bgr
    if image_bgr is None or image_bgr.size == 0:
        return np.full((canvas_n, canvas_n, 3), bg, dtype=np.uint8)
    h, w = image_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return np.full((canvas_n, canvas_n, 3), bg, dtype=np.uint8)
    long_edge = max(w, h)
    scale = lg_target / float(long_edge)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    ip = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image_bgr, (nw, nh), interpolation=ip)
    canvas = np.full((canvas_n, canvas_n, 3), bg, dtype=np.uint8)
    x0 = (canvas_n - nw) // 2
    y0 = (canvas_n - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def get_standardized_preview(
    image_bgr: np.ndarray,
    ambient_bgr: Tuple[int, int, int],
) -> str:
    """加工后预览列专用：400×400 孪生底（斑马线 A/B）→ Base64 JPEG Q=95（不写 ZIP）。"""
    canvas = pack_processed_preview_bgr(image_bgr, ambient_bgr)
    ok, enc = cv2.imencode(
        ".jpg",
        canvas,
        [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_JPEG_QUALITY],
    )
    if not ok:
        raise RuntimeError("预览虚拟包装 JPEG 编码失败")
    return base64.b64encode(enc.tobytes()).decode("ascii")


def get_standardized_preview_from_jpeg_bytes(
    jpeg_bytes: bytes,
    ambient_bgr: Tuple[int, int, int],
) -> str:
    """从与 ZIP 同源的成品 JPEG 解码后再做 400×400 孪生包装（ZIP 字节流本身不变）。"""
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法解码成品 JPEG")
    return get_standardized_preview(img, ambient_bgr)


def resize_bgr_fixed_height(
    image_bgr: np.ndarray,
    target_h: int = PREVIEW_ENCODE_HEIGHT_PX,
) -> np.ndarray:
    """将图缩放到目标高度（保持宽高比）；缩小用 INTER_AREA，放大用 INTER_LINEAR。"""
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    h, w = image_bgr.shape[:2]
    if h <= 0:
        return image_bgr
    if h == target_h:
        return image_bgr
    scale = target_h / float(h)
    tw = max(1, int(round(w * scale)))
    ip = cv2.INTER_AREA if target_h < h else cv2.INTER_LINEAR
    return cv2.resize(image_bgr, (tw, target_h), interpolation=ip)


def resize_bgr_long_edge(image_bgr: np.ndarray, long_edge: int) -> np.ndarray:
    """等比缩放使长边 = long_edge；缩小用 INTER_AREA，放大用 INTER_LINEAR。"""
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    h, w = image_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return image_bgr
    lg = max(w, h)
    if lg == long_edge:
        return image_bgr
    scale = long_edge / float(lg)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    ip = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image_bgr, (nw, nh), interpolation=ip)


def bgr_to_jpeg_base64(image_bgr: np.ndarray) -> str:
    ok, enc = cv2.imencode(
        ".jpg",
        image_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_JPEG_QUALITY],
    )
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return base64.b64encode(enc.tobytes()).decode("ascii")


def preview_img_html_from_b64_jpeg(b64_jpeg: str) -> str:
    """逻辑格固定 PREVIEW_GRID_DISPLAY_PX（默认 200×200），源图为 400px 级 Base64（data URI）。"""
    d = PREVIEW_GRID_DISPLAY_PX
    return (
        f'<img src="data:image/jpeg;base64,{b64_jpeg}" '
        f'style="height:{d}px;width:{d}px;object-fit:contain;" alt="" />'
    )


def sync_demo_images_by_unique_key() -> None:
    """unique_key = order_id_item_id → 示意图字节；一行一键，杜绝 SKU 张冠李戴。"""
    m: Dict[str, bytes] = {}
    for row in st.session_state.demo_rows:
        oid = str(row.get("order_id", "")).strip()
        iid = str(row.get("item_id", "")).strip()
        if not oid or not iid:
            continue
        uk = make_unique_key(oid, iid)
        b = row.get("demo_bytes")
        if b:
            m[uk] = b
    st.session_state.demo_images = m


def sync_matched_df_from_demo_rows() -> None:
    """从 demo_rows 同步监控表（含 unique_key、base64 预览与加工状态）。"""
    rows: List[Dict[str, Any]] = []
    for r in st.session_state.demo_rows:
        note = (r.get("last_process_note") or "").strip()
        if note.startswith("✅"):
            status = "✅ 已完成"
        elif note.startswith("❌"):
            status = note[:200]
        else:
            status = "待加工"
        oid = str(r.get("order_id", "")).strip()
        iid = str(r.get("item_id", "")).strip()
        unique_key = make_unique_key(oid, iid) if oid and iid else ""
        rows.append(
            {
                "uid": r.get("uid", ""),
                "unique_key": unique_key,
                "order_id": r.get("order_id", ""),
                "item_id": r.get("item_id", ""),
                "sku": r.get("sku", "未匹配"),
                "canvas_inch": r.get("canvas_size_token", "未匹配"),
                "processed_preview_b64": r.get("processed_preview_thumb_b64"),
                "process_status": status,
            }
        )
    st.session_state.matched_df = pd.DataFrame(rows)


def load_preset_table(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx"):
        try:
            return pd.read_excel(uploaded_file)
        except ImportError as exc:
            raise RuntimeError("读取 XLSX 需要 openpyxl，请安装后重试。") from exc
    raise ValueError("仅支持 CSV/XLSX 预设表。")


def _find_canvas_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        name = str(c).strip()
        low = name.lower()
        if name == "画布尺寸" or low == "画布尺寸":
            return c
    return None


def _find_sku_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if str(c).strip().lower() == "sku":
            return c
    return None


def load_order_list_map(uploaded_file) -> Dict[str, str]:
    """
    《订单列表》：A 列 order-id，B 列 order-item-id，J 列 SKU（列位置 0/1/9）。
    仅建立 Order+Item ID → SKU，与 SKU 预设总表解耦。
    """
    df = load_preset_table(uploaded_file)
    if df.shape[1] < 10:
        raise ValueError("订单列表需至少包含 A 列到 J 列（共 10 列），请检查文件。")
    out: Dict[str, str] = {}
    for _, row in df.iterrows():
        oid = row.iloc[0]
        iid = row.iloc[1]
        sku = row.iloc[9]
        if pd.isna(oid) or pd.isna(iid):
            continue
        oid_s = str(oid).strip()
        iid_s = str(iid).strip()
        if not oid_s or not iid_s:
            continue
        key = normalize_order_key(oid_s, iid_s)
        if pd.isna(sku):
            continue
        sku_s = str(sku).strip()
        if sku_s:
            out[key] = sku_s
    if not out:
        raise ValueError("未从订单列表中解析到任何有效行。")
    return out


def load_sku_canvas_preset_map(uploaded_file) -> Dict[str, str]:
    """
    《SKU 预设总表》：列「SKU」与「画布尺寸」（如 24x36）。
    持久化于 st.session_state.sku_canvas_map，换人订单列表后仍可沿用。
    """
    df = load_preset_table(uploaded_file)
    col_sku = _find_sku_column(df)
    col_canvas = _find_canvas_column(df)
    if col_sku is None or col_canvas is None:
        raise ValueError("SKU 预设总表必须包含「SKU」与「画布尺寸」列。")
    out: Dict[str, str] = {}
    for _, row in df.iterrows():
        sku = row[col_sku]
        canvas_val = row[col_canvas]
        if pd.isna(sku):
            continue
        k = normalize_sku_key(str(sku))
        if not k:
            continue
        if pd.isna(canvas_val):
            continue
        raw = str(canvas_val).strip()
        if not raw:
            continue
        out[k] = normalize_size_token(raw) or raw
    if not out:
        raise ValueError("未从 SKU 预设总表中解析到任何有效 SKU→画布 映射。")
    return out


def ensure_state() -> None:
    if "demo_rows" not in st.session_state:
        st.session_state.demo_rows = []
    if "original_map" not in st.session_state:
        st.session_state.original_map = {}
    if "order_map" not in st.session_state:
        st.session_state.order_map = {}
    # sku_canvas_map 仅在上传《SKU总表》时整体替换；换订单表/换图不重置
    if "sku_canvas_map" not in st.session_state:
        st.session_state.sku_canvas_map = {}
    if "data_lookup_revision" not in st.session_state:
        st.session_state.data_lookup_revision = 0
    if "rows" not in st.session_state:
        st.session_state.rows = {}
    if "matched_df" not in st.session_state:
        st.session_state.matched_df = pd.DataFrame()
    if "demo_images" not in st.session_state:
        st.session_state.demo_images = {}
    for row in st.session_state.demo_rows:
        if "processed_preview_thumb_b64" not in row:
            row["processed_preview_thumb_b64"] = None


def apply_data_driven_sizes() -> None:
    """数据链路：订单表 Order+Item→SKU；SKU总表 SKU→画布尺寸。"""
    order_map: Dict[str, str] = st.session_state.get("order_map") or {}
    sku_canvas: Dict[str, str] = st.session_state.get("sku_canvas_map") or {}
    rev = int(st.session_state.get("data_lookup_revision", 0))

    for row in st.session_state.demo_rows:
        oid = str(row.get("order_id", "")).strip()
        iid = str(row.get("item_id", "")).strip()
        key = normalize_order_key(oid, iid) if oid and iid else ""
        prev_key = row.get("_applied_lookup_key")
        prev_rev = row.get("_applied_data_lookup_revision", -1)

        if key == prev_key and rev == prev_rev:
            continue

        row["_applied_lookup_key"] = key
        row["_applied_data_lookup_revision"] = rev

        if not oid or not iid:
            row["canvas_size_token"] = "未匹配"
            row["sku"] = "未匹配"
        else:
            sku_val = order_map.get(key)
            if not sku_val:
                row["canvas_size_token"] = "未匹配"
                row["sku"] = "未匹配"
            else:
                row["sku"] = sku_val
                sk = normalize_sku_key(sku_val)
                row["canvas_size_token"] = sku_canvas.get(sk) or "未匹配"
        row["ocr_size"] = row["canvas_size_token"]
        row["recognized_size"] = row["canvas_size_token"]


def upsert_demo_rows(files) -> None:
    if not files:
        return
    existing_by_name = {row["demo_name"]: row for row in st.session_state.demo_rows}
    for f in files:
        order_id, item_id = extract_order_item_id(f.name)
        data = f.getvalue()
        prev = existing_by_name.get(f.name)
        # 同示意图、同字节：禁止覆盖行（否则每次 rerun / 点下载都会清空 Base64 与加工备注）
        if prev is not None and prev.get("demo_bytes") == data:
            continue
        row = {
            "uid": f"demo::{f.name}",
            "demo_name": f.name,
            "demo_bytes": data,
            "order_id": order_id,
            "item_id": item_id,
            "canvas_size_token": "",
            "ocr_size": "",
            "recognized_size": "",
            "sku": "",
            "_applied_lookup_key": None,
            "_applied_data_lookup_revision": -1,
            "matched_original_name": None,
            "matched_original_bytes": None,
            "match_status": "❌未找到原图",
            "last_process_note": "",
            "processed_preview_thumb_b64": None,
            "delete": False,
        }
        existing_by_name[f.name] = row
    st.session_state.demo_rows = list(existing_by_name.values())


def upsert_original_map(files) -> None:
    if not files:
        return
    data_map = {}
    for f in files:
        order_id, item_id = extract_order_item_id(f.name)
        key = normalize_order_key(order_id, item_id)
        data_map[key] = {"name": f.name, "bytes": f.getvalue()}
    st.session_state.original_map = data_map


def refresh_match_status() -> None:
    original_map = st.session_state.original_map
    for row in st.session_state.demo_rows:
        key = normalize_order_key(row["order_id"], row["item_id"])
        matched = original_map.get(key)
        if matched:
            row["matched_original_name"] = matched["name"]
            row["matched_original_bytes"] = matched["bytes"]
            row["match_status"] = "✅已匹配原图"
        else:
            row["matched_original_name"] = None
            row["matched_original_bytes"] = None
            row["match_status"] = "❌未找到原图"


def sync_rows_to_display_dict() -> None:
    st.session_state.rows = {}
    for row in st.session_state.demo_rows:
        key = row.get("uid") or row.get("demo_name", "")
        if key:
            st.session_state.rows[key] = row


def render_upload_dashboard() -> None:
    """
    A/B 下方：示意图/原图计数 + 数量对齐预警。
    C/D 下方：订单表与 SKU总表的加载条数（会话内 SKU总表持久）。
    """
    sync_rows_to_display_dict()
    n_demo = len(st.session_state.rows)
    n_orig = len(st.session_state.original_map)
    n_order = len(st.session_state.get("order_map") or {})
    n_sku_preset = len(st.session_state.get("sku_canvas_map") or {})

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        st.caption(f"📊 已解析示意图：**{n_demo}** 张")
    with c2:
        st.caption(f"🖼️ 已解析原图：**{n_orig}** 张")
    with c3:
        if n_order > 0:
            st.caption(f"📋 订单表：已加载 **{n_order}** 条 Order→SKU")
        else:
            st.caption("📋 订单表：未加载")
    with c4:
        if n_sku_preset > 0:
            st.caption(f"🏷️ SKU总表：已加载 **{n_sku_preset}** 条 SKU→画布")
        else:
            st.caption("🏷️ SKU总表：未加载")

    if n_demo == n_orig:
        st.success("✅ 示意图与原图数量匹配，可放心加工。")
    else:
        st.warning(
            f"⚠️ 数量不匹配！示意图 ({n_demo}) vs 原图 ({n_orig})。请检查是否有漏传或文件名 ID 错误。"
        )


def qc_columns(weights: List[float]):
    """行内列：垂直居中对齐（与 TABLE_CELL_ROW_HEIGHT_PX 一致）。"""
    try:
        return st.columns(weights, vertical_alignment="center", gap="small")
    except TypeError:
        try:
            return st.columns(weights, vertical_alignment="center")
        except TypeError:
            return st.columns(weights)


def _vcenter_text_cell(markdown_slot: Any, text: str) -> None:
    """Order ID / SKU / 画布尺寸：格内水平、垂直居中。"""
    safe = html.escape(str(text))
    h = TABLE_CELL_ROW_HEIGHT_PX
    markdown_slot.markdown(
        f'<div style="min-height:{h}px;height:{h}px;display:flex;align-items:center;justify-content:center;'
        f'margin:0;padding:0;width:100%;box-sizing:border-box;text-align:center">{safe}</div>',
        unsafe_allow_html=True,
    )


def _vcenter_status_cell(markdown_slot: Any, status_text: str) -> None:
    """加工结果：格内水平、垂直居中；成功/失败用颜色区分。"""
    safe = html.escape(str(status_text))
    h = TABLE_CELL_ROW_HEIGHT_PX
    if status_text.startswith("✅"):
        markdown_slot.markdown(
            f'<div style="min-height:{h}px;height:{h}px;display:flex;align-items:center;justify-content:center;'
            f'margin:0;padding:0;width:100%;box-sizing:border-box;color:#0f5132;font-weight:600">'
            f"{safe}</div>",
            unsafe_allow_html=True,
        )
    elif status_text.startswith("❌"):
        markdown_slot.markdown(
            f'<div style="min-height:{h}px;height:{h}px;display:flex;align-items:center;justify-content:center;'
            f'margin:0;padding:0;width:100%;box-sizing:border-box;color:#842029;font-weight:600">'
            f"{safe}</div>",
            unsafe_allow_html=True,
        )
    else:
        _vcenter_text_cell(markdown_slot, status_text)


def inject_spreadsheet_grid_css() -> None:
    """行高随 200px 预览撑开、1px 浅灰格线；预览由 HTML img 内联尺寸控制，勿用 !important 抢宽高。"""
    hrow = TABLE_CELL_ROW_HEIGHT_PX + 8
    st.markdown(
        f"""
<style>
section[data-testid="stMain"] div[data-testid="column"] {{
    border: 1px solid #e6e9ef;
    padding: 2px 8px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    box-sizing: border-box;
    min-height: {hrow}px;
    overflow-x: hidden;
    overflow-y: auto;
    text-align: center;
}}
section[data-testid="stMain"] div[data-testid="column"] img {{
    display: block;
    margin: auto;
}}
section[data-testid="stMain"] div[data-testid="column"] p {{
    margin: 0 auto;
    text-align: center;
}}
</style>
""",
        unsafe_allow_html=True,
    )


def render_table() -> None:
    sync_rows_to_display_dict()
    sync_demo_images_by_unique_key()
    sync_matched_df_from_demo_rows()

    if not st.session_state.demo_rows:
        st.info("尚未上传任何示意图。")
        return

    st.markdown("### 数据展示表格（电子表格 · 生产报表）")

    # 示意图预览 | Order ID | SKU | 画布尺寸(inch) | 加工后预览 | 加工结果（双预览列等宽孪生）
    col_weights = TABLE_COL_WEIGHTS
    header = qc_columns(col_weights)
    header[0].markdown("**示意图预览**")
    header[1].markdown("**Order ID**")
    header[2].markdown("**SKU**")
    header[3].markdown("**画布尺寸(inch)**")
    header[4].markdown("**加工后预览**")
    header[5].markdown("**加工结果**")

    demo_map: Dict[str, bytes] = st.session_state.get("demo_images") or {}
    df = st.session_state.matched_df

    for index, rdf in df.iterrows():
        row_cols = qc_columns(col_weights)

        unique_key_str = ""
        if "unique_key" in rdf.index and pd.notna(rdf["unique_key"]):
            unique_key_str = str(rdf["unique_key"]).strip()

        if "sku" in rdf.index and pd.notna(rdf["sku"]):
            sku_display = str(rdf["sku"])
        else:
            sku_display = "未匹配"

        oid = (
            str(rdf["order_id"])
            if "order_id" in rdf.index and pd.notna(rdf["order_id"])
            else ""
        )
        cinch = (
            str(rdf["canvas_inch"])
            if "canvas_inch" in rdf.index and pd.notna(rdf["canvas_inch"])
            else "未匹配"
        )

        if unique_key_str and unique_key_str in demo_map:
            bytes_for_this_row = demo_map[unique_key_str]
            try:
                schematic_bgr = to_bgr(bytes_for_this_row)
                hi_bgr = resize_bgr_long_edge(
                    schematic_bgr, SCHEMATIC_PREVIEW_ENCODE_LONG_EDGE_PX
                )
                sch_b64 = bgr_to_jpeg_base64(hi_bgr)
                row_cols[0].markdown(
                    preview_img_html_from_b64_jpeg(sch_b64),
                    unsafe_allow_html=True,
                )
            except Exception:
                row_cols[0].markdown(
                    f'<div style="height:{TABLE_CELL_ROW_HEIGHT_PX}px;display:flex;align-items:center;'
                    f'justify-content:center;color:#9ca3af;width:100%">未上传</div>',
                    unsafe_allow_html=True,
                )
        else:
            row_cols[0].markdown(
                f'<div style="height:{TABLE_CELL_ROW_HEIGHT_PX}px;display:flex;align-items:center;'
                f'justify-content:center;color:#9ca3af;width:100%">未上传</div>',
                unsafe_allow_html=True,
            )

        _vcenter_text_cell(row_cols[1], oid)
        _vcenter_text_cell(row_cols[2], sku_display)
        _vcenter_text_cell(row_cols[3], cinch)

        b64_val = rdf["processed_preview_b64"] if "processed_preview_b64" in rdf.index else None
        if b64_val is not None and pd.notna(b64_val) and isinstance(b64_val, str) and len(b64_val) > 0:
            try:
                raw_jpg = base64.b64decode(b64_val)
                arr_jpg = np.frombuffer(raw_jpg, dtype=np.uint8)
                if cv2.imdecode(arr_jpg, cv2.IMREAD_COLOR) is None:
                    raise ValueError("无效预览图")
                row_cols[4].markdown(
                    preview_img_html_from_b64_jpeg(b64_val),
                    unsafe_allow_html=True,
                )
            except Exception:
                row_cols[4].markdown(
                    f'<div style="height:{TABLE_CELL_ROW_HEIGHT_PX}px;max-width:{PREVIEW_COL_MAX_WIDTH_PX}px;'
                    f'display:flex;align-items:center;justify-content:center;margin:0 auto;color:#9ca3af;">待加工</div>',
                    unsafe_allow_html=True,
                )
        else:
            row_cols[4].markdown(
                f'<div style="height:{TABLE_CELL_ROW_HEIGHT_PX}px;max-width:{PREVIEW_COL_MAX_WIDTH_PX}px;'
                f'display:flex;align-items:center;justify-content:center;margin:0 auto;color:#9ca3af;">待加工</div>',
                unsafe_allow_html=True,
            )

        if "process_status" in rdf.index and pd.notna(rdf["process_status"]):
            status_text = str(rdf["process_status"])
        else:
            status_text = "待加工"

        _vcenter_status_cell(row_cols[5], status_text[:200])


def refresh_table_slot(table_slot: Any) -> None:
    """批量加工循环内调用，实现预览与状态实时刷新。"""
    sync_matched_df_from_demo_rows()
    table_slot.empty()
    with table_slot.container():
        render_table()


def canvas_token_for_pipeline(row: Dict[str, object]) -> str:
    raw = (row.get("canvas_size_token") or row.get("ocr_size") or row.get("recognized_size") or "").strip()
    if not raw or raw == "未匹配":
        raise ValueError(
            f"{row['demo_name']} 画布尺寸未匹配，请检查订单表与 SKU总表。"
        )
    n = normalize_size_token(raw)
    return n or raw


def process_single_image(
    row: Dict[str, object],
    original_map: Dict[str, Dict[str, bytes]],
) -> Tuple[str, bytes, bytes]:
    key = normalize_order_key(str(row["order_id"]), str(row["item_id"]))
    canvas_size_token = canvas_token_for_pipeline(row)

    matched_name = row.get("matched_original_name")
    matched_bytes = row.get("matched_original_bytes")
    if not matched_name or not matched_bytes:
        for map_key, item in original_map.items():
            if map_key == key:
                matched_name = item["name"]
                matched_bytes = item["bytes"]
                break

    if not matched_name or not matched_bytes:
        raise ValueError(f"{row['demo_name']} 未找到匹配原图。")

    demo_bytes = row.get("demo_bytes")
    if demo_bytes is not None and matched_bytes == demo_bytes:
        raise ValueError(
            f"{row['demo_name']}：匹配到的原图字节与示意图完全相同，成品会变成示意图；"
            "请检查原图包是否误传了与示意图相同的文件。"
        )

    demo_bgr = to_bgr(row["demo_bytes"])
    original_bgr = to_bgr(matched_bytes)
    result = run_pipeline(
        demo_image_bgr=demo_bgr,
        original_image_bgr=original_bgr,
        canvas_size_token=canvas_size_token,
        dpi=100,
        perform_ocr=False,
    )
    try:
        # ZIP 与看板源共用同一套成品 JPEG 字节（100 DPI）；看板 Base64 仅在批完成后主线程内做 400×400 孪生包装
        zip_bytes = finalize_image_with_dpi(result.rendered_image_bgr)
    except Exception as exc:
        raise RuntimeError(f"导出失败：{row['demo_name']}") from exc
    out_name = str(matched_name)
    return out_name, zip_bytes, zip_bytes


def _clear_batch_process_notes() -> None:
    for row in st.session_state.demo_rows:
        row["last_process_note"] = ""
        row["processed_preview_thumb_b64"] = None


def run_batch_and_build_zip(
    status_slot: Any,
    progress_slot: Any,
    table_slot: Optional[Any] = None,
) -> Tuple[bytes, int, int, int]:
    """
    全量揭幕：循环内仅更新 batch_status / batch_progress，不刷新表格、不逐行写预览。
    ZIP 构建完成后，一次性将缩略图与「✅/❌」写回 demo_rows，再同步 matched_df 并可选刷新 table_slot。
    并行数 = min(批次数, BATCH_MAX_WORKERS_CAP=4)。
    返回：(zip_bytes, success_count, fail_count, total_count)
    """
    rows = [r for r in st.session_state.demo_rows if r.get("match_status", "").startswith("✅")]
    if not rows:
        raise ValueError("没有可加工数据：请确保至少有一行已匹配原图。")
    if not st.session_state.sku_canvas_map:
        raise ValueError("未加载 SKU总表或无可用的 SKU→画布 映射，无法加工。")

    _clear_batch_process_notes()

    total = len(rows)
    memory_zip = io.BytesIO()
    max_workers = max(1, min(total, BATCH_MAX_WORKERS_CAP))

    status_slot.markdown(
        f"🚀 正在全速生产：已完成 **0** / 总计 **{total}** 张（**0.0%**）"
    )
    progress_slot.progress(0.0, text=f"准备中（{max_workers} 线程）…")

    results: List[Tuple[str, bytes]] = []
    success_count = 0
    fail_count = 0
    done = 0
    pending_by_uid: Dict[str, Tuple[Optional[bytes], str]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_row = {
            executor.submit(
                process_single_image,
                row,
                st.session_state.original_map,
            ): row
            for row in rows
        }
        for future in as_completed(future_to_row):
            row = future_to_row[future]
            uid = str(row.get("uid", ""))
            done += 1
            pct = 100.0 * done / total
            short = str(row.get("demo_name", ""))[:40]
            status_slot.markdown(
                f"🚀 正在全速生产：已完成 **{done}** / 总计 **{total}** 张（**{pct:.1f}%**）"
            )
            progress_slot.progress(
                done / total,
                text=f"并行 {max_workers} 线程 · 当前：{short}…",
            )
            try:
                out_name, data, thumb_bytes = future.result()
                results.append((out_name, data))
                success_count += 1
                pending_by_uid[uid] = (thumb_bytes, "✅ 已完成")
            except Exception as exc:
                fail_count += 1
                err = str(exc)[:180]
                pending_by_uid[uid] = (None, f"❌ {err}")

    progress_slot.progress(1.0, text="写入 ZIP…")
    with zipfile.ZipFile(memory_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for out_name, data in results:
            zf.writestr(out_name, data)

    # 全量揭幕：ZIP 就绪后再写回会话中的行数据与 Base64；孪生底按 demo_rows 行序斑马线
    uid_to_zebra_index: Dict[str, int] = {}
    for zi, dr in enumerate(st.session_state.demo_rows):
        uid_to_zebra_index[str(dr.get("uid", ""))] = zi

    for row in rows:
        uid = str(row.get("uid", ""))
        if uid not in pending_by_uid:
            continue
        thumb_bytes, note = pending_by_uid[uid]
        row["last_process_note"] = note
        if thumb_bytes is not None:
            z_i = uid_to_zebra_index.get(uid, 0)
            ambient = zebra_preview_ambient_bgr(z_i)
            row["processed_preview_thumb_b64"] = get_standardized_preview_from_jpeg_bytes(
                thumb_bytes, ambient
            )
        else:
            row["processed_preview_thumb_b64"] = None

    if table_slot is not None:
        refresh_table_slot(table_slot)
    else:
        sync_matched_df_from_demo_rows()

    progress_slot.empty()
    status_slot.empty()
    memory_zip.seek(0)
    data = memory_zip.getvalue()
    return data, success_count, fail_count, total


def main() -> None:
    ensure_state()
    st.title("一键改图系统")
    inject_spreadsheet_grid_css()

    st.markdown("### 顶层控制区")
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        demo_files = st.file_uploader(
            "上传《示意图》",
            type=["png", "jpg", "jpeg", "webp", "bmp"],
            accept_multiple_files=True,
            key="uploader_demo_schematics",
        )
        st.caption("(A: 匹配内容)")
    with col_b:
        original_files = st.file_uploader(
            "上传《原图》",
            type=["png", "jpg", "jpeg", "webp", "bmp"],
            accept_multiple_files=True,
            key="uploader_original_material",
        )
        st.caption("(B: 加工素材)")
    with col_c:
        order_list_file = st.file_uploader(
            "上传《订单表》",
            type=["csv", "xlsx"],
            accept_multiple_files=False,
            key="uploader_order_list",
        )
        st.caption("(C: 查找SKU)")
    with col_d:
        sku_preset_file = st.file_uploader(
            "上传《SKU总表》",
            type=["csv", "xlsx"],
            accept_multiple_files=False,
            key="uploader_sku_preset_master",
        )
        st.caption("(D: 固定预设)")

    if demo_files:
        upsert_demo_rows(demo_files)
    if original_files:
        upsert_original_map(original_files)

    if order_list_file is not None:
        try:
            st.session_state.order_map = load_order_list_map(order_list_file)
            st.session_state.data_lookup_revision = int(st.session_state.get("data_lookup_revision", 0)) + 1
            st.success(f"订单表已载入：**{len(st.session_state.order_map)}** 条。")
        except Exception as exc:
            st.error(f"订单表加载失败：{exc}")

    if sku_preset_file is not None:
        try:
            st.session_state.sku_canvas_map = load_sku_canvas_preset_map(sku_preset_file)
            st.session_state.data_lookup_revision = int(st.session_state.get("data_lookup_revision", 0)) + 1
            st.success(
                f"SKU总表已写入会话（持久）：**{len(st.session_state.sku_canvas_map)}** 条 SKU→画布。"
            )
        except Exception as exc:
            st.error(f"SKU总表加载失败：{exc}")

    refresh_match_status()
    apply_data_driven_sizes()

    render_upload_dashboard()

    st.caption(
        "订单表→SKU；SKU总表→画布。Picture 物理对齐（无 XOR）；批量最多 4 线程；加工中仅进度条、完成后全表揭幕。"
    )

    table_slot = st.empty()
    with table_slot.container():
        render_table()

    st.markdown("### 执行区")
    ex1, ex2 = st.columns(2)
    with ex1:
        start_batch = st.button("开始批量加工", type="primary")
    with ex2:
        clear_all = st.button("清除所有文件")

    if clear_all:
        st.session_state.demo_rows = []
        st.session_state.original_map = {}
        st.session_state.rows = {}
        st.session_state.matched_df = pd.DataFrame()
        st.session_state.demo_images = {}
        for k in ("last_zip", "last_batch_ok", "last_batch_fail", "last_batch_total"):
            st.session_state.pop(k, None)
        ensure_state()
        st.rerun()

    if start_batch:
        try:
            batch_status = st.empty()
            batch_progress = st.empty()
            zip_bytes, ok_n, fail_n, total_n = run_batch_and_build_zip(
                batch_status,
                batch_progress,
                table_slot=table_slot,
            )
            st.session_state.last_zip = zip_bytes
            st.session_state.last_batch_ok = ok_n
            st.session_state.last_batch_fail = fail_n
            st.session_state.last_batch_total = total_n

            st.success(
                f"✅ 生产任务圆满完成！共计成功处理 **{ok_n}** 张图片。"
            )
            if fail_n > 0:
                st.error(
                    f"❌ 失败：**{fail_n}** 张（详情请见下方表格「加工结果」列）。"
                )
        except Exception as exc:
            st.error(f"加工失败：{exc}")
            sync_matched_df_from_demo_rows()
            if table_slot is not None:
                refresh_table_slot(table_slot)

    # 下载仅触发浏览器取货，不修改 session_state；加工结果以 demo_rows 为唯一真源，经 sync 写入 matched_df
    if st.session_state.get("last_zip"):
        st.download_button(
            "下载加工结果 ZIP",
            data=st.session_state.last_zip,
            file_name="mark10_batch_output.zip",
            mime="application/zip",
            key="download_batch_zip_mark10",
        )


if __name__ == "__main__":
    main()
