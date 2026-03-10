import sys
import os
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
import numpy as np
import cv2
from PIL import Image
from numba import njit, prange
from pillow_heif import register_heif_opener

register_heif_opener()

def get_target_directory() -> str:
    root = tk.Tk()
    root.withdraw()
    selected_directory = filedialog.askdirectory(title="Select input/output folder")
    root.destroy()
    if not selected_directory:
        sys.exit()
    return selected_directory

# kurvagyors matek

@njit(fastmath=True, cache=True)
def get_luminance(r, g, b):
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


@njit(fastmath=True, cache=True)
def get_hue(r, g, b):
    r_f = r / 255.0
    g_f = g / 255.0
    b_f = b / 255.0
    mx = max(r_f, max(g_f, b_f))
    mn = min(r_f, min(g_f, b_f))
    df = mx - mn
    if mx == mn:
        h = 0.0
    elif mx == r_f:
        h = (60 * ((g_f - b_f) / df) + 360) % 360
    elif mx == g_f:
        h = (60 * ((b_f - r_f) / df) + 120) % 360
    elif mx == b_f:
        h = (60 * ((r_f - g_f) / df) + 240) % 360
    return h / 360.0


@njit(fastmath=True, cache=True)
def get_intensity(r, g, b):
    return (float(r) + float(g) + float(b)) / 3.0


@njit(parallel=True, fastmath=True, cache=True)
def compute_sort_keys(img_arr, sort_mode):
    H, W, _ = img_arr.shape
    keys = np.zeros((H, W), dtype=np.float32)
    for y in prange(H):
        for x in range(W):
            r, g, b = img_arr[y, x, 0], img_arr[y, x, 1], img_arr[y, x, 2]
            if sort_mode == 0:
                keys[y, x] = get_hue(r, g, b)
            elif sort_mode == 1:
                keys[y, x] = get_luminance(r, g, b)
            else:
                keys[y, x] = get_intensity(r, g, b)
    return keys


@njit(parallel=True, fastmath=True, cache=True)
def fast_sobel_mask(img_arr, threshold):
    H, W, _ = img_arr.shape
    mask = np.ones((H, W), dtype=np.bool_)
    gray = np.empty((H, W), dtype=np.float32)

    for y in prange(H):
        for x in range(W):
            gray[y, x] = 0.2989 * img_arr[y, x, 0] + 0.5870 * img_arr[y, x, 1] + 0.1140 * img_arr[y, x, 2]

    for y in prange(1, H - 1):
        for x in range(1, W - 1):
            gx = (-1 * gray[y - 1, x - 1] + 1 * gray[y - 1, x + 1] + -2 * gray[y, x - 1] + 2 * gray[y, x + 1] + -1 *
                  gray[y + 1, x - 1] + 1 * gray[y + 1, x + 1])
            gy = (-1 * gray[y - 1, x - 1] - 2 * gray[y - 1, x] - 1 * gray[y - 1, x + 1] + 1 * gray[y + 1, x - 1] + 2 *
                  gray[y + 1, x] + 1 * gray[y + 1, x + 1])
            mag = np.sqrt(gx * gx + gy * gy)

            if min(255, mag) < threshold:
                mask[y, x] = True
            else:
                mask[y, x] = False
    return mask


@njit(parallel=True, fastmath=True, cache=True)
def fast_intensity_mask(img_arr, threshold, invert):
    H, W, _ = img_arr.shape
    mask = np.empty((H, W), dtype=np.bool_)
    for y in prange(H):
        for x in range(W):
            lum = 0.2126 * img_arr[y, x, 0] + 0.7152 * img_arr[y, x, 1] + 0.0722 * img_arr[y, x, 2]
            if invert:
                mask[y, x] = (lum < threshold)
            else:
                mask[y, x] = (lum > threshold)
    return mask


@njit(parallel=True, fastmath=True, cache=True)
def _numba_pixel_sort_1d(img, keys, mask, axis, min_len, reverse):
    rows, cols, _ = img.shape
    limit_outer = rows if axis == 1 else cols
    limit_inner = cols if axis == 1 else rows

    for i in prange(limit_outer):
        start = 0
        while start < limit_inner:
            if axis == 1:
                valid = mask[i, start]
            else:
                valid = mask[start, i]

            if not valid:
                start += 1
                continue

            end = start
            while end < limit_inner:
                if axis == 1:
                    valid_end = mask[i, end]
                else:
                    valid_end = mask[end, i]
                if not valid_end: break
                end += 1

            length = end - start
            if length >= min_len:
                seg_keys = np.empty(length, dtype=np.float32)
                for k in range(length):
                    if axis == 1:
                        seg_keys[k] = keys[i, start + k]
                    else:
                        seg_keys[k] = keys[start + k, i]

                order = np.argsort(seg_keys)
                if reverse: order = order[::-1]

                temp = np.empty((length, 3), dtype=np.uint8)
                for k in range(length):
                    idx = order[k]
                    if axis == 1:
                        temp[k] = img[i, start + idx]
                    else:
                        temp[k] = img[start + idx, i]

                for k in range(length):
                    if axis == 1:
                        img[i, start + k] = temp[k]
                    else:
                        img[start + k, i] = temp[k]
            start = end

# sorbarakó wrapperek

def pixel_sort_fast(array, mask_mode, sort_mode_idx, direction,
                    threshold, min_segment_len=2, reverse=False):
    out_arr = array.copy()

    if mask_mode == "sobel":
        mask = fast_sobel_mask(out_arr, threshold)
    elif mask_mode == "bright":
        mask = fast_intensity_mask(out_arr, threshold, False)
    elif mask_mode == "dark":
        mask = fast_intensity_mask(out_arr, threshold, True)

    keys = compute_sort_keys(out_arr, sort_mode_idx)

    axis = 1 if direction == "horizontal" else 0
    _numba_pixel_sort_1d(out_arr, keys, mask, axis, min_segment_len, reverse)
    return out_arr


def pixel_sort_diagonal(array, mask_mode, sort_mode_idx, threshold, angle=45):
    img_pil = Image.fromarray(array)
    img_rot = img_pil.rotate(angle, expand=True, resample=Image.NEAREST)
    arr_rot = np.array(img_rot)
    arr_sorted = pixel_sort_fast(arr_rot, mask_mode, sort_mode_idx, "horizontal", threshold, 2, False)
    img_sorted_pil = Image.fromarray(arr_sorted)
    img_back = img_sorted_pil.rotate(-angle, expand=True, resample=Image.NEAREST)

    w_orig, h_orig = array.shape[1], array.shape[0]
    w_new, h_new = img_back.size
    left = (w_new - w_orig) / 2
    top = (h_new - h_orig) / 2
    img_final = img_back.crop((left, top, left + w_orig, top + h_orig))

    return np.array(img_final)


def pixel_sort_polar(array, mask_mode, sort_mode_idx, threshold, mode='circle'):
    H, W = array.shape[:2]
    center = (W / 2, H / 2)
    max_radius = np.sqrt((W / 2) ** 2 + (H / 2) ** 2)

    polar_img = cv2.warpPolar(array, (W, H), center, max_radius, cv2.WARP_POLAR_LINEAR)
    direction = "vertical" if mode == 'circle' else "horizontal"
    polar_sorted = pixel_sort_fast(polar_img, mask_mode, sort_mode_idx, direction, threshold, 2, False)
    final_img = cv2.warpPolar(polar_sorted, (W, H), center, max_radius,
                              cv2.WARP_POLAR_LINEAR | cv2.WARP_INVERSE_MAP)
    return final_img


def warmup_jit():
    print("Compiling Engine...")
    d = np.zeros((10, 10, 3), dtype=np.uint8)
    pixel_sort_fast(d, "sobel", 0, "horizontal", 10)
    print("Engine Ready.")

# ui

class Button:
    def __init__(self, x, y, w, h, text, callback, group_id=None, is_toggle=False):
        self.rect = (x, y, w, h)
        self.text = text
        self.callback = callback
        self.group_id = group_id
        self.is_toggle = is_toggle
        self.active = False
        self.hover = False

    def draw(self, canvas):
        x, y, w, h = self.rect
        if self.active:
            color = (0, 180, 0)
        elif self.hover:
            color = (80, 80, 80)
        else:
            color = (50, 50, 50)

        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (200, 200, 200), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        text_size = cv2.getTextSize(self.text, font, font_scale, thickness)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        cv2.putText(canvas, self.text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

    def is_inside(self, ix, iy):
        x, y, w, h = self.rect
        return x <= ix <= x + w and y <= iy <= y + h


class Slider:
    def __init__(self, x, y, w, h, min_val, max_val, initial, label):
        self.rect = (x, y, w, h)
        self.min = min_val
        self.max = max_val
        self.val = initial
        self.label = label
        self.dragging = False

    def draw(self, canvas):
        x, y, w, h = self.rect
        cv2.putText(canvas, f"{self.label}: {self.val}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (40, 40, 40), -1)
        ratio = (self.val - self.min) / (self.max - self.min)
        kx = int(x + ratio * w)
        cv2.rectangle(canvas, (kx - 5, y), (kx + 5, y + h), (0, 150, 255), -1)

    def handle_input(self, ix, iy, event):
        x, y, w, h = self.rect
        if event == cv2.EVENT_LBUTTONDOWN:
            if x <= ix <= x + w and y <= iy <= y + h:
                self.dragging = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False

        if self.dragging:
            ratio = (ix - x) / w
            ratio = max(0, min(1, ratio))
            self.val = int(self.min + ratio * (self.max - self.min))


# applikáció

class PixelSortApp:
    def __init__(self, input_path):
        self.input_path = input_path
        self.image_files = []
        self.current_index = 0

        self.ui_width = 220
        self.buttons = []
        self.sliders = []
        self.mouse_x, self.mouse_y = 0, 0
        self.history = []

        if os.path.isdir(input_path):
            exts = ('.png', '.jpg', '.jpeg', '.heic', '.heif')
            self.image_files = [
                os.path.join(input_path, f) for f in os.listdir(input_path)
                if f.lower().endswith(exts)
            ]
            self.image_files.sort()
            if not self.image_files:
                print("No images found in folder!")
                self.image_files = ["NO_IMG"]
        else:
            self.image_files = [input_path]

        self.img_orig = None
        self.img_curr = None

        self.sort_modes = ["Hue", "Luma", "Intens"]
        self.current_sort_idx = 0
        self.mask_modes = ["sobel", "bright", "dark"]
        self.current_mask_mode = "sobel"

        self.load_image_from_index(0)

        self.setup_ui()

    def load_image_from_index(self, index):
        if not self.image_files or self.image_files[0] == "NO_IMG":
            self.img_orig = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
            self.img_curr = self.img_orig.copy()
            self.canvas_h, w = self.img_curr.shape[:2]
            self.canvas_w = w + self.ui_width
            return

        self.current_index = index % len(self.image_files)
        filepath = self.image_files[self.current_index]
        print(f"Loading: {filepath}")

        try:
            pil = Image.open(filepath).convert("RGB")
            pil.thumbnail((1200, 900))
            self.img_orig = np.array(pil)
            self.img_curr = self.img_orig.copy()
            self.history = []
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            self.img_orig = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
            self.img_curr = self.img_orig.copy()

        h, w = self.img_curr.shape[:2]
        self.canvas_h = h
        self.canvas_w = w + self.ui_width

        if hasattr(self, 'buttons') and len(self.buttons) > 0:
            self.setup_ui()

    def change_image(self, delta):
        new_idx = self.current_index + delta
        self.load_image_from_index(new_idx)

    def save_state(self):
        self.history.append(self.img_curr.copy())

    def undo(self):
        if len(self.history) > 0:
            self.img_curr = self.history.pop()

    def setup_ui(self):
        self.buttons = []
        self.sliders = []

        bx = self.img_curr.shape[1] + 10
        by = 20
        bw = 200
        bh = 30

        half_w = (bw - 5) // 2
        self.buttons.append(Button(bx, by, half_w, bh, "< PREV", lambda: self.change_image(-1)))
        self.buttons.append(Button(bx + half_w + 5, by, half_w, bh, "NEXT >", lambda: self.change_image(1)))
        by += bh + 20

        init_val = 60
        if hasattr(self, 'thresh_slider'): init_val = self.thresh_slider.val

        self.thresh_slider = Slider(bx, by, bw, 20, 0, 255, init_val, "Threshold")
        self.sliders.append(self.thresh_slider)
        by += 50

        def set_sort(idx):
            self.current_sort_idx = idx
            for b in self.buttons:
                if b.group_id == "sort": b.active = (b.text == self.sort_modes[idx])

        cv2.putText(self.img_curr, "Sort Mode", (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        for current_mode_index, mode_name in enumerate(self.sort_modes):
            btn = Button(bx, by, bw, bh, mode_name, lambda index=current_mode_index: set_sort(index), group_id="sort",
                         is_toggle=True)
            if current_mode_index == self.current_sort_idx:
                btn.active = True
            self.buttons.append(btn)
            by += bh + 5

        by += 10

        # --- MASK MODES ---
        def set_mask(mode):
            self.current_mask_mode = mode
            for b in self.buttons:
                if b.group_id == "mask": b.active = (b.text.lower() == mode)

        for mode in self.mask_modes:
            btn = Button(bx, by, bw, bh, mode.upper(), lambda m=mode: set_mask(m), group_id="mask", is_toggle=True)
            if mode == self.current_mask_mode: btn.active = True
            self.buttons.append(btn)
            by += bh + 5

        by += 20

        # --- SORT ACTIONS ---
        def sort_linear(direction, rev):
            self.save_state()
            self.img_curr = pixel_sort_fast(
                self.img_curr, self.current_mask_mode, self.current_sort_idx,
                direction, self.thresh_slider.val, 2, rev
            )

        self.buttons.append(Button(bx, by, bw, bh, "UP", lambda: sort_linear("vertical", False)))
        by += bh + 5
        self.buttons.append(Button(bx, by, half_w, bh, "LEFT", lambda: sort_linear("horizontal", True)))
        self.buttons.append(Button(bx + half_w + 5, by, half_w, bh, "RIGHT", lambda: sort_linear("horizontal", False)))
        by += bh + 5
        self.buttons.append(Button(bx, by, bw, bh, "DOWN", lambda: sort_linear("vertical", True)))
        by += bh + 20

        def sort_diag(angle):
            self.save_state()
            self.img_curr = pixel_sort_diagonal(
                self.img_curr, self.current_mask_mode, self.current_sort_idx,
                self.thresh_slider.val, angle
            )

        def sort_circle(mode):
            self.save_state()
            self.img_curr = pixel_sort_polar(
                self.img_curr, self.current_mask_mode, self.current_sort_idx,
                self.thresh_slider.val, mode
            )

        self.buttons.append(Button(bx, by, half_w, bh, "DIAG /", lambda: sort_diag(45)))
        self.buttons.append(Button(bx + half_w + 5, by, half_w, bh, "DIAG \\", lambda: sort_diag(-45)))
        by += bh + 5
        self.buttons.append(Button(bx, by, half_w, bh, "CIRCLE", lambda: sort_circle('circle')))
        self.buttons.append(Button(bx + half_w + 5, by, half_w, bh, "BURST", lambda: sort_circle('burst')))

        by += bh + 30

        self.buttons.append(Button(bx, by, bw, bh, "UNDO", self.undo))
        by += bh + 5

        def reset_action():
            self.save_state()
            self.img_curr = self.img_orig.copy()

        self.buttons.append(Button(bx, by, bw, bh, "RESET", reset_action))
        by += bh + 5

        def save_file():
            output_filename = f"sort_{datetime.now().strftime('%H%M%S')}.png"
            output_filepath = os.path.join(self.input_path, output_filename)
            Image.fromarray(self.img_curr).save(output_filepath)
            print(f"Saved {output_filepath}")

        self.buttons.append(Button(bx, by, bw, bh, "SAVE IMG", save_file))

    def mouse_callback(self, event, x, y, flags, param):
        self.mouse_x, self.mouse_y = x, y
        for s in self.sliders: s.handle_input(x, y, event)
        if event == cv2.EVENT_LBUTTONDOWN:
            for b in self.buttons:
                if b.is_inside(x, y): b.callback()
        for b in self.buttons: b.hover = b.is_inside(x, y)

    def run(self):
        cv2.namedWindow("Pixel Sort Studio")
        cv2.setMouseCallback("Pixel Sort Studio", self.mouse_callback)
        warmup_jit()

        while True:
            canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)

            h, w = self.img_curr.shape[:2]
            bgr_img = cv2.cvtColor(self.img_curr, cv2.COLOR_RGB2BGR)
            canvas[:h, :w] = bgr_img

            cv2.rectangle(canvas, (w, 0), (self.canvas_w, self.canvas_h), (30, 30, 30), -1)

            if self.image_files:
                fname = os.path.basename(self.image_files[self.current_index])
                idx_str = f"{self.current_index + 1} / {len(self.image_files)}"

                cv2.putText(canvas, fname, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(canvas, idx_str, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            status_text = f"Undo: {len(self.history)}"
            cv2.putText(canvas, status_text, (w + 10, self.canvas_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            for b in self.buttons: b.draw(canvas)
            for s in self.sliders: s.draw(canvas)

            cv2.imshow("Pixel Sort Studio", canvas)

            key = cv2.waitKey(1)
            if key == 27: break
            if key in [ord('z'), ord('Z')]: self.undo()

            if key == ord('n'): self.change_image(1)
            if key == ord('p'): self.change_image(-1)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    target_directory = get_target_directory()
    app = PixelSortApp(target_directory)
    app.run()