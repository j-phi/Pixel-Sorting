import sys
import os
from datetime import datetime
import numpy as np
import cv2
from PIL import Image
from numba import njit, prange
from pillow_heif import register_heif_opener
from typing import Optional
import time

register_heif_opener()

def get_target_directory() -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        print("\n[!] Error: tkinter is not installed. \nOn Linux, install it via: 'sudo apt-get install python3-tk'")
        sys.exit(1)

    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    selected_directory = filedialog.askdirectory(title="Select input/output folder")
    root.destroy()
    if not selected_directory:
        sys.exit()
    return selected_directory


def apply_mild_gaussian_blur(input_image_array: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(input_image_array, (3, 3), 0.5)

def apply_mild_noise_injection(input_image_array: np.ndarray) -> np.ndarray:
    noise_matrix = np.zeros(input_image_array.shape, dtype=np.int16)
    cv2.randn(noise_matrix, 0, 15)
    blended_image_matrix = cv2.addWeighted(input_image_array.astype(np.int16), 1.0, noise_matrix, 1.0, 0.0)
    return np.clip(blended_image_matrix, 0, 255).astype(np.uint8)

def apply_mild_chromatic_aberration(input_image_array: np.ndarray) -> np.ndarray:
    output_image_matrix = np.empty_like(input_image_array)
    output_image_matrix[:, :, 0] = np.roll(input_image_array[:, :, 0], -2, axis=1)
    output_image_matrix[:, :, 1] = input_image_array[:, :, 1]
    output_image_matrix[:, :, 2] = np.roll(input_image_array[:, :, 2], 2, axis=1)
    return output_image_matrix


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
    def __init__(self, x: int, y: int, w: int, h: int, text: str, callback: callable, group_id: str = None, is_toggle: bool = False) -> None:
        self.rect: tuple = (x, y, w, h)
        self.text: str = text
        self.callback: callable = callback
        self.group_id: str = group_id
        self.is_toggle: bool = is_toggle
        self.active: bool = False
        self.hover: bool = False
        self.color_idle: tuple = (50, 50, 50)

    def draw(self, canvas: np.ndarray) -> None:
        x, y, w, h = self.rect
        if self.active:
            color = (0, 180, 0)
        elif self.hover:
            color = (80, 80, 80)
        else:
            color = self.color_idle

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

        self.is_recording_sequence: bool = False
        self.recorded_sequence: list = []
        self.last_interaction_time: float = 0.0
        self.record_button: Optional[Button] = None

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

    def load_image_from_index(self, target_index: int) -> None:
        if not self.image_files or self.image_files[0] == "NO_IMG":
            self.img_orig = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
            self.img_curr = self.img_orig.copy()
            return

        self.current_index = target_index % len(self.image_files)
        target_filepath = self.image_files[self.current_index]
        print(f"Loading: {target_filepath}")

        try:
            loaded_pillow_image = Image.open(target_filepath).convert("RGB")
            self.img_orig = np.array(loaded_pillow_image)
            self.img_curr = self.img_orig.copy()
            self.history = []
        except Exception as loading_error:
            print(f"Error loading {target_filepath}: {loading_error}")
            self.img_orig = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
            self.img_curr = self.img_orig.copy()

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

    def setup_ui(self) -> None:
        self.buttons = []
        self.sliders = []

        base_x = 10
        base_y = 20
        button_width = self.ui_width - 20
        button_height = 30
        half_width = (button_width - 5) // 2

        def wrap_action(target_callable: callable) -> callable:
            def wrapped_execution() -> None:
                if self.is_recording_sequence:
                    self.recorded_sequence.append(target_callable)
                    self.last_interaction_time = time.time()
                target_callable()

            return wrapped_execution

        def toggle_record() -> None:
            if self.is_recording_sequence:
                self.is_recording_sequence = False
                self.record_button.text = "RECORD"
                self.record_button.color_idle = (50, 50, 50)
            else:
                self.is_recording_sequence = True
                self.recorded_sequence = []
                self.last_interaction_time = time.time()
                self.record_button.text = "STOP"
                self.record_button.color_idle = (0, 0, 200)

        def play_recorded_sequence() -> None:
            if not self.is_recording_sequence and self.recorded_sequence:
                for recorded_action in self.recorded_sequence:
                    recorded_action()

        self.record_button = Button(base_x, base_y, half_width, button_height, "RECORD", toggle_record)
        if self.is_recording_sequence:
            self.record_button.text = "STOP"
            self.record_button.color_idle = (0, 0, 200)

        self.buttons.append(self.record_button)
        self.buttons.append(
            Button(base_x + half_width + 5, base_y, half_width, button_height, "PLAY", play_recorded_sequence))
        base_y += button_height + 20

        self.buttons.append(
            Button(base_x, base_y, half_width, button_height, "< PREV", wrap_action(lambda: self.change_image(-1))))
        self.buttons.append(Button(base_x + half_width + 5, base_y, half_width, button_height, "NEXT >",
                                   wrap_action(lambda: self.change_image(1))))
        base_y += button_height + 20

        initial_threshold_value = self.thresh_slider.val if hasattr(self, 'thresh_slider') else 60
        self.thresh_slider = Slider(base_x, base_y, button_width, 20, 0, 255, initial_threshold_value, "Threshold")
        self.sliders.append(self.thresh_slider)
        base_y += 40

        def set_sort_mode(target_index: int) -> None:
            self.current_sort_idx = target_index
            for button_element in self.buttons:
                if button_element.group_id == "sort":
                    button_element.active = (button_element.text == self.sort_modes[target_index])

        for mode_index, mode_name in enumerate(self.sort_modes):
            button_instance = Button(base_x, base_y, button_width, button_height, mode_name,
                                     wrap_action(lambda target_index=mode_index: set_sort_mode(target_index)),
                                     group_id="sort", is_toggle=True)
            if mode_index == self.current_sort_idx:
                button_instance.active = True
            self.buttons.append(button_instance)
            base_y += button_height + 5

        base_y += 10

        def set_mask_mode(target_mode: str) -> None:
            self.current_mask_mode = target_mode
            for button_element in self.buttons:
                if button_element.group_id == "mask":
                    button_element.active = (button_element.text.lower() == target_mode)

        for mask_mode_name in self.mask_modes:
            button_instance = Button(base_x, base_y, button_width, button_height, mask_mode_name.upper(),
                                     wrap_action(lambda target_mode=mask_mode_name: set_mask_mode(target_mode)),
                                     group_id="mask", is_toggle=True)
            if mask_mode_name == self.current_mask_mode:
                button_instance.active = True
            self.buttons.append(button_instance)
            base_y += button_height + 5

        base_y += 20

        def sort_linear_action(direction_string: str, reverse_boolean: bool) -> None:
            self.save_state()
            self.img_curr = pixel_sort_fast(self.img_curr, self.current_mask_mode, self.current_sort_idx,
                                            direction_string, self.thresh_slider.val, 2, reverse_boolean)

        self.buttons.append(Button(base_x, base_y, button_width, button_height, "UP",
                                   wrap_action(lambda: sort_linear_action("vertical", False))))
        base_y += button_height + 5
        self.buttons.append(Button(base_x, base_y, half_width, button_height, "LEFT",
                                   wrap_action(lambda: sort_linear_action("horizontal", True))))
        self.buttons.append(Button(base_x + half_width + 5, base_y, half_width, button_height, "RIGHT",
                                   wrap_action(lambda: sort_linear_action("horizontal", False))))
        base_y += button_height + 5
        self.buttons.append(Button(base_x, base_y, button_width, button_height, "DOWN",
                                   wrap_action(lambda: sort_linear_action("vertical", True))))
        base_y += button_height + 20

        def sort_diagonal_action(angle_integer: int) -> None:
            self.save_state()
            self.img_curr = pixel_sort_diagonal(self.img_curr, self.current_mask_mode, self.current_sort_idx,
                                                self.thresh_slider.val, angle_integer)

        def sort_circular_action(mode_string: str) -> None:
            self.save_state()
            self.img_curr = pixel_sort_polar(self.img_curr, self.current_mask_mode, self.current_sort_idx,
                                             self.thresh_slider.val, mode_string)

        self.buttons.append(
            Button(base_x, base_y, half_width, button_height, "DIAG /", wrap_action(lambda: sort_diagonal_action(45))))
        self.buttons.append(Button(base_x + half_width + 5, base_y, half_width, button_height, "DIAG \\",
                                   wrap_action(lambda: sort_diagonal_action(-45))))
        base_y += button_height + 5
        self.buttons.append(Button(base_x, base_y, half_width, button_height, "CIRCLE",
                                   wrap_action(lambda: sort_circular_action('circle'))))
        self.buttons.append(Button(base_x + half_width + 5, base_y, half_width, button_height, "BURST",
                                   wrap_action(lambda: sort_circular_action('burst'))))

        base_y += button_height + 20

        def trigger_gaussian_blur() -> None:
            self.save_state()
            self.img_curr = apply_mild_gaussian_blur(self.img_curr)

        def trigger_noise_injection() -> None:
            self.save_state()
            self.img_curr = apply_mild_noise_injection(self.img_curr)

        def trigger_chromatic_aberration() -> None:
            self.save_state()
            self.img_curr = apply_mild_chromatic_aberration(self.img_curr)

        self.buttons.append(
            Button(base_x, base_y, button_width, button_height, "BLUR (MILD)", wrap_action(trigger_gaussian_blur)))
        base_y += button_height + 5
        self.buttons.append(
            Button(base_x, base_y, button_width, button_height, "NOISE (MILD)", wrap_action(trigger_noise_injection)))
        base_y += button_height + 5
        self.buttons.append(
            Button(base_x, base_y, button_width, button_height, "RGB SHIFT", wrap_action(trigger_chromatic_aberration)))

        base_y += button_height + 30

        self.buttons.append(Button(base_x, base_y, button_width, button_height, "UNDO", wrap_action(self.undo)))
        base_y += button_height + 5

        def reset_image_action() -> None:
            self.save_state()
            self.img_curr = self.img_orig.copy()

        self.buttons.append(
            Button(base_x, base_y, button_width, button_height, "RESET", wrap_action(reset_image_action)))
        base_y += button_height + 5

        def save_processed_image() -> None:
            output_filename = f"sort_{datetime.now().strftime('%H%M%S')}.png"
            output_filepath = os.path.join(self.input_path, output_filename)
            Image.fromarray(self.img_curr).save(output_filepath)
            print(f"Saved {output_filepath}")

        self.buttons.append(
            Button(base_x, base_y, button_width, button_height, "SAVE IMG", wrap_action(save_processed_image)))

    def mouse_callback(self, event: int, x: int, y: int, flags: int, param: any) -> None:
        self.mouse_x, self.mouse_y = x, y
        for slider_instance in self.sliders:
            was_dragging = slider_instance.dragging
            slider_instance.handle_input(x, y, event)
            if was_dragging and not slider_instance.dragging:
                if self.is_recording_sequence:
                    captured_value = slider_instance.val
                    self.recorded_sequence.append(
                        lambda current_value=captured_value: setattr(slider_instance, 'val', current_value))
                    self.last_interaction_time = time.time()

        if event == cv2.EVENT_LBUTTONDOWN:
            for button_instance in self.buttons:
                if button_instance.is_inside(x, y):
                    button_instance.callback()

        for button_instance in self.buttons:
            button_instance.hover = button_instance.is_inside(x, y)

    def run(self) -> None:
        window_name: str = "Pixel Sort Studio"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1440, 960)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        warmup_jit()

        internal_width: int = 1440
        internal_height: int = 960
        display_width: int = internal_width - self.ui_width

        while True:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            canvas = np.zeros((internal_height, internal_width, 3), dtype=np.uint8)
            cv2.rectangle(canvas, (0, 0), (self.ui_width, internal_height), (35, 35, 35), -1)

            image_height, image_width = self.img_curr.shape[:2]
            scale_ratio = min(display_width / image_width, internal_height / image_height)
            new_width = int(image_width * scale_ratio)
            new_height = int(image_height * scale_ratio)

            bgr_image = cv2.cvtColor(self.img_curr, cv2.COLOR_RGB2BGR)
            resized_image = cv2.resize(bgr_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

            x_offset = self.ui_width + (display_width - new_width) // 2
            y_offset = (internal_height - new_height) // 2

            canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image

            if self.image_files:
                file_name = os.path.basename(self.image_files[self.current_index])
                index_string = f"{self.current_index + 1} / {len(self.image_files)}"
                cv2.putText(canvas, file_name, (self.ui_width + 10, internal_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(canvas, index_string, (self.ui_width + 10, internal_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            status_text = f"Undo: {len(self.history)}"
            cv2.putText(canvas, status_text, (10, internal_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            for button_instance in self.buttons:
                button_instance.draw(canvas)
            for slider_instance in self.sliders:
                slider_instance.draw(canvas)

            cv2.imshow(window_name, canvas)

            keyboard_input = cv2.waitKey(1)
            if keyboard_input == 27:
                break
            if keyboard_input in [ord('z'), ord('Z')]:
                self.undo()
            if keyboard_input == ord('n'):
                self.change_image(1)
            if keyboard_input == ord('p'):
                self.change_image(-1)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    target_directory = get_target_directory()
    app = PixelSortApp(target_directory)
    app.run()