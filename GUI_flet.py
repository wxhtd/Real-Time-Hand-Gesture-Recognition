import flet as ft
import controller
import time
import json
from PIL import Image
import asyncio
import colorsys

# Load configuration
with open("config.json", "r") as config_file:
    config = json.load(config_file)

# Prediction delay and last prediction time
prediction_delay = float(config.get("prediction_delay", 2.0))
last_prediction_time = 0

# Initialize the serial connection
controller.initialize_serial()

# Main application
def main(page: ft.Page):
    global last_prediction_time
    page.title = "Hand Gesture Recognition"
    page.window_width = 800
    page.window_height = 600

    # Variables
    is_running = False  # Track whether updates should run

    # Zone Matrix Configuration
    zone_rows, zone_cols = 8, 8
    zone_size = 80  # Increased size for each zone
    
    # Zone Mode and Ranging Rate Controls
    zone_mode_dropdown = ft.Dropdown(
        options=[ft.dropdown.Option("4x4"), ft.dropdown.Option("8x8")],
        value="8x8",
        label="Zone Mode",
    )
    # Input Box for Ranging Rate
    ranging_rate_input = ft.TextField(
        label="Ranging Rate (Hz)",
        hint_text="Enter value (5-50)",
        value="15",
        # width=120,
        keyboard_type=ft.KeyboardType.NUMBER,
        on_change=lambda e: validate_ranging_rate(e.control),
    )

    def validate_ranging_rate(input_field):
        try:
            value = int(input_field.value)
            if value < 5:
                input_field.error_text = "Value must be at least 5"
            elif value > 50:
                input_field.error_text = "Value must not exceed 50"
            else:
                input_field.error_text = None
        except ValueError:
            input_field.error_text = "Enter a valid integer"
        input_field.update()

    change_setting_button = ft.ElevatedButton(
        text="Change Setting",
        on_click=lambda e: controller.send_updated_settings(
            zone_mode_dropdown.value, ranging_rate_input.value
        ),
    )

    # Start and Stop Buttons
    start_button = ft.ElevatedButton(
        text="Start",
        on_click=lambda e: [
            controller.send_command("0"),
            controller.start_reading(),
            start_button.disabled(True),
            stop_button.disabled(False),
            page.update(),
        ],
    )
    stop_button = ft.ElevatedButton(
        text="Stop",
        on_click=lambda e: [
            controller.send_command("1"),
            controller.stop_reading(),
            start_button.disabled(False),
            stop_button.disabled(True),
            page.update(),
        ],
        disabled=True,
    )

    # Hand Gesture Display
    gesture_image_size = 160  # Increased size for hand gesture display
    gesture_image = ft.Container(
        bgcolor="white",
        width=gesture_image_size,
        height=gesture_image_size,
        border_radius=4,
        border=ft.border.all(1, "black"),
    )
    gesture_label = ft.Text("Gesture: None", size=16)
    # gesture_image = ft.Image(src="./src/idle.png", width=80, height=80)
    
    # UI Components
    def create_zone_grid():
        """Create a grid of zones with text labels."""
        return ft.GridView(
            expand=True,
            runs_count=zone_cols,
            spacing=4,
            run_spacing=4,
            controls=[
                ft.Container(
                    content=ft.Text(f"R:0", size=14, text_align=ft.TextAlign.CENTER),
                    alignment=ft.alignment.center,
                    bgcolor="white",
                    width=zone_size,
                    height=zone_size,
                    border_radius=4,
                    border=ft.border.all(1, "black"),
                )
                for _ in range(zone_rows * zone_cols)
            ],
        )
    
        # Update zone display function
    async def update_zone_data():
        latest_data = controller.get_latest_data()
        if latest_data:
            try:
                zone_entries = latest_data.split(";")
                for entry in zone_entries:
                    zone_entry = entry.split(",")
                    if len(zone_entry) == 4:
                        zone_id = int(zone_entry[0])
                        distance = int(zone_entry[1]) if zone_entry[1] != "X" else -1
                        if distance == -1:
                            color = "#FFFFFF"  # White for missing data
                        else:
                            norm_distance = min(max(distance / 3000, 0), 1)  # Normalize distance to [0, 1]
                            hue = norm_distance * 240 / 360  # Map distance to hue range [0, 2/3]
                            # Adjust brightness (value): darker for mid-range
                            brightness = 0.7 if 0.3 <= norm_distance <= 0.7 else 1.0
                            rgb = colorsys.hsv_to_rgb(hue, 1, brightness)  # Full saturation and value
                            color = f"#{int(rgb[0] * 255):02X}{int(rgb[1] * 255):02X}{int(rgb[2] * 255):02X}"

                        # color = "#FFFFFF" if distance == -1 else f"#{int((1 - min(distance / 3000, 1)) * 255):02X}0000"
                        row, col = divmod(zone_id, 8)
                        zone_grid.controls[row*8+col].bgcolor = color
                        zone_grid.controls[row*8+col].content.value = f"R:{distance}"
                zone_grid.update()
            except Exception as e:
                print(f"Error updating zone data: {e}")

    # Update hand gesture display function
    async def update_gesture_display():
        gesture_name, probabilities = controller.get_gesture_prediction()
        if gesture_name:
            gesture_image.src = f"./src/{gesture_name}.png"
            gesture_label.value = f"Gesture: {gesture_name}"
        else:
            gesture_image.src = "./src/idle.png"
            gesture_label.value = "Gesture: None"
        page.update()

    # # Periodic update function zone_grid, gesture_image, gesture_label, page
    # async def periodic_update(zone_grid, gesture_image, gesture_label, page):
    #     global last_prediction_time
    #     while True:
    #         update_zone_data(zone_grid)
    #         current_time = time.time()
    #         if current_time - last_prediction_time >= prediction_delay:
    #             last_prediction_time = current_time
    #             update_gesture_display(page, gesture_image, gesture_label)
    #         page.update()
    #         await asyncio.sleep(0.2)  # 200ms delay

    async def run_periodic_updates():
        """Run periodic updates for zones and gestures."""
        while is_running:
            await asyncio.gather(update_zone_data(), update_gesture_display())
            await asyncio.sleep(0.2)  # Adjust the interval as needed

    def start_sampling(control):
        nonlocal is_running
        controller.send_command("0")  # Send the start command to the board
        controller.start_reading()    # Start reading data in background
        is_running = True
        # Schedule the periodic updates
        asyncio.run(run_periodic_updates())

        stop_button.disabled = False
        start_button.disabled = True
        change_setting_button.disabled = True
        ranging_rate_input.disabled = True
        zone_mode_dropdown.disabled = True
        page.update()

    def stop_sampling(control):
        """Handles the Stop button click event."""
        nonlocal is_running
        controller.send_command("1")  # Send stop command to the controller
        controller.stop_reading()     # Stop reading data
        is_running = False
        stop_button.disabled = True
        start_button.disabled = False
        change_setting_button.disabled = False
        ranging_rate_input.disabled = False
        zone_mode_dropdown.disabled = False
        page.update()

    # Function to send updated settings to the board
    def change_setting(control):
        controller.send_updated_settings(zone_mode_dropdown.value, ranging_rate_input.value)
        
    # Layout and Components
    zone_grid = create_zone_grid()    
    
    layout = ft.Row(
        controls=[
            # Left Side: Zone Grid
            ft.Container(
                content=zone_grid,
                padding=10,
                expand=True,
            ),
            # Right Side: Control Panel
            ft.Column(
                controls=[
                    ft.Dropdown(
                        label="Zone Mode",
                        value="8x8",
                        options=[
                            ft.dropdown.Option("4x4"),
                            ft.dropdown.Option("8x8"),
                        ],
                    ),
                    ranging_rate_input,
                    ft.ElevatedButton("Change Setting", on_click=change_setting),
                    ft.ElevatedButton("Start", on_click=start_sampling),
                    ft.ElevatedButton("Stop", on_click=stop_sampling),
                    ft.Divider(),
                    ft.Text("Hand Gesture:", size=16),
                    gesture_image,
                    gesture_label,
                ],
                expand=False,
                spacing=10,
            ),
        ],
        expand=True,
    )

    page.add(layout)

ft.app(target=main)