import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import controller
import time
import json


with open("config.json", "r") as config_file:
    config = json.load(config_file)

# how long the prediction result will remain on the UI
prediction_delay = float(config.get("prediction_delay", 2.0))
# Initialize the last prediction time to control prediction intervals
last_prediction_time = 0
# Initialize the serial connection in controller
controller.initialize_serial()

# Function to send start command and start data reading
def start_sampling():
    controller.send_command("0")  # Send the start command to the board
    controller.start_reading()    # Start reading data in background
    start_button.config(state="disabled")
    stop_button.config(state="normal")
    zone_mode_option.config(state="disabled")
    ranging_rate_entry.config(state="disabled")
    change_setting_button.config(state="disabled")

# Function to send stop command and stop data reading
def stop_sampling():
    controller.send_command("1")  # Send the stop command to the board
    controller.stop_reading()     # Stop reading data
    start_button.config(state="normal")
    stop_button.config(state="disabled")
    zone_mode_option.config(state="normal")
    ranging_rate_entry.config(state="normal")
    change_setting_button.config(state="normal")

# Function to send updated settings to the board
def change_setting():
    zone_mode = zone_mode_var.get()
    try:
        ranging_rate = min(50, int(ranging_rate_var.get()))  # Limit ranging rate to 50
        ranging_rate_var.set(str(ranging_rate))  # Update entry to reflect limited value
    except ValueError:
        ranging_rate = 50
        ranging_rate_var.set("50")  # Default to 50 if input is invalid

    controller.send_updated_settings(zone_mode, ranging_rate)  # Send settings to board

# Function to update the zone data
def update_zone_data():
    latest_data = controller.get_latest_data()
    if latest_data:
        try:
            # Split data by semicolon to get each zone's data
            zone_entries = latest_data.split(';')
            colormap = plt.cm.plasma  # Use 'plasma' colormap for rainbow effect
            if len(zone_entries) < 64:
                return
            for entry in zone_entries:
                zone_entry = entry.split(',')
                if len(zone_entry) == 4:
                    zone_id = (int)(zone_entry[0])
                    # zone_entry = [0,1680,0,0]
                    # The distance value is the 4th entry in each zone data (based on your CSV format)
                    if zone_entry[1] == 'X':
                        distance = -1  # Indicate missing data
                        color = (1.0, 1.0, 1.0)  # White color for missing data
                    else:
                        distance = int(zone_entry[1])
                        # Normalize distance to the range [0, 1] and get color from colormap
                        norm_distance = 1 - min(max(distance / 3000, 0), 1)  # Clamp between 0 and 1
                        color = colormap(norm_distance)  # Get the color from colormap
                    row = zone_id // 8
                    col = zone_id % 8
                    # Set color and text for each rectangle in the grid
                    rects[row][col].set_facecolor(color)
                    texts[row][col].set_text(f"R:{distance}")
            
            # Redraw the canvas after updating all zones
            canvas.draw()
        
        except ValueError as ve:
            print("ValueError encountered in update_zone_data:")
            print(f"Data: {zone_entries}")
            print(f"Error details: {ve}")
        
        except IndexError as ie:
            print("IndexError encountered in update_zone_data:")
            print(f"Data: {zone_entries}")
            print(f"Error details: {ie}")

        except Exception as e:
            print("Unexpected error encountered in update_zone_data:")
            print(f"Data: {zone_entries}")
            print(f"Error details: {e}")

def update_zone_data_builtin():
    global last_prediction_time

    latest_data = controller.get_latest_data()
    if latest_data:
        result = [a.strip() for a in latest_data.split(' ')]
        if len(result) > 1:
            gesture_name = "idle"
            if result[0] == "STATIC":
                if result[1] == '0':
                    gesture_name = "idle"
                elif result[1] == '21':
                    gesture_name = "thumbs_up"
                elif result[1] == '25':
                    gesture_name = "thumbs_down"
                elif result[1] == '3':
                    gesture_name = "palm_left"
                elif result[1] == '4':
                    gesture_name = "palm_right"
                elif result[1] == '5':
                    gesture_name = "palm_down"
            elif result[0] == "DYNAMIC":
                if result[1] == '0':
                    gesture_name = "idle"
                elif result[1] == '1':
                    gesture_name = "palm_left"
                elif result[1] == '2':
                    gesture_name = "palm_right"
                elif result[1] == '3':
                    gesture_name = "palm_down"
                elif result[1] == '4':
                    gesture_name = "palm_up"
                elif result[1] == '5':
                    gesture_name = "palm_forward"
                elif result[1] == '6':
                    gesture_name = "palm_backward"
                elif result[1] == '7':
                    gesture_name = "palm_double_tap"
            elif result[0] == "Gesture":
                if result[1] == '':
                    gesture_name = "idle"
                elif result[1] == 'LEFT':
                    gesture_name = "palm_left"
                elif result[1] == 'RIGHT':
                    gesture_name = "palm_right"
                elif result[1] == 'DOWN':
                    gesture_name = "palm_down"
                elif result[1] == 'UP':
                    gesture_name = "palm_up"
                elif result[1] == 'FORWARD':
                    gesture_name = "palm_forward"
                elif result[1] == 'BACKWARD':
                    gesture_name = "palm_backward"
                elif result[1] == 'DOUBLE':
                    gesture_name = "palm_double_tap"
            if gesture_name != 'idle':
                last_prediction_time = time.time()
                print(f'result:{result} | pic:{gesture_name}')
                # Display the predicted gesture image
                image_path = f".\\src\\{gesture_name}.png"  # Assuming images are named as 'gesture_name.png'
                gesture_image = Image.open(image_path).resize((gesture_display_width, gesture_display_height))
                gesture_photo = ImageTk.PhotoImage(gesture_image)
                gesture_display.config(image=gesture_photo)
                gesture_display.image = gesture_photo  # Keep reference to avoid garbage collection
                return
            # # Display the probability for each gesture
            # probability_text = "\n".join([f"{gesture}: {prob:.2%}" for gesture, prob in probabilities])
            # probability_display.config(state="normal")
            # probability_display.delete(1.0, tk.END)
            # probability_display.insert(tk.END, probability_text)
            # probability_display.config(state="disabled")
    
    # Use the placeholder image if no gesture is detected
    gesture_display.config(image=placeholder_photo)
    gesture_display.image = placeholder_photo  # Keep reference to avoid garbage collection
    probability_display.config(state="normal")
    probability_display.delete(1.0, tk.END)
    probability_display.config(state="disabled")

# Function to update hand gesture image and probability display
def update_gesture_display(gesture_name, probabilities):
    if gesture_name:
        # Display the predicted gesture image
        image_path = f".\\src\\{gesture_name}.png"  # Assuming images are named as 'gesture_name.png'
        gesture_image = Image.open(image_path).resize((gesture_display_width, gesture_display_height))
        gesture_photo = ImageTk.PhotoImage(gesture_image)
        gesture_display.config(image=gesture_photo)
        gesture_display.image = gesture_photo  # Keep reference to avoid garbage collection

        # Display the probability for each gesture
        probability_text = "\n".join([f"{gesture}: {prob:.2%}" for gesture, prob in probabilities])
        probability_display.config(state="normal")
        probability_display.delete(1.0, tk.END)
        probability_display.insert(tk.END, probability_text)
        probability_display.config(state="disabled")
    else:
        # Use the placeholder image if no gesture is detected
        gesture_display.config(image=placeholder_photo)
        gesture_display.image = placeholder_photo  # Keep reference to avoid garbage collection

        probability_display.config(state="normal")
        probability_display.delete(1.0, tk.END)
        probability_display.config(state="disabled")

    
# Set up the main GUI window
root = tk.Tk()
root.title("Hand Gesture Recognition")

# Zone Display (using Matplotlib)
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xticks([])
ax.set_yticks([])
rects = []
texts = []

# Create 8x8 grid of rectangles with distance labels
for i in range(8):
    row_rects = []
    row_texts = []
    for j in range(8):
        rect = plt.Rectangle((j, 7 - i), 1, 1, edgecolor='black', facecolor='white')
        ax.add_patch(rect)
        text = ax.text(j + 0.5, 7 - i + 0.5, f"R:0", ha="center", va="center", color="black")
        row_rects.append(rect)
        row_texts.append(text)
    rects.append(row_rects)
    texts.append(row_texts)

ax.set_xlim(0, 8)
ax.set_ylim(0, 8)

# Embed Matplotlib figure in Tkinter
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(row=0, column=0, padx=10, pady=10, sticky="nw")  # Move to top-left

# Control Panel on the right
control_panel = tk.Frame(root)
control_panel.grid(row=0, column=1, padx=10, pady=10, sticky="n")

# Zone Mode dropdown
tk.Label(control_panel, text="Zone Mode:").pack(anchor="w")
zone_mode_var = tk.StringVar(value="8x8")
zone_mode_option = ttk.Combobox(control_panel, textvariable=zone_mode_var, values=["4x4", "8x8"], state="readonly")
zone_mode_option.pack(anchor="w", fill="x")

# Ranging Rate entry
tk.Label(control_panel, text="Ranging Rate (Hz):").pack(anchor="w")
ranging_rate_var = tk.StringVar(value="15")
ranging_rate_entry = ttk.Entry(control_panel, textvariable=ranging_rate_var)
ranging_rate_entry.pack(anchor="w", fill="x")

# Change Setting button
change_setting_button = tk.Button(control_panel, text="Change Setting", command=change_setting, state="normal")
change_setting_button.pack(anchor="w", fill="x", pady=(5, 5))

# Start and Stop buttons
start_button = tk.Button(control_panel, text="Start", command=start_sampling)
start_button.pack(anchor="w", fill="x", pady=(10, 0))
stop_button = tk.Button(control_panel, text="Stop", command=stop_sampling, state="disabled")
stop_button.pack(anchor="w", fill="x")


# Define fixed width and height for the gesture display label
gesture_display_width = 80
gesture_display_height = 80

# Load the placeholder image initially
placeholder_image_path = ".\\src\\idle.png"
placeholder_image = Image.open(placeholder_image_path).resize((gesture_display_width, gesture_display_height))
placeholder_photo = ImageTk.PhotoImage(placeholder_image)

# Hand Gesture display
gesture_label = tk.Label(control_panel, text="Hand Gesture:")
gesture_label.pack(anchor="w")

# Gesture display label with fixed size
gesture_display = tk.Label(control_panel, image=placeholder_photo, bg="lightgray", relief="solid")
gesture_display.image = placeholder_photo  # Keep reference to avoid garbage collection
gesture_display.pack(anchor="w", padx=5, pady=5)

# # Create a frame with fixed size to contain the gesture display label
# gesture_frame = tk.Frame(control_panel, width=gesture_display_width, height=gesture_display_height, bg="white")
# gesture_frame.pack(anchor="w", padx=5, pady=5)  # Pack the frame normally

# # Hand Gesture display
# gesture_label = tk.Label(control_panel, text="Hand Gesture:")
# gesture_label.pack(anchor="w")
# # Place the gesture display label inside this fixed-size frame
# gesture_display = tk.Label(gesture_frame, bg="white", relief="solid")
# gesture_display.pack(fill="both", expand=True)

# Probability display
probability_label = tk.Label(control_panel, text="Probability:")
probability_label.pack(anchor="w")
# probability_display = tk.Label(control_panel, width=10, height=10, wrap="word", state="disabled", anchor="w", relief="solid")
probability_display = tk.Text(control_panel, height=10, width=10, wrap="word", state="disabled")
probability_display.pack(anchor="w",padx=5,pady=5, fill="x")

# Update zone data periodically
def periodic_update():
    global last_prediction_time
    # Get the current time
    current_time = time.time()
    # Check if 1 second has passed since the last prediction
    if current_time - last_prediction_time >= prediction_delay:
    # if controller.reading:
    #update_zone_data() # Update zone data before scheduling the next update
        update_zone_data_builtin()

    # # Check if 1 second has passed since the last prediction
    # if current_time - last_prediction_time >= prediction_delay:
    #     # Update the last prediction time
    #     last_prediction_time = current_time
        
    #     # Retrieve the latest gesture prediction from the controller
    #     gesture_name, probability_array = controller.get_gesture_prediction()
        
    #     if gesture_name:
    #         # Display new gesture prediction
    #         update_gesture_display(gesture_name, probability_array)
    #     else:
    #         # Clear display if no gesture is detected
    #         update_gesture_display(None, [])
        
    global update_task_id
    update_task_id = root.after(200, periodic_update)

def on_close():
    """Cleanup tasks before closing the GUI."""
    # Stop the periodic update
    root.after_cancel(update_task_id)    
    # Stop reading data from the controller
    controller.stop_reading()    
    # Close the serial connection
    controller.close_serial()    
    # Close the GUI
    root.destroy()

# Start the periodic update
update_task_id = root.after(200, periodic_update)
# Bind the close event to the on_close function
root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
