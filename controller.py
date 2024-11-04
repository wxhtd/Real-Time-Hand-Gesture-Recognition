import serial
import time
import threading
import logging
import json
from collections import deque
from evaluate import evaluate_gesture

with open("config.json", "r") as config_file:
    config = json.load(config_file)

frame_number = int(config.get("prediction_frame_number", 10))
# Configure logging
logging.basicConfig(filename='output.log', level=logging.INFO, format='%(asctime)s - %(message)s')

ser = None
latest_data = None
reading = False
data_buffer = deque(maxlen=frame_number)  # Buffer to store the last `frame_number` frames

# Initialize the serial connection
def initialize_serial(port='COM3', baudrate=921600, retries=3, delay=2):
    global ser
    for attempt in range(retries):
        try:
            ser = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1
            )
            if ser.is_open:
                print(f"Connection established on {port}")
            return
        except serial.SerialException as e:
            print(f"Connection attempt {attempt + 1} failed: {e}")
            time.sleep(delay)
    raise serial.SerialException(f"Failed to open port {port} after {retries} attempts.")

# Function to send command to the board
def send_command(command):
    if ser and ser.is_open and command in ['0', '1', 'r', 's', 'c']:
        ser.write(command.encode())
        print(f"Sent command: {command}")
    else:
        print("Invalid command or serial not initialized.")

# Function to start reading data from the board
def start_reading():
    global reading
    reading = True
    threading.Thread(target=read_from_serial, daemon=True).start()

# Function to stop reading data
def stop_reading():
    global reading
    reading = False

# Function to handle serial data reading
def read_from_serial():
    global latest_data
    while reading and ser:
        if ser.in_waiting > 0:
            ln = ser.readline()
            try:
                data = ln.decode('utf-8').strip()
                # Store the latest data for retrieval by GUI
                logging.info(data)
                if "data:" in data:
                    latest_data = data.split("data:", 1)[1]
                    process_data(latest_data)  # Process and store data in buffer for gesture prediction
            except UnicodeDecodeError:
                # Fallback to binary handling if decoding fails
                latest_data = ' '.join(f'{byte:02x}' for byte in ln)
        time.sleep(0.1)  # Polling delay

# Function to retrieve the latest data (to be called by GUI)
def get_latest_data():
    return latest_data

# Function to handle keyboard input
def keyboard_input():
    while True:
        command = input("Enter command (r/s/c) or 'q' to quit: ").strip()
        if command == 'q':
            print("Exiting...")
            break
        send_command(command)

def close_serial():
    if ser and ser.is_open:
        ser.close()
        print("Serial connection closed.")

def send_updated_settings(zone_mode, ranging_rate):
    pass

# Function to process the data and store in buffer
def process_data(data):
    zones = [3000]*64
    for entry in data.split(";"):
        zone_data = entry.split(",")
        if len(zone_data) == 4:
            try:
                zone_id = int(zone_data[0])
                distance = int(zone_data[1]) if zone_data[1] != 'X' else 3000  # Set 'X' to 3000
                zones[zone_id] = distance
            except ValueError:
                # Skip this entry if conversion fails
                zones[zone_id] = 3000  # Default to 3000 for invalid entries
    if len(zones) == 64:
        data_buffer.append(zones)  # Store only complete frames

def get_gesture_prediction():
    # Return the buffer as a list of lists, ensuring it has exactly 10 frames
    prediction_matrix = list(data_buffer)
    # # If we have less than 10 frames, pad with 3000s for missing data
    # while len(prediction_matrix) < frame_number:
    #     prediction_matrix.insert(0, [3000] * 64)  # Pad with 3000 values
    if len(prediction_matrix) > 0:
        return evaluate_gesture(prediction_matrix)
    return '', []

if __name__ == '__main__':
    try:
        initialize_serial()
        reading = True
        # Start threads for reading serial and handling keyboard input
        serial_thread = threading.Thread(target=read_from_serial)
        input_thread = threading.Thread(target=keyboard_input)

        # Set threads as daemon so they close when the main program exits
        serial_thread.daemon = True
        input_thread.daemon = True

        # Start the threads
        serial_thread.start()
        input_thread.start()

        # Keep the main thread alive until the input thread completes
        input_thread.join()

    except KeyboardInterrupt:
        print("Closing connection.")
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
    finally:
        if ser.is_open:
            ser.close()