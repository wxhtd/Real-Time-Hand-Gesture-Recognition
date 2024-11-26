import serial
import time
import threading
import logging
import json
from collections import deque
from queue import PriorityQueue
from evaluate import evaluate_gesture

with open("config.json", "r") as config_file:
    config = json.load(config_file)

frame_number = int(config.get("prediction_frame_number", 10))
background_distance_threshold = float(config.get("background_distance_threshold", 600))
relative_background_distance  = float(config.get("relative_background_distance", 200))
port = config["port"]

# Configure logging
log_level = logging.getLevelName(config.get("log_level", "INFO"))
# Configure logging
logging.basicConfig(filename='output2.log', level=log_level, format='%(asctime)s - %(message)s')

ser = None
latest_data = None
reading = False
data_distance_buffer = deque(maxlen=frame_number)  # Buffer to store the distance data of last frame
data_signal_buffer = deque(maxlen=frame_number)  # Buffer to store the signal data of last frame

# Initialize the serial connection
def initialize_serial(port=port, baudrate=921600, retries=3, delay=2):
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
    #raise serial.SerialException(f"Failed to open port {port} after {retries} attempts.")

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
                logging.warning(data)
                if "data:" in data:
                    process_data(data.split("data:", 1)[1])  # Process and store data in buffer for gesture prediction
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

def clear_buffer():
    data_distance_buffer.clear()
    data_signal_buffer.clear()

# Function to process the data and store in buffer
def process_data(data):
    global latest_data
    zones = [3000]*64
    signals = [0]*64
    cells_to_process = []
    pqueue = PriorityQueue()
    for entry in data.split(";"):
        zone_data = entry.split(",")
        if len(zone_data) == 4:
            try:
                zone_id = int(zone_data[0])
                if zone_data[1] == 'X':
                    cells_to_process.append(zone_id)
                else:
                    if int(zone_data[1]) > background_distance_threshold:
                        distance = 3000
                        signal = 0
                    else:
                        distance = int(zone_data[1])
                        signal = int(zone_data[3])
                    pqueue.put(distance)
                    zones[zone_id] = distance
                    signals[zone_id] = signal
            except ValueError:
                # Skip this entry if conversion fails
                zones[zone_id] = 3000  # Default to 3000 for invalid entries
                signals[zone_id] = 0  # Default to 0 for invalid entries
    # for cell in cells_to_process:
    #     neighbors = {cell-9,cell-8,cell-7,cell-1,cell+1,cell+7,cell+8,cell+9}
    #     if cell %8 == 0:
    #         neighbors -= {cell-1,cell-9,cell+7}
    #     elif (cell - 7) %8 == 0:
    #         neighbors -= {cell+1,cell-7,cell+9}
    #     if cell/8 < 1:
    #         neighbors -= {cell-7,cell-8,cell-9}
    #     elif cell/8 >= 7:
    #         neighbors -= {cell+7,cell+8,cell+9}
    #     divider, distance, signal = 0,0,0
    #     for n in neighbors:
    #         if zones[n] != 3000:
    #             divider += 1
    #             distance += zones[n]
    #             signal += signals[n]
    #     if divider != 0:
    #         distance /= divider
    #         pqueue.put(distance)
    #         signal /= divider
    #         if round(distance) < background_distance_threshold:                
    #             zones[cell] = round(distance)
    #             signals[cell] = round(signal)
    
    closest_distance, closest_distance_count = 0, 0
    # Remove elements based on priority
    while not pqueue.empty():
        dist = pqueue.get()
        # logging.info(f"Get {dist} from priority queue")
        closest_distance += dist
        closest_distance_count += 1
        if closest_distance_count == 5:
            break
    if closest_distance_count == 0:
        distance_threshold = closest_distance
    else:
        distance_threshold = closest_distance / closest_distance_count + relative_background_distance
    logging.info(f'Distance threshold = {distance_threshold}')
    for i in range(64):
        if zones[i] > distance_threshold:
            zones[i] = 3000
            signals[i] = 0
    latest_data = zones, signals
    logging.warning(zones+signals)
    if len(zones) == 64:
        data_distance_buffer.append(zones)  # Store only complete frames
        data_signal_buffer.append(signals)

def get_gesture_prediction():
    # Return the buffer as a list of lists, ensuring it has exactly 10 frames
    prediction_matrix_distance = list(data_distance_buffer)
    prediction_matrix_signal = list(data_signal_buffer)
    # # If we have less than 10 frames, pad with 3000s for missing data
    # while len(prediction_matrix) < frame_number:
    #     prediction_matrix.insert(0, [3000] * 64)  # Pad with 3000 values
    if len(prediction_matrix_distance) > 0:
        return evaluate_gesture(prediction_matrix_distance, prediction_matrix_signal)
        # return evaluate_gesture_both(prediction_matrix, prediction_matrix_signal)
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