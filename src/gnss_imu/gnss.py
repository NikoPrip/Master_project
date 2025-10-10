import paho.mqtt.client as mqtt
import json
import math
from pyproj import Transformer
import sys

# MQTT broker configuration
BROKER_HOST = "192.168.40.2"
BROKER_PORT = 1883
TOPIC = "GNSSInterface/Data"

def transform_to_utm(lat, lon):
    """Transform WGS84 coordinates to UTM."""
    # Convert radians to degrees if needed
    if abs(lat) <= math.pi and abs(lon) <= 2 * math.pi:
        lat, lon = math.degrees(lat), math.degrees(lon)
    
    # Calculate UTM zone and hemisphere
    zone = int((lon + 180) / 6) + 1
    hemisphere = 'N' if lat >= 0 else 'S'
    
    # Create transformer from WGS84 to UTM
    utm_crs = f"EPSG:{32600 + zone if hemisphere == 'N' else 32700 + zone}"
    transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    
    return f"{zone}{hemisphere}", x, y

def on_connect(client, userdata, flags, rc):
    """Callback for when the client receives a CONNACK response from the server."""
    if rc == 0:
        print("Connected. Starting GNSS data stream...")
        client.subscribe(TOPIC)
    else:
        print(f"Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    """Callback for when a PUBLISH message is received from the server."""
    try:
        data = json.loads(msg.payload.decode('utf-8'))
        
        # Extract required fields
        date = data.get('DATE', 'N/A')
        timestamp = data.get('timestamp', 'N/A')
        lat = float(data.get('LAT', 0))
        lon = float(data.get('LONG', 0))
        
        # Transform and print
        zone, x, y = transform_to_utm(lat, lon)
        print(f"Date: {date} | Timestamp: {timestamp}")
        print(f"UTM Zone: {zone} | Easting: {x:.2f} m | Northing: {y:.2f} m")
        print("-" * 50)
        
    except Exception as e:
        print(f"Error: {e}")

def on_disconnect(client, userdata, rc):
    """Callback for when the client disconnects from the broker."""
    print("Disconnected from MQTT broker")

def main():
    """Main function to set up MQTT client and start listening."""
    # Create MQTT client
    client = mqtt.Client()
    
    # Set callbacks
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect
    
    try:
        # Connect to broker
        print(f"Connecting to MQTT broker at {BROKER_HOST}:{BROKER_PORT}...")
        client.connect(BROKER_HOST, BROKER_PORT, 60)
        
        # Start the loop to process network traffic and dispatch callbacks
        print("Starting MQTT client loop. Press Ctrl+C to exit.")
        client.loop_forever()
        
    except KeyboardInterrupt:
        print("\nReceived interrupt signal. Disconnecting...")
        client.disconnect()
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()