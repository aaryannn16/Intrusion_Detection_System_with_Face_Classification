#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include "time.h"
#include "mbedtls/base64.h" // Include Base64 encoding library

// Camera configuration
#define CAMERA_MODEL_AI_THINKER // Use AI_THINKER for ESP32-CAM

// PIR sensor pin
const int PIR_PIN = 13;

// WiFi credentials
const char* ssid = "Nothing phone (1)";
const char* password = "9876543210";

// Flask server URL
const char* serverURL = "http://192.168.232.1:5000/upload"; // Update with your Flask server IP

// NTP settings
const char* ntpServer = "pool.ntp.org";
const long gmtOffset_sec = 19800; // Offset for IST (UTC+5:30)
const int daylightOffset_sec = 0; // No daylight saving in IST

void setup() {
    Serial.begin(115200);
    pinMode(PIR_PIN, INPUT);

    // Connect to WiFi
    connectToWiFi();

    // Initialize NTP
    configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);
    Serial.println("NTP initialized. Waiting for time sync...");

    // Wait for time to be set
    struct tm timeInfo;
    while (!getLocalTime(&timeInfo)) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nTime synchronized!");

    // Camera configuration
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = 5;   // D0
    config.pin_d1 = 18;  // D1
    config.pin_d2 = 19;  // D2
    config.pin_d3 = 21;  // D3
    config.pin_d4 = 36;  // D4
    config.pin_d5 = 39;  // D5
    config.pin_d6 = 34;  // D6
    config.pin_d7 = 35;  // D7
    config.pin_xclk = 0; // XCLK
    config.pin_pclk = 22; // PCLK
    config.pin_vsync = 25; // VSYNC
    config.pin_href = 23; // HREF
    config.pin_sccb_sda = 26; // SDA
    config.pin_sccb_scl = 27; // SCL
    config.pin_pwdn = 32; // PWDN
    config.pin_reset = -1; // RESET

    // Set the clock signal to 20 MHz
    config.xclk_freq_hz = 20000000; // 20 MHz

    // Frame size and image quality
    config.frame_size = FRAMESIZE_VGA;  // Set frame size
    config.pixel_format = PIXFORMAT_RGB565;  // Capture in RGB565
    config.jpeg_quality = 12;  // JPEG quality (lower is better quality, larger size)
    config.fb_count = 1;  // Use only one frame buffer

    // Initialize camera
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed with error 0x%x", err);
        return;
    }
}

void loop() {
    if (digitalRead(PIR_PIN) == HIGH) {
        Serial.println("Motion detected!");
        captureAndSendImage();
        delay(10000); // Delay to avoid multiple captures on a single motion
    }
}

void captureAndSendImage() {
    // Capture image in RGB565 format
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Camera capture failed");
        return;
    }

    // Convert the RGB565 image to JPEG format
    uint8_t* jpg_buf = NULL;
    size_t jpg_len = 0;

    bool converted = frame2jpg(fb, 80, &jpg_buf, &jpg_len); // Convert RGB565 to JPEG with quality 80
    if (!converted) {
        Serial.println("JPEG conversion failed");
        esp_camera_fb_return(fb);
        return;
    }

    // Get the current timestamp
    struct tm timeInfo;
    if (getLocalTime(&timeInfo)) {
        char timeStamp[30];
        strftime(timeStamp, sizeof(timeStamp), "%Y-%m-%d %H:%M:%S", &timeInfo);
        Serial.printf("Captured at: %s\n", timeStamp);

        // Encode the image to Base64
        size_t outputLen = 4 * ((jpg_len + 2) / 3) + 1; // Base64 size calculation
        char* base64Image = (char*)malloc(outputLen);
        if (base64Image) {
            size_t encodedLen = 0;
            mbedtls_base64_encode((unsigned char*)base64Image, outputLen, &encodedLen, jpg_buf, jpg_len);

            // Send the image and timestamp to the server
            if (sendImage(base64Image, timeStamp)) {
                Serial.println("Image sent successfully with timestamp");
            } else {
                Serial.println("Failed to send image");
            }

            free(base64Image); // Free the allocated memory
        } else {
            Serial.println("Failed to allocate memory for Base64 encoding");
        }
    } else {
        Serial.println("Failed to get current time.");
    }

    // Free the memory used for JPEG
    free(jpg_buf);

    // Return the frame buffer back to be reused
    esp_camera_fb_return(fb);
}

bool sendImage(const char* base64Image, const char* timestamp) {
    if (WiFi.status() == WL_CONNECTED) {
        HTTPClient http;
        http.begin(serverURL);
        http.addHeader("Content-Type", "application/json");

        // Prepare JSON payload
        String payload = "{\"timestamp\":\"" + String(timestamp) + "\",\"image\":\"" + String(base64Image) + "\"}";

        // POST the JSON payload
        int httpResponseCode = http.POST(payload);
        http.end();

        return (httpResponseCode > 0);
    }
    return false;
}

void connectToWiFi() {
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(1000);
        Serial.println("Connecting to WiFi...");
    }
    Serial.println("Connected to WiFi!");
}
