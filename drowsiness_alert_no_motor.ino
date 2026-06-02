// ============================================================
//   SMART DROWSINESS DETECTION SYSTEM
//   File    : drowsiness_alert_5sec.ino
//   Board   : Arduino Uno
//   Alerts  : Buzzer (Pin 8) + LED (Pin 13)
//             ON for 5 seconds when drowsiness is detected
// ============================================================
//
//   WIRING:
//   ┌──────────────────────────────────┐
//   │  BUZZER (Active 5V)              │
//   │    + pin  →  Arduino Pin 8       │
//   │    - pin  →  GND                 │
//   │                                  │
//   │  LED (optional)                  │
//   │    LED +  →  220Ω  →  Pin 13     │
//   │    LED -  →  GND                 │
//   └──────────────────────────────────┘
//
//   SERIAL PROTOCOL:
//   Python sends '1' → Buzzer ON + LED ON for 5 seconds
// ============================================================

#define BUZZER_PIN 8
#define LED_PIN    13

bool alertActive = false;

unsigned long alertStartTime = 0;
const unsigned long ALERT_DURATION = 5000; // 5 seconds

// ─────────────────────────────────────────
void setup() {
  Serial.begin(9600);

  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);

  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(LED_PIN, LOW);

  // 3 blinks on startup — confirms Arduino is ready
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(150);
    digitalWrite(LED_PIN, LOW);
    delay(150);
  }

  Serial.println("Arduino ready. Waiting for signal...");
}

// ─────────────────────────────────────────
void activateAlert() {
  alertActive = true;
  alertStartTime = millis();

  digitalWrite(BUZZER_PIN, HIGH);
  digitalWrite(LED_PIN, HIGH);

  Serial.println("ALERT ON — Drowsiness detected!");
}

// ─────────────────────────────────────────
void deactivateAlert() {
  alertActive = false;

  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(LED_PIN, LOW);

  Serial.println("Alert OFF");
}

// ─────────────────────────────────────────
void loop() {

  // Listen for Python command
  if (Serial.available() > 0) {
    char cmd = Serial.read();

    if (cmd == '1') {
      activateAlert();
    }
  }

  // Turn OFF buzzer and LED after 5 seconds
  if (alertActive && (millis() - alertStartTime >= ALERT_DURATION)) {
    deactivateAlert();
  }
}