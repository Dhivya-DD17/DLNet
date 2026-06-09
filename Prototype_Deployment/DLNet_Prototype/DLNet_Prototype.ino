// #include <Arduino.h>
// #include <Wire.h>

// // ===== Sensors =====
// #include <DHT.h>
// #include <INA219_WE.h>

// // ===== LCD =====
// #include <DFRobot_RGBLCD1602.h>

// // ===== TFLite =====
// #include <Chirale_TensorFlowLite.h>
// #include "model.h"
// #include "tensorflow/lite/micro/all_ops_resolver.h"
// #include "tensorflow/lite/micro/micro_interpreter.h"
// #include "tensorflow/lite/schema/schema_generated.h"

// // ================= CONFIG =================
// #define DHTPIN 3
// #define DHTTYPE DHT11

// const float SCALE_MIN = 0.8178250862438334f;
// const float SCALE_MAX = 1.006409511628937f;

// constexpr int INPUT_LEN = 100;
// constexpr int OUTPUT_LEN = 100;
// constexpr int kTensorArenaSize = 25000;
// alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// // ================= FIXED SOH INPUT =================
// const float inputValues[INPUT_LEN] = {
//   0.9608134745,0.9606085219,0.9604032350,0.9601976085,0.9599916364,
//   0.9597853129,0.9595786321,0.9593715875,0.9591641728,0.9589563815,
//   0.9587482068,0.9585396417,0.9583306793,0.9581213122,0.9579115330,
//   0.9577013341,0.9574907079,0.9572796462,0.9570681410,0.9568561841,
//   0.9566437669,0.9564308808,0.9562175171,0.9560036667,0.9557893206,
//   0.9555744693,0.9553591034,0.9551432132,0.9549267888,0.9547098203,
//   0.9544922973,0.9542742097,0.9540555467,0.9538362978,0.9536164519,
//   0.9533959980,0.9531749248,0.9529532209,0.9527308747,0.9525078744,
//   0.9522842081,0.9520598635,0.9518348284,0.9516090902,0.9513826363,
//   0.9511554537,0.9509275295,0.9506988504,0.9504694030,0.9502391736,
//   0.9500081485,0.9497763138,0.9495436552,0.9493101584,0.9490758089,
//   0.9488405919,0.9486044925,0.9483674957,0.9481295860,0.9478907481,
//   0.9476509662,0.9474102244,0.9471685066,0.9469257965,0.9466820776,
//   0.9464373332,0.9461915465,0.9459447002,0.9456967770,0.9454477594,
//   0.9451976296,0.9449463697,0.9446939613,0.9444403862,0.9441856255,
//   0.9439296605,0.9436724719,0.9434140405,0.9431543466,0.9428933704,
//   0.9426310917,0.9423674902,0.9421025454,0.9418362363,0.9415685417,
//   0.9412994404,0.9410289106,0.9407569303,0.9404834774,0.9402085292,
//   0.9399320630,0.9396540556,0.9393744837,0.9390933233,0.9388105506,
//   0.9385261411,0.9382400700,0.9379523124,0.9376628428,0.9373716354
// };

// // ================= GLOBALS =================
// DHT dht(DHTPIN, DHTTYPE);
// INA219_WE ina219;

// #define LCD_ADDR 0x3E
// #define RGB_ADDR 0x2D
// DFRobot_RGBLCD1602 lcd(RGB_ADDR, 16, 2, &Wire, LCD_ADDR);

// const tflite::Model* model = nullptr;
// tflite::MicroInterpreter* interpreter = nullptr;
// TfLiteTensor* inputTensor = nullptr;
// TfLiteTensor* outputTensor = nullptr;

// float soh_forecast[OUTPUT_LEN];

// unsigned long lastScreenSwitch = 0;
// bool screenToggle = false;

// // ================= HELPERS (IDENTICAL TO CODE 1) =================
// void quantizeAndSetInput(const float *scaled_input, TfLiteTensor *tensor) {
//   if (tensor->type == kTfLiteInt8) {
//     float scale = tensor->params.scale;
//     int zero_point = tensor->params.zero_point;
//     for (int i = 0; i < INPUT_LEN; ++i) {
//       int32_t q = lroundf(scaled_input[i] / scale + zero_point);
//       if (q > INT8_MAX) q = INT8_MAX;
//       if (q < INT8_MIN) q = INT8_MIN;
//       tensor->data.int8[i] = (int8_t)q;
//     }
//   } else {
//     for (int i = 0; i < INPUT_LEN; ++i)
//       tensor->data.f[i] = scaled_input[i];
//   }
// }

// void dequantizeOutputToFloat(float *outFloat, TfLiteTensor *tensor, int len) {
//   if (tensor->type == kTfLiteInt8) {
//     float scale = tensor->params.scale;
//     int zero_point = tensor->params.zero_point;
//     for (int i = 0; i < len; ++i)
//       outFloat[i] = ((float)tensor->data.int8[i] - zero_point) * scale;
//   } else {
//     for (int i = 0; i < len; ++i)
//       outFloat[i] = tensor->data.f[i];
//   }
// }

// // ================= SETUP =================
// void setup() {
//   Serial.begin(115200);
//   Wire.begin();
//   delay(2000);

//   dht.begin();
//   ina219.init();
//   lcd.init();
//   lcd.setRGB(0, 128, 255);

//   // ===== Load model =====
//   model = tflite::GetModel(gmodel);
//   if (model->version() != TFLITE_SCHEMA_VERSION) {
//     Serial.println("Model schema mismatch!");
//     while (true);
//   }

//   static tflite::AllOpsResolver resolver;
//   static tflite::MicroInterpreter static_interpreter(
//     model, resolver, tensor_arena, kTensorArenaSize);
//   interpreter = &static_interpreter;

//   if (interpreter->AllocateTensors() != kTfLiteOk) {
//     Serial.println("AllocateTensors failed!");
//     while (true);
//   }

//   inputTensor = interpreter->input(0);
//   outputTensor = interpreter->output(0);

//   // ===== Scale input (IDENTICAL) =====
//   float scaled_input[INPUT_LEN];
//   float denom = SCALE_MAX - SCALE_MIN;
//   for (int i = 0; i < INPUT_LEN; ++i) {
//     float s = (inputValues[i] - SCALE_MIN) / denom;
//     if (s < 0.0f) s = 0.0f;
//     if (s > 1.0f) s = 1.0f;
//     scaled_input[i] = s;
//   }

//   quantizeAndSetInput(scaled_input, inputTensor);

//   if (interpreter->Invoke() != kTfLiteOk) {
//     Serial.println("Invoke failed!");
//     while (true);
//   }

//   // ===== Dequantize + rescale output (IDENTICAL) =====
//   float out_scaled[OUTPUT_LEN];
//   dequantizeOutputToFloat(out_scaled, outputTensor, OUTPUT_LEN);

//   for (int i = 0; i < OUTPUT_LEN; ++i)
//     soh_forecast[i] = out_scaled[i] * denom + SCALE_MIN;

//   Serial.println("HistoricalSOH,ForecastSOH");
// }

// // ================= LOOP =================
// void loop() {
//   float tempC = dht.readTemperature();
//   float currentA = ina219.getCurrent_mA();

//   float currentSOH = inputValues[INPUT_LEN - 1] * 100.0f;

//   if (millis() - lastScreenSwitch > 3000) {
//     screenToggle = !screenToggle;
//     lastScreenSwitch = millis();
//     lcd.clear();
//   }

//   if (!screenToggle) {
//     lcd.setCursor(0, 0);
//     lcd.print("T:");
//     if (isnan(tempC)) lcd.print("Err");
//     else lcd.print(tempC, 1);
//     lcd.print("C I:");
//     lcd.print(currentA, 2);

//     lcd.setCursor(0, 1);
//     lcd.print("SOH:");
//     lcd.print(currentSOH, 1);
//     lcd.print("%");
//   } else {
//     lcd.setCursor(0, 0);
//     lcd.print("SOH+10:");
//     lcd.print(soh_forecast[9] * 100.0f, 1);

//     lcd.setCursor(0, 1);
//     lcd.print("SOH+100:");
//     lcd.print(soh_forecast[99] * 100.0f, 1);
//   }

//   // ===== Serial Plotter (every 2 samples) =====
//   for (int i = 0; i < INPUT_LEN; i ++) {
//     Serial.print(inputValues[i] * 100.0f);
//     Serial.print(",");
//     Serial.println(soh_forecast[i] * 100.0f);
//     delay(20);
//   }

//   delay(20000);
// }


#include <Arduino.h>
#include <Wire.h>
#include <ArduinoBLE.h>

// ===== Sensors =====
#include <DHT.h>
#include <INA219_WE.h>

// ===== LCD =====
#include <DFRobot_RGBLCD1602.h>

// ===== TFLite =====
#include <Chirale_TensorFlowLite.h>
#include "model.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ================= BLE CONFIG =================
BLEService sohService("180C"); 
BLEFloatCharacteristic histChar("2A58", BLERead | BLENotify); // Historical
BLEFloatCharacteristic foreChar("2A59", BLERead | BLENotify); // Forecast
BLEByteCharacteristic updateReqChar("2A5A", BLEWrite);         // Trigger update

// ================= CONFIG =================
#define DHTPIN 3
#define DHTTYPE DHT11
const float SCALE_MIN = 0.8178250862438334f;
const float SCALE_MAX = 1.006409511628937f;
constexpr int INPUT_LEN = 100;
constexpr int OUTPUT_LEN = 100;
constexpr int kTensorArenaSize = 25000;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const float inputValues[INPUT_LEN] = {
  0.9608134745,0.9606085219,0.9604032350,0.9601976085,0.9599916364,
  0.9597853129,0.9595786321,0.9593715875,0.9591641728,0.9589563815,
  0.9587482068,0.9585396417,0.9583306793,0.9581213122,0.9579115330,
  0.9577013341,0.9574907079,0.9572796462,0.9570681410,0.9568561841,
  0.9566437669,0.9564308808,0.9562175171,0.9560036667,0.9557893206,
  0.9555744693,0.9553591034,0.9551432132,0.9549267888,0.9547098203,
  0.9544922973,0.9542742097,0.9540555467,0.9538362978,0.9536164519,
  0.9533959980,0.9531749248,0.9529532209,0.9527308747,0.9525078744,
  0.9522842081,0.9520598635,0.9518348284,0.9516090902,0.9513826363,
  0.9511554537,0.9509275295,0.9506988504,0.9504694030,0.9502391736,
  0.9500081485,0.9497763138,0.9495436552,0.9493101584,0.9490758089,
  0.9488405919,0.9486044925,0.9483674957,0.9481295860,0.9478907481,
  0.9476509662,0.9474102244,0.9471685066,0.9469257965,0.9466820776,
  0.9464373332,0.9461915465,0.9459447002,0.9456967770,0.9454477594,
  0.9451976296,0.9449463697,0.9446939613,0.9444403862,0.9441856255,
  0.9439296605,0.9436724719,0.9434140405,0.9431543466,0.9428933704,
  0.9426310917,0.9423674902,0.9421025454,0.9418362363,0.9415685417,
  0.9412994404,0.9410289106,0.9407569303,0.9404834774,0.9402085292,
  0.9399320630,0.9396540556,0.9393744837,0.9390933233,0.9388105506,
  0.9385261411,0.9382400700,0.9379523124,0.9376628428,0.9373716354
};

// ================= GLOBALS =================
DHT dht(DHTPIN, DHTTYPE);
INA219_WE ina219;
#define LCD_ADDR 0x3E
#define RGB_ADDR 0x2D
DFRobot_RGBLCD1602 lcd(RGB_ADDR, 16, 2, &Wire, LCD_ADDR);

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* inputTensor = nullptr;
TfLiteTensor* outputTensor = nullptr;

float soh_forecast[OUTPUT_LEN];
unsigned long lastScreenSwitch = 0;
bool screenToggle = false;

// Quantize input for TFLite model
void quantizeAndSetInput(const float *scaled_input, TfLiteTensor *tensor) {
  if (tensor->type == kTfLiteInt8) {
    float scale = tensor->params.scale;
    int zero_point = tensor->params.zero_point;
    for (int i = 0; i < INPUT_LEN; ++i) {
      int32_t q = lroundf(scaled_input[i] / scale + zero_point);
      q = constrain(q, INT8_MIN, INT8_MAX);
      tensor->data.int8[i] = (int8_t)q;
    }
  } else {
    for (int i = 0; i < INPUT_LEN; ++i) tensor->data.f[i] = scaled_input[i];
  }
}

// Dequantize output from TFLite model
void dequantizeOutputToFloat(float *outFloat, TfLiteTensor *tensor, int len) {
  if (tensor->type == kTfLiteInt8) {
    float scale = tensor->params.scale;
    int zero_point = tensor->params.zero_point;
    for (int i = 0; i < len; ++i)
      outFloat[i] = ((float)tensor->data.int8[i] - zero_point) * scale;
  } else {
    for (int i = 0; i < len; ++i) outFloat[i] = tensor->data.f[i];
  }
}

void setup() {
  Serial.begin(115200);
  Wire.begin();

  if (!BLE.begin()) { Serial.println("BLE failed!"); while(1); }
  BLE.setLocalName("Nano33_SOH_Split");
  BLE.setAdvertisedService(sohService);

  sohService.addCharacteristic(histChar);
  sohService.addCharacteristic(foreChar);
  sohService.addCharacteristic(updateReqChar);

  BLE.addService(sohService);
  BLE.advertise();

  dht.begin();
  ina219.init();
  lcd.init();
  lcd.setRGB(0, 128, 255);

  model = tflite::GetModel(gmodel);
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  interpreter->AllocateTensors();
  inputTensor = interpreter->input(0);
  outputTensor = interpreter->output(0);

  // Scale input
  float scaled_input[INPUT_LEN];
  float denom = SCALE_MAX - SCALE_MIN;
  for (int i = 0; i < INPUT_LEN; ++i) {
    float s = (inputValues[i] - SCALE_MIN) / denom;
    scaled_input[i] = constrain(s, 0.0f, 1.0f);
  }

  quantizeAndSetInput(scaled_input, inputTensor);
  interpreter->Invoke();

  float out_scaled[OUTPUT_LEN];
  dequantizeOutputToFloat(out_scaled, outputTensor, OUTPUT_LEN);
  for (int i = 0; i < OUTPUT_LEN; ++i)
    soh_forecast[i] = out_scaled[i] * denom + SCALE_MIN;
}

void loop() {
  BLEDevice central = BLE.central();
  float tempC = dht.readTemperature();
  float currentA = ina219.getCurrent_mA();
  float currentSOH = inputValues[INPUT_LEN - 1] * 100.0f;

  // Toggle LCD screen every 3s
  if (millis() - lastScreenSwitch > 3000) {
    screenToggle = !screenToggle;
    lastScreenSwitch = millis();
    lcd.clear();
  }

  if (!screenToggle) {
    lcd.setCursor(0,0); lcd.print("T:"); lcd.print(tempC,1); lcd.print("C I:"); lcd.print(currentA,1); lcd.print("mA");
    lcd.setCursor(0,1); lcd.print("SOH:"); lcd.print(currentSOH,2); lcd.print("%");
  } else {
    lcd.setCursor(0,0); lcd.print("SOH+10:"); lcd.print(soh_forecast[9]*100.0f,2); lcd.print("%");
    lcd.setCursor(0,1); lcd.print("SOH+100:"); lcd.print(soh_forecast[99]*100.0f,2); lcd.print("%");
  }
  delay(500);

  // Only send data when update is requested
  if (central && central.connected()) {
      if (updateReqChar.written()) {
          Serial.println("Sending snapshot data...");

          const int batchSize = 2;       // 2 points per batch
          const int delayPerBatch = 100;  // 50 ms pause between batches

          for (int i = 0; i < INPUT_LEN; i += batchSize) {
              // Send batch of historical points
              for (int j = 0; j < batchSize && (i + j) < INPUT_LEN; ++j) {
                  histChar.writeValue(inputValues[i + j] * 100.0f);
              }
              delay(50);
              // Send batch of forecast points
              for (int j = 0; j < batchSize && (i + j) < INPUT_LEN; ++j) {
                  foreChar.writeValue(soh_forecast[i + j] * 100.0f);
              }

              // Small delay to prevent BLE buffer overflow
              delay(delayPerBatch);
          }

          Serial.println("Snapshot sent!");
      }
  }

  delay(500); // small main loop delay
}

