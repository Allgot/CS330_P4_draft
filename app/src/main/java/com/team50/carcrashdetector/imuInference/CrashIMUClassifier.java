package com.team50.carcrashdetector.imuInference;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.util.Log;

import com.team50.carcrashdetector.ml.Imumodel;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.util.Arrays;
import java.util.Timer;
import java.util.TimerTask;

public class CrashIMUClassifier {
    Imumodel classifier;
    SensorManager recorder;

    private DetectorListener detectorListener;
    private TimerTask task;
    private Sensor accSensor, gyrSensor;
    private AccSensorEventListener accListener;
    private GyrSensorEventListener gyrListener;

    public void initialize(Context context) {
        try {
            classifier = Imumodel.newInstance(context);
            imuInitialize(context);
            startRecording();
            startInferencing();
        } catch (IOException e) {
            Log.d("CrashIMUClassifier", "Load failed");
        }
    }

    private void imuInitialize(Context context) {
        this.recorder = (SensorManager) context.getSystemService(Context.SENSOR_SERVICE);
        this.accSensor = recorder.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        this.accListener = new AccSensorEventListener();
        this.gyrSensor = recorder.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        this.gyrListener = new GyrSensorEventListener();
    }

    static class AccSensorEventListener implements SensorEventListener {
        float[][] window = new float[33][3]; // 야매긴함;

        public AccSensorEventListener() {
            float[] initializer = {0.0f, 0.0f, 0.0f};
            Arrays.fill(this.window, initializer);
        }

        @Override
        public void onSensorChanged(SensorEvent event) {
            float accX = event.values[0];
            float accY = event.values[1];
            float accZ = event.values[2];
            float[] window_item = {accX, accY, accZ};

            window = Arrays.copyOfRange(window, 1, 33);
            window = Arrays.copyOf(window, 33);
            window[32] = window_item;
        }

        @Override
        public void onAccuracyChanged(Sensor sensor, int i) {}
    }

    static class GyrSensorEventListener implements SensorEventListener {
        float[][] window = new float[33][3];

        public GyrSensorEventListener() {
            float[] initializer = {0.0f, 0.0f, 0.0f};
            Arrays.fill(this.window, initializer);
        }

        @Override
        public void onSensorChanged(SensorEvent event) {
            float gyrX = event.values[0];
            float gyrY = event.values[1];
            float gyrZ = event.values[2];
            float[] window_item = {gyrX, gyrY, gyrZ};

            window = Arrays.copyOfRange(window, 1, 33);
            window = Arrays.copyOf(window, 33);
            window[32] = window_item;
        }

        @Override
        public void onAccuracyChanged(Sensor sensor, int i) {}
    }

    private void startRecording() {
        Log.d("CrashIMUClassifier", "Records begins");
        recorder.registerListener(accListener, accSensor, SensorManager.SENSOR_DELAY_NORMAL);
        recorder.registerListener(gyrListener, gyrSensor, SensorManager.SENSOR_DELAY_NORMAL);
    }

    private void stopRecording() {
        Log.d("CrashIMUClassifier", "Recording ends");
        recorder.unregisterListener(accListener);
        recorder.unregisterListener(gyrListener);
    }

    public boolean inference() {
        TensorBuffer input = TensorBuffer.createFixedSize(new int[]{1, 33, 6}, DataType.FLOAT32);
        float[][] concat = new float[33][6];
        float[] flattenedConcat = new float[33 * 6];

        for (int i = 0; i < 33; i++) {
            System.arraycopy(this.accListener.window[i], 0, concat[i], 0, 3);
            System.arraycopy(this.gyrListener.window[i], 0, concat[i], 3, 3);
        }

        for (int i = 0; i < 33; i++) {
            System.arraycopy(concat[i], 0, flattenedConcat, 6 * i, 6);
        }

        input.loadArray(flattenedConcat);
        TensorBuffer outputs = classifier.process(input).getOutputFeature0AsTensorBuffer();
        float[] scores = outputs.getFloatArray();

        // 0 - Not crash; 1 - Crash
        return scores[1] > 0.5;
    }

    public void startInferencing() {
        if (task == null) {
            Timer timer = new Timer();
            task = new TimerTask() {
                @Override
                public void run() {
                    detectorListener.onResults(inference());
                }
            };

            timer.scheduleAtFixedRate(task, 0, 33L);
        }
    }

    public void stopInferencing() {
        if (task != null) {
            task.cancel();
            task = null;
        }
    }

    public interface DetectorListener {
        void onResults(boolean isCrash);
    }

    public void setDetectorListener(DetectorListener listener) {
        this.detectorListener = listener;
    }
}
