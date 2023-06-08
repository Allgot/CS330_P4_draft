package com.team50.carcrashdetector.micInference;

import android.content.Context;
import android.media.AudioRecord;
import android.util.Log;

import org.tensorflow.lite.support.audio.TensorAudio;
import org.tensorflow.lite.support.label.Category;
import org.tensorflow.lite.task.audio.classifier.AudioClassifier;
import org.tensorflow.lite.task.audio.classifier.Classifications;

import java.io.IOException;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;

public class CrashMicClassifier {
    TensorAudio tensor;
    AudioClassifier classifier;
    AudioRecord recorder;

    private DetectorListener detectorListener;
    private TimerTask task;

    public void initialize(Context context) {
        try {
            this.classifier = AudioClassifier.createFromFile(context, "yamnet_classification.tflite");
            audioInitialize();
            startRecording();
            startInferencing();
        } catch (IOException e) {
            Log.d("CrashMicClassifier", "Load failed");
        }
    }

    private void audioInitialize() {
        this.tensor = classifier.createInputTensorAudio();
        recorder = classifier.createAudioRecord();
    }

    private void startRecording() {
        Log.d("CrashMicClassifier", "Recording begins");
        recorder.startRecording();
    }

    private void stopRecording() {
        Log.d("CrashMicClassifier", "Recording ends");
        recorder.stop();
    }

    public boolean inference() {
        tensor.load(recorder);
        List<Classifications> output = classifier.classify(tensor);

        return output.get(0).getCategories().stream().anyMatch(
                (Category category) -> category.getLabel().equals("Whack") && category.getScore() > 0.5
        );
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
        task.cancel();
        task = null;
    }

    public interface DetectorListener {
        void onResults(boolean isCrash);
    }

    public void setDetectorListener(DetectorListener listener) {
        this.detectorListener = listener;
    }
}
