package net.snurkabill.neuralnetworks.utilities;

public class Timer {

    private long startingTime;
    private long totalTime;
    private boolean isRunning;

    public boolean isRunning() {
        return isRunning;
    }

    public void startTimer() {
        startingTime = System.currentTimeMillis();
        isRunning = true;
    }

    public void stopTimer() {
        totalTime = System.currentTimeMillis() - startingTime;
        isRunning = false;
    }

    public long getTotalTime() {
        if(isRunning) {
            stopTimer();
        }
        return totalTime;
    }

    public long secondsSpent() {
        if(isRunning) {
            stopTimer();
        }
        return totalTime / 1000;
    }

    public double samplesPerSec(int samples) {
        return (double) samples / (getTotalTime() / 1000.0);
    }
}
