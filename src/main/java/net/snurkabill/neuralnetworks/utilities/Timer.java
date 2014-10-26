package net.snurkabill.neuralnetworks.utilities;

public class Timer {

    private long startingTime;
    private long totalTime;

    public void startTimer() {
        startingTime = System.currentTimeMillis();
    }

    public void stopTimer() {
        totalTime = System.currentTimeMillis() - startingTime;
    }

    public long getTotalTime() {
        return totalTime;
    }

    public long secondsSpent() {
        return totalTime / 1000;
    }

    public double samplesPerSec(int samples) {
        return (double) samples / (totalTime / 1000.0);
    }
}
