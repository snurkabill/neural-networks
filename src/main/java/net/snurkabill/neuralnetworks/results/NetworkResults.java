package net.snurkabill.neuralnetworks.results;

public abstract class NetworkResults {

    private int numOfTrainedItems;
    private double globalError;
    private long milliseconds;

    public NetworkResults(int numOfTrainedItems, double globalError, long milliseconds) {
        this.numOfTrainedItems = numOfTrainedItems;
        this.globalError = globalError;
        this.milliseconds = milliseconds;
    }

    public long getMilliseconds() {
        return milliseconds;
    }

    public void setMilliseconds(long milliseconds) {
        this.milliseconds = milliseconds;
    }

    public double getGlobalError() {
        return globalError;
    }

    public void setGlobalError(double globalError) {
        this.globalError = globalError;
    }

    public int getNumOfTrainedItems() {
        return numOfTrainedItems;
    }

    public void setNumOfTrainedItems(int numOfTrainedItems) {
        this.numOfTrainedItems = numOfTrainedItems;
    }

    public double getAverageGlobalError() {
        return globalError / numOfTrainedItems;
    }

    public abstract double getComparableSuccess();
}
