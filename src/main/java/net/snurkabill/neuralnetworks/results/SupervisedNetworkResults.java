package net.snurkabill.neuralnetworks.results;

public class SupervisedNetworkResults extends NetworkResults {

    private double successPercentage;

    public SupervisedNetworkResults(int numOfTrainedItems, double globalError, long milliseconds, double successPercentage) {
        super(numOfTrainedItems, globalError, milliseconds);
        this.successPercentage = successPercentage;
    }

    public double getSuccessPercentage() {
        return successPercentage;
    }

    public void setSuccessPercentage(double successPercentage) {
        this.successPercentage = successPercentage;
    }

    @Override
    public double getSuccess() {
        return successPercentage;
    }
}
