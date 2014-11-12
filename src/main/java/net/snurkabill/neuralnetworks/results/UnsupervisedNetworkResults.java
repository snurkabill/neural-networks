package net.snurkabill.neuralnetworks.results;

public class UnsupervisedNetworkResults extends NetworkResults {

    private double avgRightPercentageOfUnknownPart;
    private double percentageSizeOfUnknownSize;

    public UnsupervisedNetworkResults(int numOfTestedItems, double globalError, long milliseconds,
                                      double avgRightPercentageOfUnknownPart, double percentageSizeOfUnknownSize) {
        super(numOfTestedItems, globalError, milliseconds);
        this.percentageSizeOfUnknownSize = percentageSizeOfUnknownSize;
        this.avgRightPercentageOfUnknownPart = avgRightPercentageOfUnknownPart;
    }

    public double getPercentageSizeOfUnknownSize() {
        return percentageSizeOfUnknownSize;
    }

    public void setPercentageSizeOfUnknownSize(double percentageSizeOfUnknownSize) {
        this.percentageSizeOfUnknownSize = percentageSizeOfUnknownSize;
    }

    public double getAvgRightPercentageOfUnknownPart() {
        return avgRightPercentageOfUnknownPart;
    }

    public void setAvgRightPercentageOfUnknownPart(double avgRightPercentageOfUnknownPart) {
        this.avgRightPercentageOfUnknownPart = avgRightPercentageOfUnknownPart;
    }

    @Override
    public double getComparableSuccess() {
        return avgRightPercentageOfUnknownPart;
    }
}
