package net.snurkabill.neuralnetworks.results;

public class UnsupervisedTestResults extends TestResults {

    private double avgRightPercentageOfUnknownPart;
    private double percentageSizeOfUnknownSize;

    public UnsupervisedTestResults(int numOfTestedItems, double globalError, long milliseconds,
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
    public double getSuccess() {
        return avgRightPercentageOfUnknownPart;
    }
}
