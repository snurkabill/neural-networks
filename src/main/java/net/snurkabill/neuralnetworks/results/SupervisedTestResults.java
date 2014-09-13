package net.snurkabill.neuralnetworks.results;

public class SupervisedTestResults extends BasicTestResults {
	
	private double successPercentage;

    public SupervisedTestResults(int numOfTestedItems, double globalError, long milliseconds, double successPercentage) {
        super(numOfTestedItems, globalError, milliseconds);
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
