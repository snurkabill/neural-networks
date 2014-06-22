package net.snurkabill.neuralnetworks.results;

public class SupervisedTestResults extends BasicTestResults {
	
	private double successPercentage;

	public double getSuccessPercentage() {
		return successPercentage;
	}

	public void setSuccessPercentage(double successPercentage) {
		this.successPercentage = successPercentage;
	}
}
