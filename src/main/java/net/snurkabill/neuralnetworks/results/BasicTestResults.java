package net.snurkabill.neuralnetworks.results;

public class BasicTestResults {
	
	private int numOfTestedItems;
	private double globalError;
	private long milliseconds;

    public BasicTestResults(int numOfTestedItems, double globalError, long milliseconds) {
        this.numOfTestedItems = numOfTestedItems;
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

	public int getNumOfTestedItems() {
		return numOfTestedItems;
	}

	public void setNumOfTestedItems(int numOfTestedItems) {
		this.numOfTestedItems = numOfTestedItems;
	}
	
	public double getAverageGlobalError() {
		return globalError / numOfTestedItems;
	}
}
