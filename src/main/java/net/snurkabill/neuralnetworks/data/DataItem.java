package net.snurkabill.neuralnetworks.data;

public class DataItem {
	
	public double[] data;
	
	public DataItem(double[] data) {
		this.data = new double [data.length];
		for (int i = 0; i < data.length; i++) {
			this.data[i] = data[i];
		}
	}
}
