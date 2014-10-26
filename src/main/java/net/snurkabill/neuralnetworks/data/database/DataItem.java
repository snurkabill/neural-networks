package net.snurkabill.neuralnetworks.data.database;

public class DataItem {

    public double[] data;

    public DataItem(double[] data) {
        this.data = new double[data.length];
		System.arraycopy(data, 0, this.data, 0, data.length);
    }
}
