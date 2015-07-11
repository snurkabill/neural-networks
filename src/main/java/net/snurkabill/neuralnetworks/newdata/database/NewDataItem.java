package net.snurkabill.neuralnetworks.newdata.database;

public class NewDataItem {

    public double[] data;
    public double[] target;

    public NewDataItem(double[] data) {
        this(data, null);
    }

    public NewDataItem(double[] data, double[] target) {
        this.data = data;
        this.target = target;
    }

    public boolean hasTarget() {
        return target != null;
    }

    public int getDataVectorSize() {
        return data.length;
    }

    public int getTargetSize() {
        return target == null ? 0 : target.length;
    }

}
