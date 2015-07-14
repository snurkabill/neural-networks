package net.snurkabill.neuralnetworks.math.linear;

public abstract class GeneralProjection {

    protected final int vectorSize;

    public GeneralProjection(int vectorSize) {
        this.vectorSize = vectorSize;
    }

    public boolean checkSize(double[] vector) {
        return vector.length == vectorSize;
    }

    public abstract double[] transform(double[] vector);
}
