package net.snurkabill.neuralnetworks.utilities.datagenerator;

public class GaussianGenerator extends AbstractGenerator {

    private final double modifiedVariance;
    private final double mean;

    public GaussianGenerator(double variance, double mean) {
        this.modifiedVariance = Math.sqrt(variance);
        this.mean = mean;
    }

    public double getModifiedVariance() {
        return modifiedVariance;
    }

    public double getMean() {
        return mean;
    }

    @Override
    public double generateNext() {
        return (AbstractGenerator.globalRandom.nextGaussian() / modifiedVariance) + mean ;
    }
}
