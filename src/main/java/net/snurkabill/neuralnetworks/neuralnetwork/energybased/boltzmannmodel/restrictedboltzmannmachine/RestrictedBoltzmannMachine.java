package net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine;

import net.snurkabill.neuralnetworks.heuristic.Heuristic;
import net.snurkabill.neuralnetworks.heuristic.HeuristicRBM;
import net.snurkabill.neuralnetworks.neuralnetwork.FeedForwardable;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.BoltzmannMachine;
import net.snurkabill.neuralnetworks.utilities.Utilities;
import net.snurkabill.neuralnetworks.weights.weightfactory.WeightsFactory;

public abstract class RestrictedBoltzmannMachine extends BoltzmannMachine implements FeedForwardable {

    private static final int FIX_VISIBLE = 0;

    protected final int sizeOfVisibleVector;
    protected final int sizeOfHiddenVector;

    protected final double[] visibleNeurons;
    protected final double[] hiddenNeurons;
    protected final double[] visibleBias;
    protected final double[] hiddenBias;
    protected final double[][] weights;
    private final double[][] deltaWeights;
    private final double[] deltaVisibleBias;
    private final double[] deltaHiddenBias;

    private HeuristicRBM heuristic;
    private int numOfVectorsInBatch;
    private final double[][] diffWeights;
    private final double[][] reconWeights;
    private final double[] diffVisibleBiases;
    private final double[] reconVisibleBiases;
    private final double[] diffHiddenBiases;
    private final double[] reconHiddenBiases;

    public RestrictedBoltzmannMachine(String name, int numOfVisible, int numOfHidden, WeightsFactory wFactory,
                                      HeuristicRBM heuristic, long seed) {
        super(name, seed);
        if (numOfVisible <= 0 || numOfHidden <= 0) {
            throw new IllegalArgumentException("Wrong size of RBM");
        }
        this.setHeuristic(heuristic);
        this.sizeOfVisibleVector = numOfVisible;
        this.sizeOfHiddenVector = numOfHidden;
        this.visibleNeurons = new double[numOfVisible];
        this.visibleBias = new double[numOfVisible];
        this.hiddenNeurons = new double[numOfHidden];
        this.hiddenBias = new double[numOfHidden];
        this.diffWeights = new double[sizeOfVisibleVector][sizeOfHiddenVector];
        this.reconWeights = new double[sizeOfVisibleVector][sizeOfHiddenVector];
        this.weights = new double[sizeOfVisibleVector][sizeOfHiddenVector];
        this.deltaWeights = new double[sizeOfVisibleVector][sizeOfHiddenVector];
        this.diffVisibleBiases = new double[sizeOfVisibleVector];
        this.reconVisibleBiases = new double[sizeOfVisibleVector];
        this.diffHiddenBiases = new double[sizeOfHiddenVector];
        this.reconHiddenBiases = new double[sizeOfHiddenVector];
        this.deltaVisibleBias = new double[sizeOfVisibleVector];
        this.deltaHiddenBias = new double[sizeOfHiddenVector];
        wFactory.setWeights(weights);
        wFactory.setWeights(visibleBias);
        wFactory.setWeights(hiddenBias);
        LOGGER.info("RBM " + numOfVisible + " -> " + numOfHidden + " created!");
    }

    public double[][] createLayerForFFNN() {
        double[][] weightsLayer = new double[this.sizeOfVisibleVector + 1][sizeOfHiddenVector];
        for (int i = 0; i < this.sizeOfVisibleVector; i++) {
            System.arraycopy(weights[i], 0, weightsLayer[i], 0, this.sizeOfHiddenVector);
        }
        System.arraycopy(this.hiddenBias, 0, weightsLayer[this.sizeOfVisibleVector], 0, sizeOfHiddenVector);
        return weightsLayer;
    }

    public double[] getVisibleNeurons() {
        return visibleNeurons;
    }

    public int getSizeOfVisibleVector() {
        return sizeOfVisibleVector;
    }

    public int getSizeOfHiddenVector() {
        return sizeOfHiddenVector;
    }

    public void setVisibleNeurons(double[] visibleNeurons) {
        if (visibleNeurons.length != sizeOfVisibleVector) {
            throw new IllegalArgumentException("VisibleNeurons have different size!");
        }
        LOGGER.trace("Setting visible neurons");
        System.arraycopy(visibleNeurons, 0, this.visibleNeurons, 0, sizeOfVisibleVector);
    }

    public void setHiddenNeurons(double[] hiddenNeurons) {
        if (hiddenNeurons.length != sizeOfHiddenVector) {
            throw new IllegalArgumentException("HiddenNeurons have different size!");
        }
        LOGGER.trace("Setting hidden neurons");
        System.arraycopy(hiddenNeurons, 0, this.hiddenNeurons, 0, sizeOfHiddenVector);
    }

    public double[] getHiddenNeurons() {
        return hiddenNeurons;
    }

    protected double calcHiddenPotential(int index) {
        double potential = 0.0;
        for (int j = 0; j < sizeOfVisibleVector; j++) {
            potential += weights[j][index] * visibleNeurons[j];
        }
        potential += hiddenBias[index];
        return potential;
    }

    protected double calcVisiblePotential(int index) {
        double potential = 0.0;
        for (int i = 0; i < sizeOfHiddenVector; i++) {
            potential += weights[index][i] * hiddenNeurons[i];
        }
        potential += visibleBias[index];
        return potential;
    }

    protected abstract void calcVisibleNeurons();

    protected abstract void calcHiddenNeurons();

    public void trainMachine(double[] inputVals) {
        //double[] hiddenProbbabilities = new double[sizeOfHiddenVector];
        // ================================================================================
        // data
        // ================================================================================
        runMachine(FIX_VISIBLE, inputVals);

        /*for (int i = 0; i < sizeOfHiddenVector; i++) {
            hiddenProbbabilities[i] = calcProbabilityOfPositiveOutput(calcHiddenPotential(i));
        }*/
        for (int i = 0; i < sizeOfVisibleVector; i++) {
            for (int j = 0; j < sizeOfHiddenVector; j++) {
                this.diffWeights[i][j] += visibleNeurons[i] * hiddenNeurons[j]; //hiddenProbbabilities[j];
            }
        }
        for (int i = 0; i < sizeOfHiddenVector; i++) {
            this.diffHiddenBiases[i] += hiddenNeurons[i];
        }
        // ================================================================================
        // reconstruction
        // ================================================================================
        runMachine(heuristic.constructiveDivergenceIndex, inputVals);
        /*for (int i = 0; i < sizeOfHiddenVector; i++) {
            hiddenProbbabilities[i] = calcProbabilityOfPositiveOutput(calcHiddenPotential(i));
        }*/
        for (int i = 0; i < sizeOfVisibleVector; i++) {
            for (int j = 0; j < sizeOfHiddenVector; j++) {
                this.reconWeights[i][j] += visibleNeurons[i] * hiddenNeurons[j]; // hiddenProbbabilities[j];
            }
        }
        for (int i = 0; i < sizeOfVisibleVector; i++) {
            this.reconVisibleBiases[i] += inputVals[i] - visibleNeurons[i];
        }
        for (int i = 0; i < sizeOfHiddenVector; i++) {
            this.reconHiddenBiases[i] += hiddenNeurons[i];
        }

        numOfVectorsInBatch++;
        if (numOfVectorsInBatch >= heuristic.batchSize) {
            for (int i = 0; i < sizeOfVisibleVector; i++) {
                for (int j = 0; j < sizeOfHiddenVector; j++) {
                    this.diffWeights[i][j] /= heuristic.batchSize;
                    this.reconWeights[i][j] /= heuristic.batchSize;
                }
            }
            for (int i = 0; i < sizeOfVisibleVector; i++) {
                this.reconVisibleBiases[i] /= heuristic.batchSize;
            }
            for (int i = 0; i < sizeOfHiddenVector; i++) {
                this.diffHiddenBiases[i] /= heuristic.batchSize;
                this.reconHiddenBiases[i] /= heuristic.batchSize;
            }
            for (int i = 0; i < sizeOfVisibleVector; i++) {
                for (int j = 0; j < sizeOfHiddenVector; j++) {
                    deltaWeights[i][j] = (heuristic.learningRate / super.temperature) *
                            (this.diffWeights[i][j] - this.reconWeights[i][j]) +
                            heuristic.momentum * deltaWeights[i][j];
                    weights[i][j] -= deltaWeights[i][j];
                }
            }
            for (int i = 0; i < sizeOfVisibleVector; i++) {
                deltaVisibleBias[i] = (heuristic.learningRate / super.temperature) * reconVisibleBiases[i] +
                        heuristic.momentum * deltaVisibleBias[i];
                visibleBias[i] -= deltaVisibleBias[i];
            }
            for (int i = 0; i < sizeOfHiddenVector; i++) {
                deltaHiddenBias[i] = (heuristic.learningRate / super.temperature) *
                        (diffHiddenBiases[i] - reconHiddenBiases[i]) +
                        heuristic.momentum * deltaHiddenBias[i];
                hiddenBias[i] -= deltaHiddenBias[i];
            }
            for (int i = 0; i < sizeOfVisibleVector; i++) {
                for (int j = 0; j < sizeOfHiddenVector; j++) {
                    this.diffWeights[i][j] = 0.0;
                    this.reconWeights[i][j] = 0.0;
                }
            }
            for (int i = 0; i < sizeOfVisibleVector; i++) {
                reconVisibleBiases[i] = 0.0;
            }
            for (int i = 0; i < sizeOfHiddenVector; i++) {
                reconHiddenBiases[i] = 0.0;
                diffHiddenBiases[i] = 0.0;
            }
            numOfVectorsInBatch = 0;
        }
    }

    public void trainMachine() {
        this.trainMachine(visibleNeurons);
    }

    private void runMachine(int iterations, double[] inputVals) {
        LOGGER.trace("Machine runs {} iterations", iterations);
        setVisibleNeurons(inputVals);
        calcHiddenNeurons();
        for (int i = 0; i < iterations; i++) {
            machineStepInverse();
        }
    }

    public abstract double[] reconstructNext();

    protected void machineStepInverse() {
        calcVisibleNeurons();
        calcHiddenNeurons();
    }

    protected void machineStep() {
        calcHiddenNeurons();
        calcVisibleNeurons();
    }

    public double[] activateHiddenNeurons() {
        calcHiddenNeurons();
        return this.getHiddenNeurons();
    }

    public double[] activateVisibleNeurons() {
        calcVisibleNeurons();
        return this.getVisibleNeurons();
    }

    @Override
    public void calculateNetwork(double[] inputVector) {
        this.setVisibleNeurons(inputVector);
        this.machineStep();
    }

    @Override
    public double[] getOutputValues() {
        double[] outputValues = new double[this.sizeOfVisibleVector];
        System.arraycopy(this.visibleNeurons, 0, outputValues, 0, this.sizeOfVisibleVector);
        return outputValues;
    }

    @Override
    public int getSizeOfOutputVector() {
        return this.getSizeOfInputVector();
    }

    @Override
    public int getSizeOfInputVector() {
        return this.getSizeOfVisibleVector();
    }

    @Override
    public void getOutputValues(double[] outputValues) {
        System.arraycopy(this.visibleNeurons, 0, outputValues, 0, this.sizeOfVisibleVector);
    }

    @Override
    public Heuristic getHeuristic() {
        return this.heuristic;
    }

    @Override
    public void setHeuristic(Heuristic heuristic) {
        //TODO: check heuristic values
        if (heuristic instanceof HeuristicRBM) {
            this.heuristic = (HeuristicRBM) heuristic;
            super.setTemperature(this.heuristic.temperature);
        } else {
            throw new IllegalArgumentException("Heuristic " + heuristic + " is not instance of " +
                    HeuristicRBM.class.getName());
        }
    }

    @Override
    protected double calcOutputVectorError(double[] targetValues) {
        return Utilities.calcError(targetValues, visibleNeurons);
    }

    @Override
    public void trainNetwork(double[] targetValues) {
        this.trainMachine(targetValues);
    }

    @Override
    public void feedForward(double[] inputVector) {
        this.setVisibleNeurons(inputVector);
        this.activateHiddenNeurons();
    }

    @Override
    public double[] getForwardedValues() {
        return this.getHiddenNeurons();
    }

    public double[][] getWeights() {
        double[][] weightsSnapshot = new double[this.sizeOfVisibleVector + 1][this.sizeOfHiddenVector];
        for (int i = 0; i < this.sizeOfVisibleVector; i++) {
            System.arraycopy(this.weights[i], 0, weightsSnapshot[i], 0, this.sizeOfHiddenVector);
        }
        System.arraycopy(hiddenBias, 0, weightsSnapshot[sizeOfVisibleVector], 0, this.sizeOfHiddenVector);
        return weightsSnapshot;
    }

    @Override
    public String determineWorkingName() {
        String workingName = super.determineWorkingName();
        workingName += " [ " + visibleNeurons + " -> " + hiddenNeurons + " ] ";
        workingName += this.heuristic;
        return workingName;
    }
}
