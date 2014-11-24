package net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine;

import net.snurkabill.neuralnetworks.heuristic.Heuristic;
import net.snurkabill.neuralnetworks.heuristic.HeuristicRBM;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.BoltzmannMachine;
import net.snurkabill.neuralnetworks.utilities.Utilities;
import net.snurkabill.neuralnetworks.weights.weightfactory.WeightsFactory;

public abstract class RestrictedBoltzmannMachine extends BoltzmannMachine {

    private static final int FIX_VISIBLE = 0;

    protected final int sizeOfVisibleVector;
    protected final int sizeOfHiddenVector;

    protected final double[] visibleNeurons;
    protected final double[] hiddenNeurons;
    protected final double[] visibleBias;
    protected final double[] hiddenBias;
    protected final double[][] weights;
    private final double[][] deltaWeights;

    private HeuristicRBM heuristic;

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
        this.weights = new double[numOfVisible][];
        for (int i = 0; i < numOfVisible; i++) {
            this.weights[i] = new double[numOfHidden];
        }
        this.deltaWeights = new double[numOfVisible][];
        for (int i = 0; i < numOfVisible; i++) {
            this.deltaWeights[i] = new double[numOfHidden];
        }
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
        LOGGER.trace("Training machine started! InputVectorLength {}, sizeOfVisibleVector {}"
                , inputVals.length, sizeOfVisibleVector);
        if (inputVals.length != sizeOfVisibleVector) {
            throw new IllegalArgumentException("Wrong size of inputVector");
        }
        double[][] diffWeights = runMachine(FIX_VISIBLE, inputVals);
        double[][] reconWeights = runMachine(heuristic.constructiveDivergenceIndex, inputVals);
        for (int i = 0; i < sizeOfVisibleVector; i++) {
            for (int j = 0; j < sizeOfHiddenVector; j++) {
                diffWeights[i][j] -= reconWeights[i][j];
            }
        }
        for (int i = 0; i < sizeOfVisibleVector; i++) {
            for (int j = 0; j < sizeOfHiddenVector; j++) {
                deltaWeights[i][j] = (heuristic.learningRate / super.temperature) * diffWeights[i][j] +
                        heuristic.momentum * deltaWeights[i][j];
                weights[i][j] -= deltaWeights[i][j];
            }
        }
    }

    public void trainMachine() {
        this.trainMachine(visibleNeurons);
    }

    private double[][] runMachine(int iterations, double[] inputVals) {
        LOGGER.trace("Machine runs {} iterations", iterations);
        double[][] weightExpected = new double[sizeOfVisibleVector][sizeOfHiddenVector];
        for (int i = 0; i < heuristic.numOfTrainingIterations; i++) {
            constrastiveDivergence(iterations, inputVals);
            for (int j = 0; j < sizeOfVisibleVector; j++) {
                for (int k = 0; k < sizeOfHiddenVector; k++) {
                    weightExpected[j][k] += visibleNeurons[j] * hiddenNeurons[k];
                }
            }
        }
        for (int i = 0; i < sizeOfVisibleVector; i++) {
            for (int j = 0; j < sizeOfHiddenVector; j++) {
                weightExpected[i][j] /= heuristic.numOfTrainingIterations;
            }
        }
        return weightExpected;
    }

    public abstract double[] reconstructNext();

    protected void constrastiveDivergence(int index, double[] inputVals) {
        setVisibleNeurons(inputVals);
        calcHiddenNeurons();
        for (int i = 0; i < index; i++) {
            machineStepInverse();
        }
    }

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
        if (targetValues.length != sizeOfVisibleVector) {
            throw new IllegalArgumentException();
        }
        return Utilities.calcError(targetValues, visibleNeurons);
    }

    @Override
    public void trainNetwork(double[] targetValues) {
        this.trainMachine(targetValues);
    }

    @Override
    public String determineWorkingName() {
        String workingName = super.determineWorkingName();
        workingName += " [ " + visibleNeurons + " -> " + hiddenNeurons + " ] ";
        workingName += this.heuristic;
        return workingName;
    }
}
