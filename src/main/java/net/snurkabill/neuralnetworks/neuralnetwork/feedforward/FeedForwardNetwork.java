package net.snurkabill.neuralnetworks.neuralnetwork.feedforward;

import net.snurkabill.neuralnetworks.neuralnetwork.FeedForwardable;
import net.snurkabill.neuralnetworks.neuralnetwork.NeuralNetwork;
import net.snurkabill.neuralnetworks.math.function.transferfunction.TransferFunctionCalculator;
import net.snurkabill.neuralnetworks.weights.weightfactory.WeightsFactory;

import java.util.List;

public abstract class FeedForwardNetwork extends NeuralNetwork implements FeedForwardable {

    private static final double ERROR_MODIFIER = 0.5;

    protected final TransferFunctionCalculator[] layerTransferFunctions;
    protected final int sizeOfInputVector;
    protected final int sizeOfOutputVector;
    protected final int numOfLayers;
    protected final int lastLayerIndex;
    protected final int[] topology;
    protected final double[][] neuronOutputValues;
    protected final double[][][] weights;

    public FeedForwardNetwork(String name, List<Integer> topology, List<TransferFunctionCalculator> layerTransferFunctions,
                              WeightsFactory wFactory) {
        super(name);
        if (topology.size() <= 2) {
            throw new IllegalArgumentException("Wrong topology of FFNN");
        }
        if(layerTransferFunctions.size() != topology.size() - 1) {
            throw new IllegalArgumentException("Wrong topology or layerCalculator sizes");
        }
        this.layerTransferFunctions = new TransferFunctionCalculator[layerTransferFunctions.size()];
        for (int i = 0; i < layerTransferFunctions.size(); i++) {
            this.layerTransferFunctions[i] = layerTransferFunctions.get(i);
        }
        this.numOfLayers = topology.size();
        this.topology = new int[numOfLayers];
        for (int i = 0; i < numOfLayers; i++) {
            this.topology[i] = topology.get(i);
        }
        this.lastLayerIndex = numOfLayers - 1;
        this.sizeOfInputVector = this.topology[0];
        this.sizeOfOutputVector = this.topology[lastLayerIndex];
        this.neuronOutputValues = new double[this.topology.length][];
        for (int i = 0; i < this.topology.length; i++) {
            this.neuronOutputValues[i] = new double[this.topology[i] + 1];
        }
        for (int i = 0; i < this.topology.length; i++) {
            for (int j = 0; j < this.neuronOutputValues[i].length; j++) {
                this.neuronOutputValues[i][j] = 1; // BIAS values
            }
        }
        this.weights = new double[topology.size() - 1][][];
        for (int i = 0; i < topology.size() - 1; i++) {
            this.weights[i] = new double[this.neuronOutputValues[i].length][];
            for (int j = 0; j < this.neuronOutputValues[i].length; j++) {
                this.weights[i][j] = new double[topology.get(i + 1)];
            }
        }
        wFactory.setWeights(this.weights);
        LOGGER.info("Created Feed forwardable neural network {}", topology.toString());
    }

    @Override
    public void feedForward(double[] inputVector) {
        System.arraycopy(inputVector, 0, neuronOutputValues[0], 0, sizeOfInputVector);
        feedForward();
    }

    protected void feedForward() {
        for (int i = 1; i < numOfLayers; i++) {
            for (int j = 0; j < topology[i]; j++) {
                double sumOfInputs = 0.0;
                for (int k = 0; k < neuronOutputValues[i - 1].length; k++) {
                    sumOfInputs += neuronOutputValues[i - 1][k] * weights[i - 1][k][j];
                }
                neuronOutputValues[i][j] = layerTransferFunctions[i - 1].calculateOutputValue(sumOfInputs);
            }
        }
    }

    @Override
    public TransferFunctionCalculator[] getTransferFunctions() {
        return this.layerTransferFunctions;
    }

    public int getFirstHighestValueIndex() {
        boolean isOnlyValueTheBiggest = false;
        int classIndex = Integer.MIN_VALUE;
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < sizeOfOutputVector; i++) {
            if (max == neuronOutputValues[lastLayerIndex][i]) {
                isOnlyValueTheBiggest = false;
            } else if (max < neuronOutputValues[lastLayerIndex][i]) {
                max = neuronOutputValues[lastLayerIndex][i];
                classIndex = i;
                isOnlyValueTheBiggest = true;
            }
        }
        return isOnlyValueTheBiggest ? classIndex : Integer.MIN_VALUE;
    }

    @Deprecated // not recomended to use
    public int getLastHighestValueIndex() {
        double max = this.layerTransferFunctions[lastLayerIndex - 1].getLowLimit();
        int bestIndex = -1;
        for (int i = 0; i < sizeOfOutputVector; i++) {
            if (neuronOutputValues[lastLayerIndex][i] >= max) {
                max = neuronOutputValues[lastLayerIndex][i];
                bestIndex = i;
            }
        }
        return bestIndex;
    }

    @Override
    public void calculateNetwork(double[] inputVector) {
        feedForward(inputVector);
    }

    @Override
    protected double calcOutputVectorError(double[] targetValues) {
        if (targetValues.length != sizeOfOutputVector) {
            throw new IllegalArgumentException();
        }
        double sum = 0.0;
        for (int i = 0; i < topology[lastLayerIndex]; i++) {
            double tmp = neuronOutputValues[lastLayerIndex][i] - targetValues[i];
            sum += (tmp * tmp);
        }
        return ERROR_MODIFIER * sum;
    }

    @Override
    public double[] getForwardedValues() {
        return getOutputValues();
    }

    @Override
    public void getOutputValues(double[] outputValues) {
        if (outputValues.length != sizeOfOutputVector) {
            throw new IllegalArgumentException();
        }
        System.arraycopy(neuronOutputValues[lastLayerIndex], 0, outputValues, 0, outputValues.length);
    }

    @Override
    public double[] getOutputValues() {
        double[] outputValues = new double[topology[lastLayerIndex]];
        System.arraycopy(neuronOutputValues[lastLayerIndex], 0, outputValues, 0, outputValues.length);
        return outputValues;
    }

    @Override
    public int getSizeOfOutputVector() {
        return this.sizeOfOutputVector;
    }

    @Override
    public int getSizeOfInputVector() {
        return this.sizeOfInputVector;
    }

    @Override
    public double[][][] getWeights() {
        return this.weights;
    }

    @Override
    public String determineWorkingName() {
        String workingName = super.determineWorkingName();
        workingName.concat(" { ");
        for (int i = 0; i < topology.length; i++) {
            workingName.concat(String.valueOf(topology[i]) + ", ");
        }
        workingName.concat(" } ");
        for (int i = 0; i < this.layerTransferFunctions.length; i++) {
            workingName.concat(", " + this.layerTransferFunctions[i].getName());
        }
        return workingName;
    }
}
