package net.snurkabill.neuralnetworks.neuralnetwork.feedforward.backpropagative;

import net.snurkabill.neuralnetworks.heuristic.BasicHeuristic;
import net.snurkabill.neuralnetworks.heuristic.FeedForwardHeuristic;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.FeedForwardNetwork;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.transferfunction.TransferFunctionCalculator;
import net.snurkabill.neuralnetworks.weights.weightfactory.WeightsFactory;

import java.util.List;

public abstract class BackPropagative extends FeedForwardNetwork {

    protected FeedForwardHeuristic heuristic;

    protected final double[][] gradients;
    protected final double[][][] deltaWeights;
    protected final double[][][] weightsEta;

    public BackPropagative(String name, List<Integer> topology, WeightsFactory wFactory,
                           FeedForwardHeuristic heuristic, TransferFunctionCalculator transferFunction) {
        super(name, topology, wFactory, transferFunction);
        this.gradients = new double[this.topology.length][];
        for (int i = 0; i < this.topology.length; i++) {
            this.gradients[i] = new double[this.topology[i] + 1];
        }
        this.deltaWeights = new double[this.topology.length - 1][][];
        this.weightsEta = new double[this.topology.length - 1][][];
        for (int i = 0; i < this.topology.length - 1; i++) {
            this.deltaWeights[i] = new double[this.neuronOutputValues[i].length][];
            this.weightsEta[i] = new double[this.neuronOutputValues[i].length][];
            for (int j = 0; j < neuronOutputValues[i].length; j++) {
                this.deltaWeights[i][j] = new double[this.topology[i + 1]];
                this.weightsEta[i][j] = new double[this.topology[i + 1]];
            }
        }
        this.heuristic = heuristic;
        this.setLearningRate();
    }

    protected abstract void calcGradients(double[] targetValues);

    protected abstract boolean isBatchFull();

    private void setLearningRate() {
        LOGGER.debug("Setting eta to each weight");
        for (int i = 0; i < topology.length - 1; i++) {
            for (int j = 0; j < neuronOutputValues[i].length; j++) {
                for (int k = 0; k < topology[i + 1]; k++) {
                    weightsEta[i][j][k] = heuristic.learningRate;
                }
            }
        }
    }

    @Override
    public void trainNetwork(double[] targetValues) {
        // TODO: vector length into manager
        calcGradients(targetValues);
        if (isBatchFull()) {
            weightsCalculation();
        }
    }

    @Override
    public BasicHeuristic getHeuristic() {
        return heuristic;
    }

    @Override
    public void setHeuristic(BasicHeuristic heuristic) {
        if (heuristic instanceof FeedForwardHeuristic) {
            this.heuristic = (FeedForwardHeuristic) heuristic;
        } else {
            throw new IllegalArgumentException("Heuristic is not instance of: " + FeedForwardHeuristic.class.getName());
        }
    }

    // ********************************************************************************
    // methods for updating weights
    // ********************************************************************************
    protected void weightsCalculation() {
        if (heuristic.dynamicKillingWeights) {
            if (heuristic.dynamicBoostingEtas) {
                complexUpdateWeights();
            } else {
                weightsKillingUpdateWeights();
            }
        } else if (heuristic.dynamicBoostingEtas) {
            etaBoostUpdateWeights();
        } else {
            simpleUpdateWeights();
        }
    }

    private double calcDeltaWeight(int i, int j, int k) {
        double oldDeltaWeight = deltaWeights[i][k][j];
        deltaWeights[i][k][j] = weightsEta[i][k][j] * neuronOutputValues[i][k] * gradients[i + 1][j]
                + heuristic.momentum * oldDeltaWeight;
        return oldDeltaWeight;
    }

    private void simpleUpdateWeights() {
        for (int i = numOfLayers - 1; i > 0; i--) {
            for (int j = 0; j < topology[i]; j++) {
                for (int k = 0; k < neuronOutputValues[i - 1].length; k++) {
                    calcDeltaWeight(i - 1, j, k);
                    weights[i - 1][k][j] += deltaWeights[i - 1][k][j];
                }
            }
        }
    }

    private void complexUpdateWeights() {
        for (int i = numOfLayers - 1; i > 0; i--) {
            for (int j = 0; j < topology[i]; j++) {
                for (int k = 0; k < neuronOutputValues[i - 1].length; k++) {
                    double oldDeltaWeight = calcDeltaWeight(i - 1, j, k);
                    if (oldDeltaWeight * deltaWeights[i - 1][k][j] >= 0) {
                        weightsEta[i - 1][k][j] *= heuristic.BOOSTING_ETA_COEFF;
                    } else {
                        weightsEta[i - 1][k][j] *= heuristic.KILLING_ETA_COEFF;
                    }
                    weights[i - 1][k][j] = heuristic.l2RegularizationConstant
                            * (weights[i - 1][k][j] + deltaWeights[i - 1][k][j]);
                }
            }
        }
    }

    private void etaBoostUpdateWeights() {
        for (int i = numOfLayers - 1; i > 0; i--) {
            for (int j = 0; j < topology[i]; j++) {
                for (int k = 0; k < neuronOutputValues[i - 1].length; k++) {
                    double oldDeltaWeight = deltaWeights[i - 1][k][j];
                    deltaWeights[i - 1][k][j] = weightsEta[i - 1][k][j] * neuronOutputValues[i - 1][k] * gradients[i][j]
                            + heuristic.momentum * oldDeltaWeight;
                    if (oldDeltaWeight * deltaWeights[i - 1][k][j] >= 0) {
                        weightsEta[i - 1][k][j] *= heuristic.BOOSTING_ETA_COEFF;
                    } else {
                        weightsEta[i - 1][k][j] *= heuristic.KILLING_ETA_COEFF;
                    }
                    weights[i - 1][k][j] += deltaWeights[i - 1][k][j];
                }
            }
        }
    }

    private void weightsKillingUpdateWeights() {
        for (int i = numOfLayers - 1; i > 0; i--) {
            for (int j = 0; j < topology[i]; j++) {
                for (int k = 0; k < neuronOutputValues[i - 1].length; k++) {
                    calcDeltaWeight(i - 1, j, k);
                    weights[i - 1][k][j] = heuristic.l2RegularizationConstant
                            * (weights[i - 1][k][j] + deltaWeights[i - 1][k][j]);
                }
            }
        }
    }

    @Override
    public String determineWorkingName() {
        String workingName = this.getName();
        workingName.concat(this.heuristic + " ");
        return workingName;
    }
}
