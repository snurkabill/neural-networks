package net.snurkabill.neuralnetworks;

import net.snurkabill.neuralnetworks.data.Database;
import net.snurkabill.neuralnetworks.data.MnistDatasetReader;
import net.snurkabill.neuralnetworks.feedforwardnetwork.core.FeedForwardNeuralNetwork;
import net.snurkabill.neuralnetworks.feedforwardnetwork.heuristic.HeuristicParamsFFNN;
import net.snurkabill.neuralnetworks.feedforwardnetwork.manager.FeedForwardNetworkOfflineManager;
import net.snurkabill.neuralnetworks.feedforwardnetwork.trasferfunction.HyperbolicTangens;
import net.snurkabill.neuralnetworks.results.SupervisedTestResults;
import net.snurkabill.neuralnetworks.weights.weightfactory.GaussianRndWeightsFactory;
import org.junit.Assert;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Created by snurkabill on 10/12/14.
 */
public class BasicTest {

    public static final Logger LOGGER = LoggerFactory.getLogger("MnistExample FFNN");
    public static final String FILE_PATH = "src/test/resources/mnist/";
    public static final int INPUT_SIZE = 784;
    public static final int OUTPUT_SIZE = 10;
    public static final int[] hiddenLayers = {200};

    public static MnistDatasetReader getReader() {
        File labels = new File(FILE_PATH + "train-labels-idx1-ubyte.gz");
        File images = new File(FILE_PATH + "train-images-idx3-ubyte.gz");
        MnistDatasetReader reader;
        try {
            reader = new MnistDatasetReader(labels, images);
        } catch (IOException ex) {
            throw new IllegalArgumentException(ex);
        }
        return reader;
    }


    @Test
    public void complexTest() {
        long seed = 0;
        MnistDatasetReader reader = getReader();

        List<Integer> topology = new ArrayList<>();
        topology.add(INPUT_SIZE);
        for (int i = 0; i < hiddenLayers.length; i++) {
            topology.add(hiddenLayers[i]);
        }
        topology.add(OUTPUT_SIZE);

        double weightsScale = 0.001;
        FeedForwardNeuralNetwork network = new FeedForwardNeuralNetwork(topology,
                new GaussianRndWeightsFactory(weightsScale, seed), "BASIC TEST",
                HeuristicParamsFFNN.createDefaultHeuristic(), new HyperbolicTangens(), seed);

        Database database = new Database(seed, reader.getTrainingData(), reader.getTestingData(), "MNIST");
        FeedForwardNetworkOfflineManager manager = new FeedForwardNetworkOfflineManager(Collections.singletonList(network), database, false);

        int sizeOfTrainingBatch = 10_000;
        manager.trainNetwork(sizeOfTrainingBatch);
        SupervisedTestResults results = (SupervisedTestResults) manager.testNetworkOnWholeTestingDataset().get(0);

        Assert.assertTrue(results.getSuccessPercentage() > 84);
    }
}
