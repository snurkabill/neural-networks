package net.snurkabill.neuralnetworks.data.mnist;

import org.junit.Test;

import java.io.File;
import java.io.IOException;

public class MnistDataReaderTest {

    @Test
    public void mnistSetSizes() throws IOException {
        File labels = new File("src/test/resources/mnist/train-labels-idx1-ubyte.gz");
        File images = new File("src/test/resources/mnist/train-images-idx3-ubyte.gz");
        MnistDatasetReader reader = new MnistDatasetReader(labels, images, 60_000, false);


        reader.getValidationData();
        reader.getTrainingData();
        reader.getTestingData();


    }

}
