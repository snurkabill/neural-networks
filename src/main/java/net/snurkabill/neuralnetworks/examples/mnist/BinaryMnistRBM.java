package net.snurkabill.neuralnetworks.examples.mnist;

import net.snurkabill.neuralnetworks.data.database.ClassFullDatabase;
import net.snurkabill.neuralnetworks.data.mnist.MnistDatasetReader;
import net.snurkabill.neuralnetworks.heuristic.BoltzmannMachineHeuristic;
import net.snurkabill.neuralnetworks.managers.NetworkManager;
import net.snurkabill.neuralnetworks.managers.boltzmannmodel.UnsupervisedRBMManager;
import net.snurkabill.neuralnetworks.managers.boltzmannmodel.validator.ProbabilisticAssociationVectorValidator;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.RestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.impl.binaryvisible.BinaryToBinaryRBM;
import net.snurkabill.neuralnetworks.weights.weightfactory.GaussianRndWeightsFactory;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;

public class BinaryMnistRBM extends Canvas {
    static int border = 10; // 10px

    static int count = 0;

    final RestrictedBoltzmannMachine machine;
    NetworkManager manager_rbm;

    public BinaryMnistRBM() {

        long seed = 0;
        double weightsScale = 0.01;
        MnistDatasetReader reader = MnistExampleFFNN.getReader(10000, true);
        ClassFullDatabase classFullDatabase = new ClassFullDatabase(seed, reader.getTrainingData(), reader.getTestingData(), "MNIST", false);

        BoltzmannMachineHeuristic heuristic = BoltzmannMachineHeuristic.createStartingHeuristicParams();
        heuristic.learningRate = 0.01;
        heuristic.constructiveDivergenceIndex = 1;
        heuristic.batchSize = 30;
        heuristic.momentum = 0.3;
        heuristic.temperature = 1;
        machine = new BinaryToBinaryRBM("RBM 1",
                (classFullDatabase.getSizeOfVector()), 200,
                new GaussianRndWeightsFactory(weightsScale, seed),
                heuristic, seed);
        manager_rbm = new UnsupervisedRBMManager(machine, classFullDatabase, seed, null,
                new ProbabilisticAssociationVectorValidator(1));
    }

    void learn() {
        manager_rbm.supervisedTraining(1000);
    }

    public void update() {
        learn();
        repaint();
    }

    public void paint(Graphics graphics__) {

        BufferedImage in = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);

        WritableRaster r = in.getRaster();
        graphics__.drawImage(in, border, border, null);

        double weights[][] = machine.getWeights()[0];

        int offset = border;
        int buf = 28 + border + border;
        for (int i = 0; i < weights[0].length; i++) {
            if (i % 10 == 0) {
                offset = border;
                buf += border + 56;
            }
            int[] start = new int[28 * 28];
            for (int j = 0; j < start.length; j++)
                start[j] = weights[j][i] > 0.0 ?
                        (int) (Math.round(weights[j][i] * 255)) << 8 :
                        (int) ((Math.round(Math.abs(weights[j][i] * 255)) << 16));

            BufferedImage out = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);

            r = out.getRaster();
            r.setDataElements(0, 0, 28, 28, start);

            //Resize
            BufferedImage newImage = new BufferedImage(56, 56, BufferedImage.TYPE_INT_RGB);

            Graphics2D g2 = newImage.createGraphics();
            try {
                g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                        RenderingHints.VALUE_INTERPOLATION_BICUBIC);
                g2.clearRect(0, 0, 56, 56);
                g2.drawImage(out, 0, 0, 56, 56, null);
            } finally {
                g2.dispose();
            }
            graphics__.drawImage(newImage, buf, offset, null);

            offset += border + 28 * 2;
        }
    }


    public static void start() {
        JFrame frame = new JFrame("MINST Draw");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(800, 600);

        BinaryMnistRBM cnvs = new BinaryMnistRBM();

        cnvs.setSize(1024, 768);
        frame.add(cnvs);

        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);

        while (true) {
            cnvs.update();

            try {
                count++;
                if (count > 1000)
                    Thread.sleep(200000);
            } catch (InterruptedException e) {
            }
        }
    }
}
