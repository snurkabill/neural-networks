package net.snurkabill.neuralnetworks.visualize.boltzmann;

import net.snurkabill.neuralnetworks.data.database.Database;
import net.snurkabill.neuralnetworks.data.mnist.MnistDatasetReader;
import net.snurkabill.neuralnetworks.examples.mnist.MnistExampleFFNN;
import net.snurkabill.neuralnetworks.heuristic.HeuristicRBM;
import net.snurkabill.neuralnetworks.managers.NetworkManager;
import net.snurkabill.neuralnetworks.managers.boltzmannmodel.UnsupervisedRBMManager;
import net.snurkabill.neuralnetworks.managers.boltzmannmodel.validator.ProbabilisticAssociationVectorValidator;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.impl.BinaryRestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.weights.weightfactory.GaussianRndWeightsFactory;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;

public class VisualizedBinaryRBM extends Canvas {

    private final BinaryRestrictedBoltzmannMachine machine;
    private final NetworkManager manager_rbm;
    private final Database database;


    public VisualizedBinaryRBM() {
        int seed = 0;
        MnistDatasetReader reader = MnistExampleFFNN.getReader(10_000, true);
        this.database = new Database(seed, reader.getTrainingData(), reader.getTestingData(), "MNIST", false);

        HeuristicRBM heuristic = new HeuristicRBM();
        heuristic.batchSize = 100;
        heuristic.learningRate = 0.001;
        heuristic.momentum = 0.5;
        this.machine = new BinaryRestrictedBoltzmannMachine("RBM 1",
                database.getSizeOfVector(), 100,
                new GaussianRndWeightsFactory(0.0, seed),
                //new SmartGaussianRndWeightsFactory(new SigmoidFunction(), seed),
                heuristic, seed);
        this.manager_rbm = new UnsupervisedRBMManager(machine, database, seed, null,
                new ProbabilisticAssociationVectorValidator(1));
    }

    public void learn() {
        manager_rbm.supervisedTraining(10000);
    }

    public void update() {
        learn();
        repaint();
    }

    @Override
    public void paint(Graphics g) {
        int border = 10;

        int offset = border;

        synchronized (machine) {
            int buf = 28 + border + border;
            for (int i = 0; i < machine.getHiddenNeurons().length; i++) {
                if (i % 10 == 0) {
                    offset = border;
                    buf += border + 56;
                }

                int[] start = new int[784];
                for (int j = 0; j < 784; j++)
                    start[j] = machine.getWeights()[j][i] > 0 ? /*255 : -255;*/
                            (int)(Math.round(machine.getWeights()[j][i] * 255)) << 8 :
                            (int)((Math.round(Math.abs(machine.getWeights()[j][i]) * 255)) << 16);

                BufferedImage out = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);

                WritableRaster r;
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
                g.drawImage(newImage, buf, offset, null);

                offset += border + 28 * 2;
            }
        }
    }

    public static void start() {
        JFrame frame = new JFrame("MINST Draw");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(800, 600);

        VisualizedBinaryRBM cnvs = new VisualizedBinaryRBM();


        cnvs.setSize(1024, 768);
        frame.add(cnvs);

        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);

        int count = 0;
        while (true) {
            cnvs.update();
            try {
                count++;
                //Thread.sleep(200);

                if (count < 0)
                    Thread.sleep(2000);

            } catch (InterruptedException e) {
            }
        }
    }
}
