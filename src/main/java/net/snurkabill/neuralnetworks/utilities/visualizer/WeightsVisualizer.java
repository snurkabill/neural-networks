package net.snurkabill.neuralnetworks.utilities.visualizer;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;

public class WeightsVisualizer extends Canvas {

/*    private static final int border = 5;

    private double data[];

    private final double[][] weights;
    private final int firstLayerSize;
    private final int secondLayerSize;
    private final int sqrtOfFirstLayerSize;
    private final int sqrtOfSecondLayerSize;

    public WeightsVisualizer(double[][] weights) {
        this.weights = weights;
        this.firstLayerSize = weights.length;
        this.secondLayerSize = weights[0].length;
        this.sqrtOfFirstLayerSize = ((int)Math.sqrt(firstLayerSize)) + 1;
        this.sqrtOfSecondLayerSize = ((int)Math.sqrt(secondLayerSize)) + 1;

        data = new double[]`

        JFrame frame = new JFrame("MINST Draw");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(800, 600);
        this.setSize(1000, 700);
        frame.add(this);
        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }

    public void paint(Graphics graphics__) {

        BufferedImage in = new BufferedImage(dr.cols, dr.rows, BufferedImage.TYPE_INT_RGB);


        WritableRaster r = in.getRaster();
        r.setDataElements(0, 0, dr.cols, dr.rows, trainItem.data);
        graphics__.drawImage(in, border, border, null);

        int offset = border;
        synchronized (outputs) {
            for (int[] output : outputs) {
                BufferedImage out = new BufferedImage(dr.cols, dr.rows, BufferedImage.TYPE_INT_RGB);


                r = out.getRaster();
                r.setDataElements(0, 0, dr.cols, dr.rows, output);

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
                graphics__.drawImage(newImage, border * 2 + 28, offset, null);

                offset += border + dr.rows * 2;
            }

            int buf = 28 + border + border;
            for (int i = 0; i < rbm.weights.length; i++) {
                if (i % 10 == 0) {
                    offset = border;
                    buf += border + 56;
                }

                int[] start = new int[dr.cols * dr.rows];
                for (int j = 0; j < start.length; j++)
                    start[j] = rbm.weights[i].get(j) > 0 ?
                            (Math.round(rbm.weights[i].get(j) * 255)) << 8 :
                            ((Math.round(Math.abs(rbm.weights[i].get(j)) * 255)) << 16);

                BufferedImage out = new BufferedImage(dr.cols, dr.rows, BufferedImage.TYPE_INT_RGB);

                r = out.getRaster();
                r.setDataElements(0, 0, dr.cols, dr.rows, start);

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

                offset += border + dr.rows * 2;
            }
        }
    }
    }*/
}
