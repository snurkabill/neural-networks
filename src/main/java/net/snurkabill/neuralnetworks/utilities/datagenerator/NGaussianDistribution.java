package net.snurkabill.neuralnetworks.utilities.datagenerator;

import net.snurkabill.neuralnetworks.newdata.database.NewDataItem;

import java.util.List;

public class NGaussianDistribution {

    private final List<GaussianGenerator> generators;

    public NGaussianDistribution(List<GaussianGenerator> generators) {
        this.generators = generators;
    }

    public int getNumOfDimensiion() {
        return generators.size();
    }

    public NewDataItem generateNextVector() {
        double[] item = new double[this.getNumOfDimensiion()];
        for (int i = 0; i < item.length; i++) {
            item[i] = generators.get(i).generateNext();
        }
        return new NewDataItem(item);
    }
}
