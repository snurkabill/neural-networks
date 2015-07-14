package net.snurkabill.neuralnetworks.utilities.datagenerator;

import net.snurkabill.neuralnetworks.newdata.database.NewDataItem;

import java.util.List;

public class DataGenerator {

    private final List<NGaussianDistribution> distributions;

    public DataGenerator(List<NGaussianDistribution> generators) {
        this.distributions = generators;
    }

    public NewDataItem generateNext() {
        return distributions.get(AbstractGenerator.globalRandom.nextInt(distributions.size())).generateNextVector();
    }
}
