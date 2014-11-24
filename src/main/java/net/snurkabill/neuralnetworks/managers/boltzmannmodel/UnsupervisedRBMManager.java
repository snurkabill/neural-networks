package net.snurkabill.neuralnetworks.managers.boltzmannmodel;

import net.snurkabill.neuralnetworks.data.database.DataItem;
import net.snurkabill.neuralnetworks.data.database.Database;
import net.snurkabill.neuralnetworks.data.database.LabelledItem;
import net.snurkabill.neuralnetworks.heuristic.calculators.HeuristicCalculator;
import net.snurkabill.neuralnetworks.managers.boltzmannmodel.validator.ProbabilisticAssociationVectorValidator;
import net.snurkabill.neuralnetworks.neuralnetwork.NeuralNetwork;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.RestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.results.UnsupervisedNetworkResults;

import java.util.Iterator;

public class UnsupervisedRBMManager extends RestrictedBoltzmannMachineManager {

    private final ProbabilisticAssociationVectorValidator validator;

    public UnsupervisedRBMManager(NeuralNetwork neuralNetwork, Database database, long seed,
                                  HeuristicCalculator heuristicCalculator,
                                  ProbabilisticAssociationVectorValidator validator) {
        super(neuralNetwork, database, seed, heuristicCalculator);
        this.validator = validator;
    }

    @Override
    protected void train(int numOfIterations) {
        for (int i = 0; i < numOfIterations; i++) {
            LabelledItem item = this.infiniteTrainingIterator.next();
            neuralNetwork.trainNetwork(item.data);
        }
    }

    @Override
    protected void test() {
        RestrictedBoltzmannMachine machine = (RestrictedBoltzmannMachine) neuralNetwork;
        globalError = 0.0;
        int all = 0;
        for (int _class = 0; _class < database.getNumberOfClasses(); _class++) {
            Iterator<DataItem> testingIterator = database.getTestingIteratorOverClass(_class);
            for (; testingIterator.hasNext(); ) {
                globalError += validator.validate(testingIterator.next().data, machine);
                all++;
            }
        }
        globalError /= all;
    }

    @Override
    protected void processResults() {
        super.results.add(new UnsupervisedNetworkResults(super.learnedVectorsBeforeTest,
                globalError, super.learningTimeBeforeTest, 0, 0));
    }
}
