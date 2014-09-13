package net.snurkabill.neuralnetworks.benchmark.RBM.Vector;

import java.util.ArrayList;
import java.util.List;

public class MyBasicFilter implements VectorFilter {

    private int[] numbers;
    private int lastAccepted = 0;

    public MyBasicFilter(int... numbers) {
        this.numbers = numbers;
    }

    @Override
    public List<Tuple> filter(List<Tuple> tuples) {
        List<Tuple> filteredTuples = new ArrayList<>();
        for (Tuple tuple : tuples) {
            if(tuple.getNum() == numbers[lastAccepted]) {
                filteredTuples.add(tuple);
                lastAccepted++;
            }
            if(lastAccepted > numbers.length) {
                break;
            }
        }
        return filteredTuples;
    }
}
