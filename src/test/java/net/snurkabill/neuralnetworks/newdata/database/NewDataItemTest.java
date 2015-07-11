package net.snurkabill.neuralnetworks.newdata.database;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class NewDataItemTest {

    @Test
    public void constructionTest() {
        double[] array = new double[0];
        double[] array2 = new double[0];
        NewDataItem newDataItem = new NewDataItem(array,  array2);
        assertEquals(array2, newDataItem.target);
        assertEquals(array, newDataItem.data);
    }

    @Test
    public void constructionNullTargetTest() {
        double[] array = new double[0];
        NewDataItem newDataItem = new NewDataItem(array,  null);
        assertEquals(null, newDataItem.target);
        assertEquals(array, newDataItem.data);
    }

    @Test
    public void sizeTest() {
        double[] array = new double[1];
        double[] array2 = new double[2];
        NewDataItem newDataItem = new NewDataItem(array,  array2);
        assertEquals(array.length, newDataItem.getDataVectorSize());
        assertEquals(array2.length, newDataItem.getTargetSize());
    }

    @Test
    public void containsTarget() {
        double[] array = new double[1];
        double[] array2 = new double[2];
        NewDataItem newDataItem = new NewDataItem(array,  array2);
        assertTrue(newDataItem.hasTarget());
        newDataItem = new NewDataItem(array);
        assertFalse(newDataItem.hasTarget());
    }

}
