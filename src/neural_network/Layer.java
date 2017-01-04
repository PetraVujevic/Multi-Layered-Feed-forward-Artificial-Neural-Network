package neural_network;

import java.util.Iterator;

public class Layer implements Iterable<Neuron> {
    Neuron[] neurons;
    int size;

    public Layer(int size) {
        this.size = size;
        this.neurons = new Neuron[size];
    }

    public double[] output() {
        double[] o = new double[neurons.length];
        for (int i = 0; i < neurons.length; i++) {
            o[i] = neurons[i].y;
        }
        return o;
    }

    @Override
    public Iterator<Neuron> iterator() {
        return new Iterator<Neuron>() {
            int i = 0;

            @Override
            public boolean hasNext() {
                return i < neurons.length;
            }

            @Override
            public Neuron next() {
                return neurons[i++];
            }
        };
    }
}
