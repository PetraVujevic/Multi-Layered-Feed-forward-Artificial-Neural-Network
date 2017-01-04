package neural_network.gesture;

import java.awt.Point;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import static neural_network.NeuralNetworkUI.alpha;
import static neural_network.NeuralNetworkUI.beta;
import static neural_network.NeuralNetworkUI.gamma;
import static neural_network.NeuralNetworkUI.delta;
import static neural_network.NeuralNetworkUI.epsilon;
import static neural_network.NeuralNetworkUI.samplesFile;

public class GestureProcessor {
    private HashMap<String, ArrayList<ArrayList<Point>>> gesturesMap = new HashMap<String, ArrayList<ArrayList<Point>>>();
    private final int M = 10;

    public void addGesture(ArrayList<Point> g, String gestureId) {
        if (gestureId == null) {
            return;
        }
        if (!gesturesMap.containsKey(gestureId)) {
            gesturesMap.put(gestureId, new ArrayList<ArrayList<Point>>());
        }
        ArrayList<ArrayList<Point>> gestures = gesturesMap.get(gestureId);
        gestures.add(g);
    }

    public ArrayList<DecimalPoint> getRepresentation(ArrayList<Point> g) {
        DecimalPoint averagePoint = getAverage(g);
        // System.out.println(averagePoint.x + " " + averagePoint.y);
        ArrayList<DecimalPoint> locationFreeGesture = substract(g, averagePoint);
        double m = getMaxDistanceFromZero(locationFreeGesture);
        // System.out.println(m);
        // scale gesture
        locationFreeGesture = scale(locationFreeGesture, m);
        return getGestureRepresentation(locationFreeGesture);
    }

    private DecimalPoint getAverage(ArrayList<Point> g) {
        double sumX = 0;
        double sumY = 0;

        for (Point p : g) {
            sumX += p.x;
            sumY += p.y;
        }

        return new DecimalPoint(sumX / g.size(), sumY / g.size());
    }

    private ArrayList<DecimalPoint> substract(ArrayList<Point> g,
            DecimalPoint average) {
        ArrayList<DecimalPoint> translatedGesture = new ArrayList<DecimalPoint>();
        for (Point p : g) {
            DecimalPoint translated = new DecimalPoint(p.x -= average.x,
                    p.y -= average.y);
            translatedGesture.add(translated);
        }

        return translatedGesture;
    }

    private double getMaxDistanceFromZero(ArrayList<DecimalPoint> g) {
        double max = 0;
        for (DecimalPoint p : g) {
            double x = Math.abs(p.x);
            double y = Math.abs(p.y);
            max = Math.max(Math.max(x, y), max);
        }
        return max;
    }

    private ArrayList<DecimalPoint> scale(ArrayList<DecimalPoint> g,
            double scalingFactor) {
        for (DecimalPoint p : g) {
            p.x /= scalingFactor;
            p.y /= scalingFactor;
        }
        return g;
    }

    private ArrayList<DecimalPoint> getGestureRepresentation(
            ArrayList<DecimalPoint> g) {
        ArrayList<DecimalPoint> gr = new ArrayList<DecimalPoint>();
        double D = totalDistance(g);
        // System.out.println("D: " + D);
        DecimalPoint start = g.get(0);
        int i = 0;
        for (int k = 0; k < M; k++) {
            double l = (k * D) / (M - 1);
            // System.out.println("l: " + l);
            while (i < g.size() && start.distance(g.get(i)) <= l) {
                i++;
            }
            gr.add(g.get(i - 1));
        }
        return gr;
    }

    private double totalDistance(ArrayList<DecimalPoint> points) {
        double totalDistance = 0;
        for (int i = 0; i < points.size() - 1; i++) {
            totalDistance += points.get(i).distance(points.get(i + 1));
        }
        return totalDistance;
    }

    public void storeAllGesturesSamples() {
        storeGesturesSamples(alpha);
        storeGesturesSamples(beta);
        storeGesturesSamples(gamma);
        storeGesturesSamples(delta);
        storeGesturesSamples(epsilon);
    }

    private void storeGesturesSamples(String gestureId) {
        ArrayList<ArrayList<Point>> gestures = gesturesMap.get(gestureId);
        if (gestures != null) {
            for (ArrayList<Point> g : gestures) {
                ArrayList<DecimalPoint> gr = getRepresentation(g);
                store(gestureId, gr);
            }
        }
    }

    private void store(String gestureId, ArrayList<DecimalPoint> gr) {
        try {
            BufferedWriter out = new BufferedWriter(new FileWriter(samplesFile,
                    true));
            out.write("\n");
            for (DecimalPoint p : gr) {
                out.write(" " + p.x + " " + p.y);
            }
            switch (gestureId) {
            case alpha:
                out.write(" 1 0 0 0 0");
                break;
            case beta:
                out.write(" 0 1 0 0 0");
                break;
            case gamma:
                out.write(" 0 0 1 0 0");
                break;
            case delta:
                out.write(" 0 0 0 1 0");
                break;
            case epsilon:
                out.write(" 0 0 0 0 1");
                break;
            default:
                break;
            }
            out.close();
        } catch (IOException e) {
            System.out
                    .println("Exception occured while trying to open the file samples.txt.");
        }
    }
}
