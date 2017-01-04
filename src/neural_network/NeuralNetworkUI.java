package neural_network;

import java.awt.BasicStroke;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.GridLayout;
import java.awt.Point;
import java.awt.Shape;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.geom.Path2D;
import java.util.ArrayList;

import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JTextArea;
import javax.swing.SwingUtilities;
import javax.swing.WindowConstants;

import neural_network.gesture.GestureProcessor;

@SuppressWarnings("serial")
public class NeuralNetworkUI extends JFrame {
    public static final String alpha = "alpha";
    public static final String beta = "beta";
    public static final String gamma = "gamma";
    public static final String delta = "delta";
    public static final String epsilon = "epsilon";
    public static final String samplesFile = System.getProperty("user.dir")
            + "/samples.txt";

    private NeuralNetwork nn = NeuralNetwork.getInstance();
    private GestureProcessor gp = new GestureProcessor();
    private Canvas canvas;

    JButton buttonAlpha;
    JButton buttonBeta;
    JButton buttonGamma;
    JButton buttonDelta;
    JButton buttonEpsilon;

    public NeuralNetworkUI() {
        setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
        initGUIandActions();
    }

    private void initGUIandActions() {
        setSize(900, 600);
        getContentPane().setLayout(new BorderLayout());
        canvas = new Canvas(nn, gp);
        getContentPane().add(canvas, BorderLayout.CENTER);

        buttonAlpha = new JButton(alpha);
        buttonAlpha.addActionListener(new ActionListener() {

            @Override
            public void actionPerformed(ActionEvent e) {
                canvas.setGestureId(alpha);
            }
        });

        buttonBeta = new JButton(beta);
        buttonBeta.addActionListener(new ActionListener() {

            @Override
            public void actionPerformed(ActionEvent e) {
                canvas.setGestureId(beta);
            }
        });

        buttonGamma = new JButton(gamma);
        buttonGamma.addActionListener(new ActionListener() {

            @Override
            public void actionPerformed(ActionEvent e) {
                canvas.setGestureId(gamma);
            }
        });

        buttonDelta = new JButton(delta);
        buttonDelta.addActionListener(new ActionListener() {

            @Override
            public void actionPerformed(ActionEvent e) {
                canvas.setGestureId(delta);
            }
        });

        buttonEpsilon = new JButton(epsilon);
        buttonEpsilon.addActionListener(new ActionListener() {

            @Override
            public void actionPerformed(ActionEvent e) {
                canvas.setGestureId(epsilon);
            }
        });

        JButton buttonDone = new JButton("Save gestures");
        buttonDone.addActionListener(new ActionListener() {

            @Override
            public void actionPerformed(ActionEvent e) {
                gp.storeAllGesturesSamples();
            }
        });

        JPanel panelButtons = new JPanel(new GridLayout(0, 6));
        panelButtons.add(buttonAlpha);
        panelButtons.add(buttonBeta);
        panelButtons.add(buttonGamma);
        panelButtons.add(buttonDelta);
        panelButtons.add(buttonEpsilon);
        panelButtons.add(buttonDone);
        getContentPane().add(panelButtons, BorderLayout.SOUTH);

        JLabel labelFile = new JLabel("File with samples");
        JTextArea textAreaFile = new JTextArea(samplesFile);
        JLabel labelM = new JLabel(
                "M (Number of points in the gesture representation)");
        JTextArea textAreaM = new JTextArea("10");
        JLabel labelArchitecture = new JLabel("Architecture");
        JTextArea textAreaArchitecture = new JTextArea("20x5x3x5");
        JLabel labelLearningRate = new JLabel("Learning rate");
        JTextArea textAreaLearningRate = new JTextArea("0.01");
        JComboBox<String> cbAlgorithm = new JComboBox<String>();
        JLabel labelMinError = new JLabel("Minimum error");
        JTextArea textAreaMinError = new JTextArea("0.02");
        cbAlgorithm.addItem("Backpropagation");
        cbAlgorithm.addItem("Stohastic Backpropagation");
        cbAlgorithm.addItem("Mini-batch Backpropagation");
        JButton buttonTrain = new JButton("Train network");

        buttonTrain.addActionListener(new ActionListener() {

            @Override
            public void actionPerformed(ActionEvent e) {
                nn.train(textAreaFile.getText(), textAreaM.getText(),
                        textAreaArchitecture.getText(),
                        textAreaLearningRate.getText(),
                        textAreaMinError.getText(),
                        cbAlgorithm.getSelectedIndex());
            }
        });

        JPanel panelAlgorithmParameters = new JPanel(new GridLayout(0, 1));
        panelAlgorithmParameters.add(labelFile);
        panelAlgorithmParameters.add(textAreaFile);
        panelAlgorithmParameters.add(labelM);
        panelAlgorithmParameters.add(textAreaM);
        panelAlgorithmParameters.add(labelArchitecture);
        panelAlgorithmParameters.add(textAreaArchitecture);
        panelAlgorithmParameters.add(labelLearningRate);
        panelAlgorithmParameters.add(textAreaLearningRate);
        panelAlgorithmParameters.add(labelMinError);
        panelAlgorithmParameters.add(textAreaMinError);
        panelAlgorithmParameters.add(cbAlgorithm);
        panelAlgorithmParameters.add(buttonTrain);
        getContentPane().add(panelAlgorithmParameters, BorderLayout.LINE_END);
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(new Runnable() {

            @Override
            public void run() {
                new NeuralNetworkUI().setVisible(true);
            }
        });
    }

    class Canvas extends JPanel {
        private Point start;
        private Point stop;
        private Shape shape;
        private ArrayList<Point> g;
        private String gestureId;
        NeuralNetwork nn;
        GestureProcessor gp;

        public Canvas(NeuralNetwork nn, GestureProcessor gp) {
            this.nn = nn;
            this.gp = gp;
            addListeners();
        }

        public void setGestureId(String gestureId) {
            this.gestureId = gestureId;
        }

        private void addListeners() {
            addMouseListener(new MouseAdapter() {
                public void mousePressed(MouseEvent event) {
                    start = event.getPoint();
                    g = new ArrayList<Point>();
                    g.add(start);
                    Path2D path = new Path2D.Double();
                    shape = path;
                }

                public void mouseReleased(MouseEvent event) {
                    Path2D path = (Path2D) shape;
                    try {
                        path.closePath();
                    } catch (Exception ingore) {
                    }
                    shape = path;

                    if (nn.trained) {
                        try {
                            String gesture = nn.infer(gp.getRepresentation(g));
                            showResult(gesture);
                        } catch (NeuralNetworkException e) {
                            JOptionPane
                                    .showInputDialog("Train the network first by clicking 'Train network' button.");
                        }
                    } else {
                        gp.addGesture(g, gestureId);
                    }
                    repaint();
                }

                private void showResult(String result) {
                    buttonAlpha.setForeground(Color.black);
                    buttonBeta.setForeground(Color.black);
                    buttonGamma.setForeground(Color.black);
                    buttonDelta.setForeground(Color.black);
                    buttonEpsilon.setForeground(Color.black);

                    switch (result) {
                    case alpha:
                        buttonAlpha.setForeground(Color.magenta);
                        break;
                    case beta:
                        buttonBeta.setForeground(Color.magenta);
                        break;
                    case gamma:
                        buttonGamma.setForeground(Color.magenta);
                        break;
                    case delta:
                        buttonDelta.setForeground(Color.magenta);
                        break;
                    case epsilon:
                        buttonEpsilon.setForeground(Color.magenta);
                        break;
                    default:
                        break;
                    }
                }
            });

            addMouseMotionListener(new MouseAdapter() {
                public void mouseDragged(MouseEvent event) {
                    stop = event.getPoint();
                    g.add(stop);
                    Path2D path = (Path2D) shape;
                    path.moveTo(start.x, start.y);
                    path.lineTo(stop.x, stop.y);
                    // System.out.println(stop.x + " " + stop.y);
                    shape = path;
                    start = stop;

                    repaint();
                }
            });
        }

        public void paintComponent(Graphics gc) {
            super.paintComponent(gc);
            Graphics2D g2 = (Graphics2D) gc;

            if (start != null && stop != null) {
                BasicStroke stroke = new BasicStroke(1);
                Shape strokedShape = stroke.createStrokedShape(shape);
                g2.draw(strokedShape);
                g2.fill(strokedShape);
            }
        }
    }
}
