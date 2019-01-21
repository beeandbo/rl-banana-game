### Learning Algorithm

The agent implements a version of the Deep Q Network described in
https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf.

Unlike the neural network described in the paper, this agent works with
a smaller number of sensor vectors and thus uses a fully connected network
as the underlying model.

It uses fixed Q targets, though the target network is updated gradually
using a weighting hyper parameter TAU rather than a sudden update.  It also
uses a buffer of experiences for learning with another hyper parameter
controlling the buffer size (i.e. experience replay).

### Training

The following plots both the per episode score and the 100 episode moving
average:

![alt Plot]('scores.png')

The final average score was 13.52
