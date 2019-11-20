import torch
from torchvision.transforms import ToTensor, Normalize
from torch.utils.tensorboard import SummaryWriter


class TorchParametrization:
    def __init__(self, model, optimizer, storage_location, **kwargs):
        # model definition
        self.model = model
        self.optimizer = None

        # saving, restoring &  logging
        self.summary_writer = None
        self.last_loss = None

    def test(self, state):
        action = self.tf_session.run([self.parameters], feed_dict={
            self.state_tensor: state,
        })
        return action

    def reset(self):
        self.tf_session.run(tf.global_variables_initializer())
        self.tf_session.run(tf.local_variables_initializer())
        tf.train.global_step(self.tf_session, self.global_step)

    def train(self, state, action):
        summary, step, _, learning_loss = self.tf_session.run(
            [self.summary_merge, self.global_step, self.optimization_fn, self.loss_function],
            feed_dict={
                self.state_tensor: state,
                self.action_tensor: action
            }
        )
        self.summary_writer.add_summary(summary, step)
        self.last_loss = learning_loss
        return learning_loss

    def commit(self):
        self.tf_saver.save(self.tf_session, self.tf_checkpoint, global_step=self.global_step)

    def architecture(self):
        raise NotImplementedError()

    def prepare_for_test(self, input_shape, action_shape, storage_location):
        if not self.parameters:
            self._create(input_shape, action_shape)
            self._storing(storage_location, fail_on_empty=True)
            self.training = False

    def prepare_for_train(self, input_shape, output_shape, optimizer, storage_location):
        if not self.parameters:
            self.optimization_fn = optimizer.minimize(loss=self.loss_function, global_step=self.global_step)

            self.reset()

            self._logging(storage_location)
            self._storing(storage_location)

            self.training = True

    def _pre_process(self):
        resize = tf.map_fn(lambda frame: tf.image.resize_images(frame, (60, 80)), self.state_tensor)
        and_standardize = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), resize)
        self._preprocessed_state = and_standardize


    def _logging(self, location):
        self.summary_merge = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(location, self.tf_session.graph)
        self.last_loss = float('inf')

    def _storing(self, location, fail_on_empty=False):
        self.tf_saver = tf.train.Saver(filename='model', max_to_keep=2)

        self.tf_checkpoint = tf.train.latest_checkpoint(location)
        if self.tf_checkpoint:
            self.tf_saver.restore(self.tf_session, self.tf_checkpoint)
        else:
            if fail_on_empty:
                raise FileNotFoundError()
            else:
                self.tf_checkpoint = location + 'model'

    def close(self):
        self.tf_session.close()



class NeuralNetworkPolicy():

    def __init__(self, parametrization, optimizer=None, input_shape=(None, 120, 160, 3), output_shape=(None, 2),
                 batch_size=16, epochs=10, storage_location=None, training=True):

        self.parametrization = parametrization
        self.batch_size = batch_size
        self.epochs = epochs
        print(storage_location)
        if training:
            self.parametrization.prepare_for_train(input_shape, output_shape, optimizer, storage_location)
        else:
            self.parametrization.initialize(input_shape, output_shape, storage_location)

    def optimize(self, observations, expert_actions, episode):
        # indexes = np.arange(len(observations))
        # np.random.shuffle(indexes)

        observations = np.array(observations) # [indexes]
        print(observations.shape)
        actions = np.array(expert_actions) # [indexes]
        print(actions.shape)

        data_size = observations.shape[0]

        for _ in tqdm(range(self.epochs)):
            for iteration in range(0, data_size, self.batch_size):
                batch_observations = observations[iteration:iteration + self.batch_size]
                batch_actions = actions[iteration:iteration + self.batch_size]
                self.parametrization.train(batch_observations, batch_actions)
            if iteration + self.batch_size < data_size:
                remainder = data_size - iteration
                batch_observations = observations[iteration:remainder]
                batch_actions = actions[iteration:remainder]
                self.parametrization.train(batch_observations, batch_actions)

    def predict(self, observation, metadata):
        return self.parametrization.test([observation])

    def save(self):
        self.parametrization.commit()

    def close(self):
        self.parametrization.close()

