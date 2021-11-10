import tensorflow as tf

class DataModel:
    def __init__(self):
        if type(self) == DataModel:
            raise Exception("<DataModel> must be subclassed.")

        self.__train_images = None
        self.__train_labels = None
        self.__test_images = None
        self.__test_labels = None
        self.model = None
        self.__name = ""
        self.__verbose = 2
        self.__use_ds = False

    def print_summary(self):
        self.model.summary()

    def get_name(self):
        raise NotImplementedError

    def get_plain_name(self):
        return self.__name

    def set_name(self, name):
        self.__name = name

    def get_verbose(self):
        return self.__verbose

    def set_verbose(self, verbose):
        self.__verbose = verbose

    def get_train_data(self):
        return (self.__train_images, self.__train_labels)

    def get_test_data(self):
        return (self.__test_images, self.__test_labels)

    def get_data(self):
        return (self.get_train_data(), self.get_test_data())

    def set_train_data(self, train_images, train_labels):
        self.__train_images = train_images
        self.__train_labels = train_labels

        if self.__use_ds:
            self.__train_ds =  self.prep_ds(self.__train_images, self.__train_labels)

    def set_test_data(self, test_images, test_labels):
        self.__test_images = test_images
        self.__test_labels = test_labels

        if self.__use_ds:
            self.__test_ds =  self.prep_ds(self.__test_images, self.__test_labels)

    def get_use_ds(self):
        return self.__use_ds

    def set_use_ds(self):
        self.__use_ds = True
        self.__train_ds = self.prep_ds(self.__train_images, self.__train_labels)
        self.__test_ds = self.prep_ds(self.__test_images, self.__test_labels)

    def prep_ds(self, ds_images, ds_labels):
        return self._proc_ds(tf.data.Dataset.from_tensor_slices((ds_images, ds_labels)))

    def train(self):
        if self.__use_ds:
            test_loss, test_acc = self._train_model_ds(
                self.__train_ds
            )
        else:
            test_loss, test_acc = self._train_model(
                self.__train_images,
                self.__train_labels,
            )
        print("Accuracy before faults for", self.get_name(), "is:", test_acc)
        self.checkpoint_save()
        return test_loss, test_acc

    def train_fi(self, train_images, train_labels):
        if self.__use_ds:
            train_ds = self.prep_ds(train_images, train_labels)
            test_loss, test_acc = self._train_model_ds(train_ds)
        else:
            test_loss, test_acc = self._train_model(
                train_images, train_labels
            )
        verbose = self.get_verbose()
        if verbose:
            print("Accuracy after faults for",
                  self.get_name(), "is:", test_acc)
        return test_loss, test_acc

    def test(self):
        if self.__use_ds:
            return self.__evaluate_ds(self.__test_ds)
        else:
            return self.evaluate(self.__test_images, self.__test_labels)

    def evaluate(self, test_images, test_labels):
        if self.__use_ds:
            test_ds = self.prep_ds(test_images, test_labels)
            return self.model.evaluate(test_ds, verbose=0)
        else:
            return self.model.evaluate(test_images, test_labels, verbose=0)

    def __evaluate_ds(self, test_ds):
        return self.model.evaluate(test_ds, verbose=0)

    def predict(self, test_images=None):
        if self.__use_ds:
            if test_images is None:
                return self.model.predict(self.__test_ds, verbose=0)
            else:
                test_ds = self.prep_ds(test_images, None)
                return self.model.predict(test_ds, verbose=0)
        else:
            if test_images is None:
                test_images = self.__test_images
            return self.model.predict(test_images, verbose=0)

    def checkpoint_save(self):
        h5_file = "./h5/" + self.get_name() + "-trained.h5"
        self.model.save_weights(h5_file)
        print("Checkpoint saved at:", h5_file)

    def checkpoint_load(self):
        h5_file = "./h5/" + self.get_name() + "-trained.h5"
        #self.model.load_weights(h5_file)
        try:
            self.model.load_weights(h5_file)
            return True
        except:
            print("Unable to find file", h5_file)
            pass
        return False

    def create_untrained(self):
        h5_file = "./h5/" + self.get_name() + "-untrained.h5"
        self.model.save_weights(h5_file)
        print("Empty untrained saved at:", h5_file)

    def clear_weights(self):
        h5_file = "./h5/" + self.get_name() + "-untrained.h5"
        try:
            self.model.load_weights(h5_file)
            return True
        except:
            print("Unable to find file", h5_file)
            pass
        return False

    # Override this for classifier-specific image preprocessing
    def _process_images(self, images):
        return images

    def _proc_ds(self, ds):
        return ds

    def _train_model(
            self,
            train_images,
            train_labels):
        raise NotImplementedError

    def _train_model_ds(
            self,
            train_ds):
        raise NotImplementedError

