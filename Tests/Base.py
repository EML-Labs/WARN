
class BaseTest:
    shuffle_data: bool = True
    batch_size: int = 32

    def __init__(self, shuffle_data: bool = True, batch_size: int = 32):
        self.shuffle_data = shuffle_data
        self.batch_size = batch_size

    def setup_method(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def load_data(self):
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def load_model(self):
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def preprocess_data(self):
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def evaluate_model(self):
        raise NotImplementedError("This method should be overridden by subclasses.")
     
    def teardown_method(self):
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def run_test(self):
        raise NotImplementedError("This method should be overridden by subclasses.")