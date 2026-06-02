""" Configuration class for the project. """
from yaecs import Configuration
import os


class CVConfig(Configuration):
    """ Configuration class for a Template project. """

    @staticmethod
    def get_default_config_path():
        """ Returns the path to the default configuration file. """
        return os.path.join(os.path.dirname(__file__), "default", "_root_default.yaml")

    def parameters_pre_processing(self):
        """ Pre-processes the parameters. """
        return {}

    def parameters_post_processing(self):
        """ Post-processes the parameters. """
        return {}
 