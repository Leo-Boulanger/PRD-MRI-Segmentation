class Configuration:
    def __init__(self, mri: str = None, out_dir: str = None,
                 nb_c: int = None, sp_rate: float = None,
                 q: float = None, p: float = None,
                 fuzz: float = None, thresh: float = None
                 ) -> None:
        self.mri_path = mri
        self.output_directory = out_dir
        self.nb_clusters = nb_c
        self.local_modifier = q
        self.global_modifier = p
        self.fuzzifier = fuzz
        self.threshold = thresh
        self.spatial_rate = sp_rate

    def __str__(self):
        return ("mri_path: " + str(self.mri_path) + '\n' +
                "output_directory: " + str(self.output_directory) + '\n' +
                "nb_clusters: " + str(self.nb_clusters) + '\n' +
                "local_modifier: " + str(self.local_modifier) + '\n' +
                "global_modifier: " + str(self.global_modifier) + '\n' +
                "fuzzifier: " + str(self.fuzzifier) + '\n' +
                "threshold: " + str(self.threshold) + '\n' +
                "spatial_rate: " + str(self.spatial_rate))

    @staticmethod
    def ask_user(text: str, expected_type: type, variable_name: str, min_value=None, max_value=None) -> object:
        """
        Ask the user for a value in the command line interface (CLI)
        :param text: the information text shown in the CLI
        :param expected_type: the expected type of the returned value
        :param variable_name: the name of the variable displayed if an error is raised
        :param min_value: the minimum value expected, will round up to this value if below
        :param max_value: the maximum value expected, will round down to this value if over
        :return: the final value
        """
        try:
            value = expected_type(input(text))
            if min_value: value = max(value, min_value)
            if max_value: value = min(value, max_value)
            return value
        except Exception:
            print(f"Error: The value of {variable_name} could not be converted to a {expected_type.__name__}.")
            raise

    def setup(self, config_file: str = None) -> None:
        """
        Load a configuration file if
        :param config_file: path the configuration file
        """
        # load the config file if it exists
        if config_file:
            self.load_settings(config_file)

        # check if every attribute is defined, and ask a value from the user otherwise
        if self.mri_path is None:
            self.mri_path = Configuration.ask_user(
                "Path to the MRI file. (Should be .nii or .nii.gz): ",
                str, "mri_path")
        if self.output_directory is None:
            self.output_directory = Configuration.ask_user(
                "Path to the output directory: ",
                str, "output_directory")
        if self.nb_clusters is None:
            self.nb_clusters = Configuration.ask_user(
                "Number of clusters. (Use '-1' to get the clusters automatically): ",
                int, "nb_clusters", min_value=-1, max_value=9999)
        if self.local_modifier is None:
            self.local_modifier = Configuration.ask_user(
                "Value of the local modifier (q): ",
                float, "local_modifier", min_value=0.0)
        if self.global_modifier is None:
            self.global_modifier = Configuration.ask_user(
                "Value of the global modifier (p): ",
                float, "global_modifier", min_value=0.0)
        if self.fuzzifier is None:
            self.fuzzifier = Configuration.ask_user(
                "Value of the fuzzifier: ",
                float, "fuzzifier", min_value=0.0)
        if self.threshold is None:
            self.threshold = Configuration.ask_user(
                "Value of the threshold: ",
                float, "threshold", min_value=0.01, max_value=1.0)
        if self.spatial_rate is None:
            self.spatial_rate = Configuration.ask_user(
                "Value of the spatial rate: ",
                float, "spatial_rate", min_value=0.0, max_value=1.0)
