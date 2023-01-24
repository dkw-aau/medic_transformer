import math

import neptune.new as neptune


class Logger:
    def __init__(self, config):
        """
        Logging class for logging information to different output channels
        Supports the stdout and neptune AI
        """

        self.run = None
        self.run_mode = 'async' if config.use_logging else 'debug'
        if config.use_logging:
            self.info("Using Neptune Logger Started")
        else:
            self.info("Neptune Logger Not Started, Logger Set To False")

        self.project_id = None
        self.api_token = None

        self.set_log_config(config)

    def set_log_config(self, config):
        self.project_id = config.neptune_project_id
        self.api_token = config.neptune_api_token

    def start_log(self):
        self.run = neptune.init_run(
            project=self.project_id,
            api_token=self.api_token,
            mode=self.run_mode
        )

    def stop_log(self):
        if self.run:
            self.run.stop()

    def log_running(self):
        return True if self.run else False

    def info(self, s):
        print(s)

    def log_value(self, name, value):
        self.run[name] = value

    def log_sequence(self, name, value):
        self.run[name].log(value)

    def log_values(self, metrics: dict, mode: str):
        for name, value in metrics.items():
            self.run[f'{mode}-{name}'] = float(value)

    def log_metrics(self, metrics: dict, mode: str):
        for name, value in metrics.items():
            self.run[f'{mode}/{name}'].log(float(value))

    def report_metrics(self, metrics: dict, mode: str):
        metric_string = f'{mode}: '
        for name, value in metrics.items():
            metric_string += f'{name} {round(value, 3)}, '
        print(metric_string)

    def metric2str(self, metric_dict):
        out = "[Evaluation metric]"
        for mode, score_dict in metric_dict.items():
            out += f"\tMode:{mode}, "
            for metric, score in score_dict.items():
                out += f"{metric}: {score:.4f}; "
        return out
                
    def feature_info(self, s):
        print('[Feature Transformation] ' + s)
